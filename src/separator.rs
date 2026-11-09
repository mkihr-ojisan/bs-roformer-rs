use std::path::Path;
use std::sync::Mutex;

use burn::store::{KeyRemapper, ModuleSnapshot, PytorchStore};
use burn::tensor::{Device, Tensor, TensorData};
use realfft::num_complex::Complex;

use crate::demix::demix;
use crate::error::{Error, Result};
use crate::model::BsRoformer;
use crate::stem::{SeparatedStem, Stem};
use crate::stft::{
    CHANNEL_COUNT, FFT_SIZE, InverseWorkspace, SPECTROGRAM_WIDTH, Spectrogram, Stft,
};

/// 既定のチャンク長（サンプル数）。
pub const DEFAULT_CHUNK_SIZE: usize = 441_000;

/// 指定できるチャンク長の下限（サンプル数）。
pub const MINIMUM_CHUNK_SIZE: usize = FFT_SIZE;

/// BS-RoFormer による音源分離器。
#[derive(Debug)]
pub struct Separator {
    /// マスク推定モデル。
    model: BsRoformer,
    /// 短時間フーリエ変換器。
    stft: Stft,
    /// 1 回の推論に渡すサンプル数。
    chunk_size: usize,
    /// 推論を実行するデバイス。
    device: Device,
}

impl Separator {
    /// チェックポイントからモデルを読み込む。
    ///
    /// # 引数
    ///
    /// - `checkpoint`: BS-Rofo-SW-Fixed の `.ckpt` ファイルのパス
    /// - `chunk_size`: 1 回の推論に渡すサンプル数。[`MINIMUM_CHUNK_SIZE`] 以上。
    /// - `device`: 推論を実行するデバイス
    pub fn load(checkpoint: impl AsRef<Path>, chunk_size: usize, device: Device) -> Result<Self> {
        if chunk_size < MINIMUM_CHUNK_SIZE {
            return Err(Error::ChunkSizeTooShort {
                chunk_size,
                minimum: MINIMUM_CHUNK_SIZE,
            });
        }

        let mut model = BsRoformer::new(&device);
        let mut store = PytorchStore::from_file(checkpoint.as_ref())
            .map_indices_contiguous(false)
            .remap(checkpoint_key_remapper());
        let result = model.load_from(&mut store)?;

        // 回転位置埋め込みの周波数は全層で同じ値なので、未使用の周波数のキーをエラーの対象から
        // 除外する。
        let unexpected_keys: Vec<_> = result
            .unused
            .into_iter()
            .filter(|key| !key.ends_with(".rotary_embed.freqs"))
            .collect();
        if !unexpected_keys.is_empty() {
            return Err(Error::UnexpectedCheckpointKeys { unexpected_keys });
        }

        Ok(Self {
            model,
            stft: Stft::new(),
            chunk_size,
            device,
        })
    }

    /// 44.1 kHz ステレオの音声をパートに分離する。
    ///
    /// # 引数
    ///
    /// - `mix`: チャンネルごとのサンプル列。2 チャンネルの長さは一致していなければならない。
    ///
    /// # 戻り値
    ///
    /// [`Stem::ALL`] の順序で、入力と同じ長さのパートを返す。
    pub fn separate(&self, mix: &[Vec<f32>; CHANNEL_COUNT]) -> Result<Vec<SeparatedStem>> {
        assert_eq!(
            mix[0].len(),
            mix[1].len(),
            "both channels must have the same length"
        );

        let stems = demix(mix, self.chunk_size, Stem::ALL.len(), |chunk, output| {
            self.separate_chunk(chunk, output)
        })?;
        Ok(Stem::ALL
            .into_iter()
            .zip(stems)
            .map(|(stem, samples)| SeparatedStem { stem, samples })
            .collect())
    }

    /// 1 チャンクをモデルで処理し、パートごとの波形を `output` へ出力する。
    ///
    /// # 引数
    ///
    /// - `chunk`: チャンネルごとの `chunk_size` サンプルの波形
    /// - `output`: `[パート, チャンネル, サンプル]` の順の出力先。長さは
    ///   `Stem::ALL.len() * CHANNEL_COUNT * chunk_size` であること。
    fn separate_chunk(&self, chunk: &[Vec<f32>; CHANNEL_COUNT], output: &mut [f32]) -> Result<()> {
        let spectrogram = self.stft.forward(chunk);
        let input = Tensor::from_data(
            TensorData::new(
                spectrogram.values.clone(),
                [1, spectrogram.frame_count, SPECTROGRAM_WIDTH],
            ),
            &self.device,
        );
        let masks = self.model.forward(input).try_into_vec_as::<f32>()?;

        self.synthesize(&spectrogram, &masks, output);
        Ok(())
    }

    /// スペクトログラムにマスクを掛け、パートとチャンネルごとの波形を合成する。
    ///
    /// # 引数
    ///
    /// - `spectrogram`: スペクトログラム
    /// - `masks`: `[パート, フレーム, 周波数ビン, チャンネル, 実部と虚部]` の順のマスク
    /// - `output`: `[パート, チャンネル, サンプル]` の順の出力先。長さは
    ///   `Stem::ALL.len() * CHANNEL_COUNT * chunk_size` であること。
    fn synthesize(&self, spectrogram: &Spectrogram, masks: &[f32], output: &mut [f32]) {
        let stem_size = spectrogram.frame_count * SPECTROGRAM_WIDTH;
        let tasks = Mutex::new(output.chunks_mut(self.chunk_size).enumerate());
        let worker_count = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(Stem::ALL.len() * CHANNEL_COUNT);

        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                scope.spawn(|| {
                    let mut workspace = InverseWorkspace::new(&self.stft, spectrogram.frame_count);
                    loop {
                        let task = tasks.lock().expect("no worker panics").next();
                        let Some((index, samples)) = task else { break };
                        let (stem, channel) = (index / CHANNEL_COUNT, index % CHANNEL_COUNT);
                        let mask = &masks[stem * stem_size..(stem + 1) * stem_size];
                        self.stft.inverse(
                            |frame, bins| {
                                apply_mask(frame, channel, &spectrogram.values, mask, bins)
                            },
                            samples,
                            &mut workspace,
                        );
                    }
                });
            }
        });
    }
}

/// 1 フレーム分のスペクトルにマスクを掛け、直流成分を 0 にする。
///
/// # 引数
///
/// - `frame`: フレームの番号
/// - `channel`: チャンネルの番号
/// - `spectrogram`: `[フレーム, 周波数ビン, チャンネル, 実部と虚部]` の順のスペクトログラム
/// - `mask`: 同じ順の 1 パート分のマスク
/// - `bins`: 出力先の周波数ビンの列
fn apply_mask(
    frame: usize,
    channel: usize,
    spectrogram: &[f32],
    mask: &[f32],
    bins: &mut [Complex<f32>],
) {
    let start = frame * SPECTROGRAM_WIDTH + channel * 2;
    for (bin, value) in bins.iter_mut().enumerate() {
        let offset = start + bin * CHANNEL_COUNT * 2;
        let source = Complex::new(spectrogram[offset], spectrogram[offset + 1]);
        *value = source * Complex::new(mask[offset], mask[offset + 1]);
    }
    bins[0] = Complex::new(0.0, 0.0);
}

/// チェックポイントのキーをモデルのパラメーターのパスへ変換する規則を作成する。
fn checkpoint_key_remapper() -> KeyRemapper {
    let mut patterns = vec![
        (
            r"^final_norm\.".to_string(),
            "final_normalization.".to_string(),
        ),
        (
            r"^layers\.0\.0\.layers\.0\.0\.rotary_embed\.freqs$".to_string(),
            "time_rotary.frequencies".to_string(),
        ),
        (
            r"^layers\.0\.1\.layers\.0\.0\.rotary_embed\.freqs$".to_string(),
            "frequency_rotary.frequencies".to_string(),
        ),
        (
            r"^band_split\.to_features\.(\d+)\.0\.".to_string(),
            "band_split.bands.$1.normalization.".to_string(),
        ),
        (
            r"^band_split\.to_features\.(\d+)\.1\.".to_string(),
            "band_split.bands.$1.linear.".to_string(),
        ),
        (
            r"^mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.0\.".to_string(),
            "mask_estimators.$1.bands.$2.input.".to_string(),
        ),
        (
            r"^mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.2\.".to_string(),
            "mask_estimators.$1.bands.$2.output.".to_string(),
        ),
    ];
    for (axis_index, axis) in ["time", "frequency"].iter().enumerate() {
        let block = format!(r"^layers\.(\d+)\.{axis_index}\.layers\.0");
        for (from, to) in [
            (r"\.0\.norm\.", "attention.normalization."),
            (r"\.0\.to_qkv\.", "attention.to_query_key_value."),
            (r"\.0\.to_gates\.", "attention.to_gates."),
            (r"\.0\.to_out\.0\.", "attention.to_out."),
            (r"\.1\.net\.0\.", "feed_forward.normalization."),
            (r"\.1\.net\.1\.", "feed_forward.input."),
            (r"\.1\.net\.4\.", "feed_forward.output."),
        ] {
            patterns.push((format!("{block}{from}"), format!("blocks.$1.{axis}.{to}")));
        }
    }
    KeyRemapper::from_patterns(patterns).expect("the key remapping patterns are valid")
}
