use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

use crate::model::attention::TransformerLayer;
use crate::model::band::{BandSplit, MaskEstimator};
use crate::model::normalization::RmsNormalization;
use crate::model::rotary::{RotaryEmbedding, Rotation};
use crate::stem::Stem;

pub mod attention;
pub mod band;
pub mod normalization;
pub mod rotary;

/// 特徴量の次元数。
pub const FEATURE_DIMENSION: usize = 256;

/// Transformer のブロック数。
const BLOCK_COUNT: usize = 12;

/// アテンションのヘッド数。
pub const HEAD_COUNT: usize = 8;

/// アテンションのヘッドあたりの次元数。
pub const HEAD_DIMENSION: usize = 64;

/// 帯域ごとの周波数ビン数。
pub const BAND_BIN_COUNTS: [usize; 62] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48,
    48, 48, 48, 128, 129,
];

/// 重みを 0 で初期化した線形層を作成する。
///
/// # 引数
///
/// - `input`: 入力の次元数
/// - `output`: 出力の次元数
/// - `bias`: バイアス項を持つかどうか
/// - `device`: パラメーターを配置するデバイス
fn linear(input: usize, output: usize, bias: bool, device: &Device) -> Linear {
    LinearConfig::new(input, output)
        .with_bias(bias)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

/// 時間方向と周波数方向の Transformer の組。
#[derive(Module, Debug)]
pub struct AxialBlock {
    /// 時間方向の Transformer。
    pub time: TransformerLayer,
    /// 周波数方向の Transformer。
    pub frequency: TransformerLayer,
}

impl AxialBlock {
    /// 時間方向、周波数方向の順に Transformer を適用する。
    ///
    /// # 引数
    ///
    /// - `features`: `[バッチ, フレーム, 帯域, 特徴量]` の特徴量
    /// - `time_rotation`: 時間方向の回転
    /// - `frequency_rotation`: 周波数方向の回転
    pub fn forward(
        &self,
        features: Tensor<4>,
        time_rotation: &Rotation,
        frequency_rotation: &Rotation,
    ) -> Tensor<4> {
        let [batch, frames, bands, dimension] = features.dims();
        let time_output = self.time.forward(
            features
                .swap_dims(1, 2)
                .reshape([batch * bands, frames, dimension]),
            time_rotation,
        );
        let frequency_input = time_output
            .reshape([batch, bands, frames, dimension])
            .swap_dims(1, 2)
            .reshape([batch * frames, bands, dimension]);
        self.frequency
            .forward(frequency_input, frequency_rotation)
            .reshape([batch, frames, bands, dimension])
    }
}

/// BS-RoFormer のマスク推定モデル。
#[derive(Module, Debug)]
pub struct BsRoformer {
    /// 帯域分割。
    pub band_split: BandSplit,
    /// 時間方向の位置埋め込み。
    pub time_rotary: RotaryEmbedding,
    /// 周波数方向の位置埋め込み。
    pub frequency_rotary: RotaryEmbedding,
    /// Transformer のブロック。
    pub blocks: Vec<AxialBlock>,
    /// マスク推定器の直前の正規化。
    pub final_normalization: RmsNormalization,
    /// パートごとのマスク推定器。
    pub mask_estimators: Vec<MaskEstimator>,
}

impl BsRoformer {
    /// パラメーターが初期値のモデルを作成する。
    ///
    /// # 引数
    ///
    /// - `device`: パラメーターを配置するデバイス
    pub fn new(device: &Device) -> Self {
        Self {
            band_split: BandSplit::new(device),
            time_rotary: RotaryEmbedding::new(device),
            frequency_rotary: RotaryEmbedding::new(device),
            blocks: (0..BLOCK_COUNT)
                .map(|_| AxialBlock {
                    time: TransformerLayer::new(device),
                    frequency: TransformerLayer::new(device),
                })
                .collect(),
            final_normalization: RmsNormalization::new(FEATURE_DIMENSION, device),
            mask_estimators: (0..Stem::ALL.len())
                .map(|_| MaskEstimator::new(device))
                .collect(),
        }
    }

    /// スペクトログラムからパートごとのマスクを推定する。
    ///
    /// # 引数
    ///
    /// - `spectrogram`: `[バッチ, フレーム, SPECTROGRAM_WIDTH]` のスペクトログラム
    ///
    /// # 戻り値
    ///
    /// `[バッチ, パート, フレーム, SPECTROGRAM_WIDTH]` のマスクを返す。
    pub fn forward(&self, spectrogram: Tensor<3>) -> Tensor<4> {
        let [_, frames, _] = spectrogram.dims();
        let device = spectrogram.device();
        let time_rotation = self.time_rotary.rotation(frames, &device);
        let frequency_rotation = self
            .frequency_rotary
            .rotation(BAND_BIN_COUNTS.len(), &device);

        let mut features = self.band_split.forward(spectrogram);
        for block in &self.blocks {
            features = block.forward(features, &time_rotation, &frequency_rotation);
        }
        let features = self.final_normalization.forward(features);

        Tensor::stack(
            self.mask_estimators
                .iter()
                .map(|estimator| estimator.forward(features.clone()))
                .collect(),
            1,
        )
    }
}
