use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

#[cfg(test)]
mod tests;

/// FFT のサイズ。
pub const FFT_SIZE: usize = 2048;

/// フレームの間隔（サンプル数）。
const HOP_LENGTH: usize = 512;

/// 周波数ビン数。
const BIN_COUNT: usize = FFT_SIZE / 2 + 1;

/// 音声のチャンネル数。
pub const CHANNEL_COUNT: usize = 2;

/// スペクトログラム 1 フレームあたりの値の個数（周波数ビン、チャンネル、実部と虚部）。
pub const SPECTROGRAM_WIDTH: usize = BIN_COUNT * CHANNEL_COUNT * 2;

/// ステレオのスペクトログラム。
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrogram {
    /// フレーム数。
    pub frame_count: usize,
    /// `[フレーム, 周波数ビン, チャンネル, 実部と虚部]` の順に並べた値。
    pub values: Vec<f32>,
}

/// 短時間フーリエ変換器。
pub struct Stft {
    /// 周期的な Hann 窓。
    window: Vec<f32>,
    /// 順方向の FFT の実行計画。
    forward: Arc<dyn RealToComplex<f32>>,
    /// 逆方向の FFT の実行計画。
    inverse: Arc<dyn ComplexToReal<f32>>,
}

impl std::fmt::Debug for Stft {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Stft").finish_non_exhaustive()
    }
}

impl Stft {
    /// 周期的な Hann 窓を使用する変換器を作成する。
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::new();
        let window = (0..FFT_SIZE)
            .map(|index| {
                let phase = 2.0 * std::f64::consts::PI * index as f64 / FFT_SIZE as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect();
        Self {
            window,
            forward: planner.plan_fft_forward(FFT_SIZE),
            inverse: planner.plan_fft_inverse(FFT_SIZE),
        }
    }

    /// ステレオの波形を短時間フーリエ変換する。
    ///
    /// # 引数
    ///
    /// - `channels`: チャンネルごとの波形。長さは一致し、`FFT_SIZE / 2` より長いこと。
    pub fn forward(&self, channels: &[Vec<f32>; CHANNEL_COUNT]) -> Spectrogram {
        let length = channels[0].len();
        assert_eq!(length, channels[1].len(), "channel lengths must match");
        assert!(
            length > FFT_SIZE / 2,
            "the signal is too short for reflect padding"
        );

        let frame_count = length / HOP_LENGTH + 1;
        let mut values = vec![0.0; frame_count * SPECTROGRAM_WIDTH];

        let mut input = vec![0.0; FFT_SIZE];
        let mut output = vec![Complex::new(0.0, 0.0); BIN_COUNT];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.forward.get_scratch_len()];

        for (channel, signal) in channels.iter().enumerate() {
            // フレームの中心を信号の先頭に一致させるため、両端を反射でパディングする。
            let padded = reflect_padded(signal, FFT_SIZE / 2);
            for frame in 0..frame_count {
                let start = frame * HOP_LENGTH;
                for (target, (sample, weight)) in input
                    .iter_mut()
                    .zip(padded[start..start + FFT_SIZE].iter().zip(&self.window))
                {
                    *target = sample * weight;
                }
                self.forward
                    .process_with_scratch(&mut input, &mut output, &mut scratch)
                    .expect("the buffer sizes are valid");

                let frame_values =
                    &mut values[frame * SPECTROGRAM_WIDTH..(frame + 1) * SPECTROGRAM_WIDTH];
                for (bin, value) in output.iter().enumerate() {
                    let offset = bin * CHANNEL_COUNT * 2 + channel * 2;
                    frame_values[offset] = value.re;
                    frame_values[offset + 1] = value.im;
                }
            }
        }

        Spectrogram {
            frame_count,
            values,
        }
    }

    /// スペクトルの列を短時間フーリエ逆変換して波形を計算する。
    ///
    /// # 引数
    ///
    /// - `spectrum_of`: フレームの番号と周波数ビンの列を受け取り、列を埋める関数
    /// - `output`: 出力先。長さは元の信号と同じであること。
    /// - `workspace`: スペクトルの列と同じフレーム数で作成した作業領域
    pub fn inverse(
        &self,
        mut spectrum_of: impl FnMut(usize, &mut [Complex<f32>]),
        output: &mut [f32],
        workspace: &mut InverseWorkspace,
    ) {
        workspace.overlap.fill(0.0);
        for frame in 0..workspace.frame_count {
            spectrum_of(frame, &mut workspace.frequency);
            // realfft は直流成分とナイキスト周波数の虚部が 0 でないとエラーを返す。
            workspace.frequency[0].im = 0.0;
            workspace.frequency[BIN_COUNT - 1].im = 0.0;
            self.inverse
                .process_with_scratch(
                    &mut workspace.frequency,
                    &mut workspace.time,
                    &mut workspace.scratch,
                )
                .expect("the buffer sizes and the imaginary parts of DC and Nyquist are valid");

            // realfft の逆変換は正規化されず `FFT_SIZE` 倍の値を返すので、窓を掛けた値を
            // `FFT_SIZE` で除算してオーバーラップ加算する。
            let start = frame * HOP_LENGTH;
            for (index, (sample, weight)) in workspace.time.iter().zip(&self.window).enumerate() {
                workspace.overlap[start + index] += sample * weight / FFT_SIZE as f32;
            }
        }

        // 窓の二乗の和で正規化し、両端のパディングの範囲を除去する。
        let trimmed = FFT_SIZE / 2;
        for (index, value) in output.iter_mut().enumerate() {
            *value = workspace.overlap[trimmed + index] / workspace.envelope[trimmed + index];
        }
    }
}

/// [`Stft::inverse`] の作業領域。
pub struct InverseWorkspace {
    /// フレーム数。
    frame_count: usize,
    /// 窓を掛けたフレームをオーバーラップ加算した結果。
    overlap: Vec<f32>,
    /// 窓の二乗をオーバーラップ加算した結果。
    envelope: Vec<f32>,
    /// 1 フレームのスペクトル。
    frequency: Vec<Complex<f32>>,
    /// 1 フレームの逆 FFT の結果。
    time: Vec<f32>,
    /// 逆 FFT の作業領域。
    scratch: Vec<Complex<f32>>,
}

impl InverseWorkspace {
    /// `frame_count` フレーム分の作業領域を確保する。
    ///
    /// # 引数
    ///
    /// - `stft`: 使用する変換器
    /// - `frame_count`: フレーム数
    pub fn new(stft: &Stft, frame_count: usize) -> Self {
        let length = FFT_SIZE + HOP_LENGTH * (frame_count - 1);
        let mut envelope = vec![0.0; length];
        for frame in 0..frame_count {
            for (index, weight) in stft.window.iter().enumerate() {
                envelope[frame * HOP_LENGTH + index] += weight * weight;
            }
        }
        Self {
            frame_count,
            overlap: vec![0.0; length],
            envelope,
            frequency: vec![Complex::new(0.0, 0.0); BIN_COUNT],
            time: vec![0.0; FFT_SIZE],
            scratch: vec![Complex::new(0.0, 0.0); stft.inverse.get_scratch_len()],
        }
    }
}

/// 端のサンプルを含めない反射で信号の両端をパディングした列を返す。
///
/// # 引数
///
/// - `signal`: 元の信号
/// - `padding`: 両端に追加するサンプル数。`signal.len()` 未満。
pub fn reflect_padded(signal: &[f32], padding: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(signal.len() + 2 * padding);
    padded.extend((1..=padding).rev().map(|offset| signal[offset]));
    padded.extend_from_slice(signal);
    padded.extend((1..=padding).map(|offset| signal[signal.len() - 1 - offset]));
    padded
}
