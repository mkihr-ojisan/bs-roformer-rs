use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation::sigmoid;
use burn::tensor::{Device, Tensor};

use crate::model::normalization::RmsNormalization;
use crate::model::{BAND_BIN_COUNTS, FEATURE_DIMENSION, linear};
use crate::stft::CHANNEL_COUNT;

/// マスク推定器の隠れ層の次元数を特徴量の次元数の何倍にするか。
const MASK_HIDDEN_MULTIPLIER: usize = 4;

/// 帯域あたりの入力の次元数を返す。
///
/// # 引数
///
/// - `bins`: 帯域の周波数ビン数
fn band_input_dimension(bins: usize) -> usize {
    bins * CHANNEL_COUNT * 2
}

/// 1 帯域の特徴量を計算する層。
#[derive(Module, Debug)]
pub struct BandSplitLayer {
    /// 入力の正規化。
    pub normalization: RmsNormalization,
    /// 特徴量への線形層。
    pub linear: Linear,
}

/// スペクトログラムを帯域ごとの特徴量に変換する層。
#[derive(Module, Debug)]
pub struct BandSplit {
    /// 帯域ごとの層。
    pub bands: Vec<BandSplitLayer>,
}

impl BandSplit {
    /// 層を作成する。
    ///
    /// # 引数
    ///
    /// - `device`: パラメーターを配置するデバイス
    pub fn new(device: &Device) -> Self {
        let bands = BAND_BIN_COUNTS
            .iter()
            .map(|bins| {
                let dimension = band_input_dimension(*bins);
                BandSplitLayer {
                    normalization: RmsNormalization::new(dimension, device),
                    linear: linear(dimension, FEATURE_DIMENSION, true, device),
                }
            })
            .collect();
        Self { bands }
    }

    /// スペクトログラムを帯域ごとの特徴量に変換する。
    ///
    /// # 引数
    ///
    /// - `spectrogram`: `[バッチ, フレーム, 値]` のスペクトログラム
    ///
    /// # 戻り値
    ///
    /// `[バッチ, フレーム, 帯域, 特徴量]` の特徴量を返す。
    pub fn forward(&self, spectrogram: Tensor<3>) -> Tensor<4> {
        let sizes = BAND_BIN_COUNTS
            .iter()
            .map(|bins| band_input_dimension(*bins))
            .collect();
        let outputs = spectrogram
            .split_with_sizes(sizes, 2)
            .into_iter()
            .zip(&self.bands)
            .map(|(band, layer)| layer.linear.forward(layer.normalization.forward(band)))
            .collect();
        Tensor::stack(outputs, 2)
    }
}

/// 1 帯域のマスクを推定する層。
#[derive(Module, Debug)]
pub struct MaskBandLayer {
    /// 隠れ層への線形層。
    pub input: Linear,
    /// 出力への線形層。
    pub output: Linear,
}

/// 帯域ごとの特徴量から 1 パートのマスクを推定する層。
#[derive(Module, Debug)]
pub struct MaskEstimator {
    /// 帯域ごとの層。
    pub bands: Vec<MaskBandLayer>,
}

impl MaskEstimator {
    /// 層を作成する。
    ///
    /// # 引数
    ///
    /// - `device`: パラメーターを配置するデバイス
    pub fn new(device: &Device) -> Self {
        let hidden_dimension = FEATURE_DIMENSION * MASK_HIDDEN_MULTIPLIER;
        let bands = BAND_BIN_COUNTS
            .iter()
            .map(|bins| MaskBandLayer {
                input: linear(FEATURE_DIMENSION, hidden_dimension, true, device),
                output: linear(
                    hidden_dimension,
                    band_input_dimension(*bins) * 2,
                    true,
                    device,
                ),
            })
            .collect();
        Self { bands }
    }

    /// 特徴量から 1 パートのマスクを推定する。
    ///
    /// # 引数
    ///
    /// - `features`: `[バッチ, フレーム, 帯域, 特徴量]` の特徴量
    ///
    /// # 戻り値
    ///
    /// `[バッチ, フレーム, 値]` のマスクを返す。
    pub fn forward(&self, features: Tensor<4>) -> Tensor<3> {
        let outputs = features
            .iter_dim(2)
            .map(|band| band.squeeze_dim::<3>(2))
            .zip(&self.bands)
            .map(|(band, layer)| {
                let hidden = layer.input.forward(band).tanh();
                gated_linear_unit(layer.output.forward(hidden))
            })
            .collect();
        Tensor::cat(outputs, 2)
    }
}

/// 最終次元を二等分し、前半に後半のシグモイドを掛ける。
///
/// # 引数
///
/// - `input`: 最終次元の大きさが偶数のテンソル
fn gated_linear_unit(input: Tensor<3>) -> Tensor<3> {
    let dimension = input.dims()[2];
    let [values, gates]: [Tensor<3>; 2] = input
        .split(dimension / 2, 2)
        .try_into()
        .expect("the tensor is split in two");
    values * sigmoid(gates)
}
