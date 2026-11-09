use burn::module::Module;
use burn::nn::Linear;
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Device, Tensor};

use crate::model::normalization::RmsNormalization;
use crate::model::rotary::Rotation;
use crate::model::{FEATURE_DIMENSION, HEAD_COUNT, HEAD_DIMENSION, linear};

/// フィードフォワード層の隠れ層の次元数を特徴量の次元数の何倍にするか。
const FEED_FORWARD_MULTIPLIER: usize = 4;

/// ヘッドごとにゲートを掛けるアテンション。
#[derive(Module, Debug)]
pub struct Attention {
    /// 入力の正規化。
    pub normalization: RmsNormalization,
    /// クエリ、キー、バリューへの射影。
    pub to_query_key_value: Linear,
    /// ヘッドごとのゲートへの射影。
    pub to_gates: Linear,
    /// 出力への射影。
    pub to_out: Linear,
}

impl Attention {
    /// 層を作成する。
    ///
    /// # 引数
    ///
    /// - `device`: パラメーターを配置するデバイス
    fn new(device: &Device) -> Self {
        let inner_dimension = HEAD_COUNT * HEAD_DIMENSION;
        Self {
            normalization: RmsNormalization::new(FEATURE_DIMENSION, device),
            to_query_key_value: linear(FEATURE_DIMENSION, inner_dimension * 3, false, device),
            to_gates: linear(FEATURE_DIMENSION, HEAD_COUNT, true, device),
            to_out: linear(inner_dimension, FEATURE_DIMENSION, false, device),
        }
    }

    /// 入力にセルフアテンションを適用する。
    ///
    /// # 引数
    ///
    /// - `input`: `[バッチ, 系列, 特徴量]` の入力
    /// - `rotation`: 系列方向の回転
    pub fn forward(&self, input: Tensor<3>, rotation: &Rotation) -> Tensor<3> {
        let [batch, sequence, _] = input.dims();
        let normalized = self.normalization.forward(input);

        let [query, key, value]: [Tensor<4>; 3] = self
            .to_query_key_value
            .forward(normalized.clone())
            .chunk(3, 2)
            .into_iter()
            .map(|part| {
                part.reshape([batch, sequence, HEAD_COUNT, HEAD_DIMENSION])
                    .swap_dims(1, 2)
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("the projection is split into three parts");

        // `AttentionModuleOptions` の `scale` に `Some` を指定すると cubecl がスコア行列を確保する
        // 低速な経路へ切り替わるので、`1 / √d` を使用する既定値の `None` のままとする。
        let attended = attention(
            rotation.apply(query),
            rotation.apply(key),
            value,
            None,
            None,
            AttentionModuleOptions::default(),
        );

        let gates = sigmoid(
            self.to_gates
                .forward(normalized)
                .swap_dims(1, 2)
                .unsqueeze_dim::<4>(3),
        );
        let output = (attended * gates).swap_dims(1, 2).reshape([
            batch,
            sequence,
            HEAD_COUNT * HEAD_DIMENSION,
        ]);
        self.to_out.forward(output)
    }
}

/// 正規化と GELU を使用するフィードフォワード層。
#[derive(Module, Debug)]
pub struct FeedForward {
    /// 入力の正規化。
    pub normalization: RmsNormalization,
    /// 隠れ層への線形層。
    pub input: Linear,
    /// 出力への線形層。
    pub output: Linear,
}

impl FeedForward {
    /// 層を作成する。
    ///
    /// # 引数
    ///
    /// - `device`: パラメーターを配置するデバイス
    fn new(device: &Device) -> Self {
        let inner_dimension = FEATURE_DIMENSION * FEED_FORWARD_MULTIPLIER;
        Self {
            normalization: RmsNormalization::new(FEATURE_DIMENSION, device),
            input: linear(FEATURE_DIMENSION, inner_dimension, true, device),
            output: linear(inner_dimension, FEATURE_DIMENSION, true, device),
        }
    }

    /// 入力に層を適用する。
    ///
    /// # 引数
    ///
    /// - `input`: 最終次元が特徴量のテンソル
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        self.output
            .forward(gelu(self.input.forward(self.normalization.forward(input))))
    }
}

/// アテンションとフィードフォワード層の組。
#[derive(Module, Debug)]
pub struct TransformerLayer {
    /// アテンション。
    pub attention: Attention,
    /// フィードフォワード層。
    pub feed_forward: FeedForward,
}

impl TransformerLayer {
    /// 層を作成する。
    ///
    /// # 引数
    ///
    /// - `device`: パラメーターを配置するデバイス
    pub fn new(device: &Device) -> Self {
        Self {
            attention: Attention::new(device),
            feed_forward: FeedForward::new(device),
        }
    }

    /// 残差接続で層を適用する。
    ///
    /// # 引数
    ///
    /// - `input`: `[バッチ, 系列, 特徴量]` の入力
    /// - `rotation`: 系列方向の回転
    pub fn forward(&self, input: Tensor<3>, rotation: &Rotation) -> Tensor<3> {
        let attended = self.attention.forward(input.clone(), rotation) + input;
        self.feed_forward.forward(attended.clone()) + attended
    }
}
