use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{Device, Tensor};

/// L2 正規化した値に学習済みのスケールを掛ける層。
#[derive(Module, Debug)]
pub struct RmsNormalization {
    /// 次元ごとのスケール。
    pub gamma: Param<Tensor<1>>,
}

impl RmsNormalization {
    /// 指定された次元数の層を作成する。
    ///
    /// # 引数
    ///
    /// - `dimension`: 最終次元の大きさ
    /// - `device`: パラメーターを配置するデバイス
    pub fn new(dimension: usize, device: &Device) -> Self {
        Self {
            gamma: Initializer::Ones.init([dimension], device),
        }
    }

    /// 最終次元を L2 正規化し、次元数の平方根とスケールを掛ける。
    ///
    /// # 引数
    ///
    /// - `input`: 最終次元の大きさが [`RmsNormalization::new`] の `dimension` と一致するテンソル
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let scale = (self.gamma.dims()[0] as f32).sqrt();
        let length = input
            .clone()
            .powi_scalar(2)
            .sum_dim(D - 1)
            .sqrt()
            .clamp_min(1e-12);
        input / length * self.gamma.val().unsqueeze::<D>() * scale
    }
}
