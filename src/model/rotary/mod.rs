use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::model::HEAD_DIMENSION;

#[cfg(test)]
mod tests;

/// 回転の周波数の個数。
const FREQUENCY_COUNT: usize = HEAD_DIMENSION / 2;

/// 位置と周波数の積を回転角とする回転位置埋め込み。
#[derive(Module, Debug)]
pub struct RotaryEmbedding {
    /// 回転の周波数。
    pub frequencies: Param<Tensor<1>>,
}

impl RotaryEmbedding {
    /// パラメーターが初期値の埋め込みを作成する。
    ///
    /// # 引数
    ///
    /// - `device`: パラメーターを配置するデバイス
    pub fn new(device: &Device) -> Self {
        Self {
            frequencies: Initializer::Ones.init([FREQUENCY_COUNT], device),
        }
    }

    /// 系列長に応じた回転を計算する。
    ///
    /// # 引数
    ///
    /// - `sequence_length`: 系列の長さ
    /// - `device`: 回転を配置するデバイス
    pub fn rotation(&self, sequence_length: usize, device: &Device) -> Rotation {
        let positions = Tensor::<1, Int>::arange(0..sequence_length as i64, device)
            .float()
            .unsqueeze_dim::<2>(1);
        // 同じ周波数を隣り合う 2 要素の組に割り当てるため、`[f0, f0, f1, f1, ...]` の順に並べる。
        let angles = (positions * self.frequencies.val().unsqueeze_dim::<2>(0))
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, 2)
            .reshape([sequence_length, HEAD_DIMENSION]);

        let pair_swap_indices: Vec<_> = (0..HEAD_DIMENSION as i32)
            .map(|index| if index % 2 == 0 { index + 1 } else { index - 1 })
            .collect();
        let signs: Vec<f32> = (0..HEAD_DIMENSION)
            .map(|index| if index % 2 == 0 { -1.0 } else { 1.0 })
            .collect();

        Rotation {
            cosine: angles.clone().cos().unsqueeze::<4>(),
            sine: angles.sin().unsqueeze::<4>(),
            pair_swap_indices: Tensor::from_data(
                TensorData::new(pair_swap_indices, [HEAD_DIMENSION]),
                device,
            ),
            signs: Tensor::<1>::from_data(TensorData::new(signs, [HEAD_DIMENSION]), device)
                .unsqueeze::<4>(),
        }
    }
}

/// `[バッチ, ヘッド, 系列, ヘッドの次元]` のテンソルに適用する回転。
#[derive(Debug, Clone)]
pub struct Rotation {
    /// `[1, 1, 系列, ヘッドの次元]` の回転角の余弦。
    cosine: Tensor<4>,
    /// `[1, 1, 系列, ヘッドの次元]` の回転角の正弦。
    sine: Tensor<4>,
    /// 隣り合う 2 要素を組ごとに入れ替える添字。
    pair_swap_indices: Tensor<1, Int>,
    /// 入れ替えた要素に掛ける `[1, 1, 1, ヘッドの次元]` の符号。
    signs: Tensor<4>,
}

impl Rotation {
    /// 最終次元の隣り合う 2 要素の組を位置に応じて回転する。
    ///
    /// # 引数
    ///
    /// - `input`: `[バッチ, ヘッド, 系列, ヘッドの次元]` のテンソル
    pub fn apply(&self, input: Tensor<4>) -> Tensor<4> {
        let rotated_half =
            input.clone().select(3, self.pair_swap_indices.clone()) * self.signs.clone();
        input * self.cosine.clone() + rotated_half * self.sine.clone()
    }
}
