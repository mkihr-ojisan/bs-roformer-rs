use burn::module::Param;
use burn::tensor::TensorData;

use super::*;

/// 隣り合う 2 要素の組に `(x0, x1) -> (x0·cos − x1·sin, x1·cos + x0·sin)` を適用し、角度が組の
/// 周波数に位置を掛けた値であることを確認する。
#[test]
fn rotation_applies_the_interleaved_pair_formula() {
    let device = Device::flex();
    let frequencies: Vec<_> = (0..FREQUENCY_COUNT)
        .map(|index| 0.1 + index as f32 * 0.01)
        .collect();
    let embedding = RotaryEmbedding {
        frequencies: Param::from_tensor(Tensor::from_data(
            TensorData::new(frequencies.clone(), [FREQUENCY_COUNT]),
            &device,
        )),
    };

    let sequence_length = 3;
    let values: Vec<_> = (0..sequence_length * HEAD_DIMENSION)
        .map(|index| index as f32 + 1.0)
        .collect();
    let input = Tensor::<4>::from_data(
        TensorData::new(values.clone(), [1, 1, sequence_length, HEAD_DIMENSION]),
        &device,
    );

    let output = embedding
        .rotation(sequence_length, &device)
        .apply(input)
        .try_into_vec_as::<f32>()
        .unwrap();

    for position in 0..sequence_length {
        for (pair, frequency) in frequencies.iter().enumerate() {
            let angle = position as f32 * frequency;
            let (cosine, sine) = (angle.cos(), angle.sin());
            let offset = position * HEAD_DIMENSION + pair * 2;
            let (x0, x1) = (values[offset], values[offset + 1]);
            let expected = [x0 * cosine - x1 * sine, x1 * cosine + x0 * sine];
            for (actual, expected) in output[offset..offset + 2].iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "position {position}, pair {pair}: {actual} != {expected}"
                );
            }
        }
    }
}
