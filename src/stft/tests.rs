use super::*;

/// 長さ `length` の擬似乱数の波形を `seed` から決定的に生成する。
fn pseudo_random_signal(length: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// スペクトログラムから周波数ビンの列を取得する。
///
/// # 引数
///
/// - `spectrogram`: 取得元のスペクトログラム
/// - `frame`: フレームの番号
/// - `channel`: チャンネルの番号
/// - `bins`: 出力先の周波数ビンの列
fn bins_of(spectrogram: &Spectrogram, frame: usize, channel: usize, bins: &mut [Complex<f32>]) {
    for (bin, value) in bins.iter_mut().enumerate() {
        let offset = frame * SPECTROGRAM_WIDTH + bin * CHANNEL_COUNT * 2 + channel * 2;
        *value = Complex::new(spectrogram.values[offset], spectrogram.values[offset + 1]);
    }
}

/// STFT と iSTFT で波形が復元されることを、ホップ長で割り切れる長さと割り切れない長さで確認する。
#[test]
fn inverse_reconstructs_the_signal() {
    let stft = Stft::new();
    for length in [HOP_LENGTH * 20, HOP_LENGTH * 20 + 37] {
        let channels = [
            pseudo_random_signal(length, 1),
            pseudo_random_signal(length, 2),
        ];
        let spectrogram = stft.forward(&channels);
        assert_eq!(spectrogram.frame_count, length / HOP_LENGTH + 1);

        let mut workspace = InverseWorkspace::new(&stft, spectrogram.frame_count);
        for (channel, signal) in channels.iter().enumerate() {
            let mut output = vec![0.0; length];
            stft.inverse(
                |frame, bins| bins_of(&spectrogram, frame, channel, bins),
                &mut output,
                &mut workspace,
            );
            let maximum_error = output
                .iter()
                .zip(signal)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(maximum_error < 1e-4, "maximum error {maximum_error}");
        }
    }
}

/// 反射パディングが端のサンプルを繰り返さないことを確認する。
#[test]
fn reflect_padding_excludes_the_edge_sample() {
    assert_eq!(
        reflect_padded(&[1.0, 2.0, 3.0, 4.0], 2),
        [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
    );
}
