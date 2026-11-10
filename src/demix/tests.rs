use super::*;

/// `chunk` をそのまま 1 パートとして `output` へ出力する処理。
fn identity(chunk: &[Vec<f32>; CHANNEL_COUNT], output: &mut [f32]) -> Result<()> {
    for (target, sample) in output.iter_mut().zip(chunk.iter().flatten()) {
        *target = *sample;
    }
    Ok(())
}

/// 長さ `length` で、サンプル番号と `scale` の積を値とする波形を生成する。
fn ramp_signal(length: usize, scale: f32) -> Vec<f32> {
    (0..length).map(|index| index as f32 * scale).collect()
}

/// 窓の重みの和で正規化すると入力が再現されることを、パディングしない長さ（チャンクの半分以下、
/// 半分超、1 チャンク）とパディングする長さで確認する。
#[test]
fn identity_process_reconstructs_the_mix() {
    let chunk_size = 2048;
    for length in [
        1000,
        chunk_size * 3 / 4,
        chunk_size,
        chunk_size + 1,
        2 * chunk_size,
        2 * chunk_size + 1,
        chunk_size * 3 + chunk_size / 4,
    ] {
        let mix = [ramp_signal(length, 0.001), ramp_signal(length, -0.002)];
        let stems = demix(&mix, chunk_size, 1, identity, |_| {}).unwrap();
        assert_eq!(stems.len(), 1);
        for (output, expected) in stems[0].iter().zip(&mix) {
            assert_eq!(output.len(), length);
            for (index, (a, b)) in output.iter().zip(expected).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "length {length}, sample {index}: {a} != {b}"
                );
            }
        }
    }
}

/// 末尾のパディング内で開始するチャンクを処理しないことを、`process` の呼び出し回数で確認する。
#[test]
fn chunks_starting_in_the_trailing_padding_are_skipped() {
    let chunk_size = 2048;
    let length = 3 * chunk_size;
    let mix = [ramp_signal(length, 0.001), ramp_signal(length, -0.002)];
    let mut call_count = 0;
    demix(
        &mix,
        chunk_size,
        1,
        |chunk, output| {
            call_count += 1;
            identity(chunk, output)
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(call_count, 7);
}

/// `process` の出力に NaN が含まれるときにエラーを返すことを確認する。
#[test]
fn non_finite_output_is_an_error() {
    let mix = [vec![0.0; 1000], vec![0.0; 1000]];
    let result = demix(
        &mix,
        2048,
        1,
        |chunk, output| {
            identity(chunk, output)?;
            output[0] = f32::NAN;
            Ok(())
        },
        |_| {},
    );
    assert!(matches!(result, Err(Error::NonFiniteOutput)));
}

/// 進捗がチャンクごとに処理済みの割合として通知され、最後に 1.0 になることを、パディングする長さ
/// （7 チャンクに分割）とパディングしない長さ（1 チャンクに分割）で確認する。
#[test]
fn progress_is_reported_after_each_chunk() {
    let chunk_size = 2048;
    for (length, chunk_count) in [(3 * chunk_size, 7), (1000, 1)] {
        let mix = [ramp_signal(length, 0.001), ramp_signal(length, -0.002)];
        let mut progress = Vec::new();
        demix(&mix, chunk_size, 1, identity, |value| progress.push(value)).unwrap();
        let expected: Vec<_> = (1..=chunk_count)
            .map(|index| index as f32 / chunk_count as f32)
            .collect();
        assert_eq!(progress, expected, "length {length}");
        assert_eq!(progress.last(), Some(&1.0));
    }
}

/// 入力が空のときは、進捗が 1.0 で 1 回だけ通知されることを確認する。
#[test]
fn progress_of_empty_mix_is_reported_once() {
    let mix = [Vec::new(), Vec::new()];
    let mut progress = Vec::new();
    demix(&mix, 2048, 1, identity, |value| progress.push(value)).unwrap();
    assert_eq!(progress, [1.0]);
}

/// エラーの原因となったチャンク以降では進捗が通知されないことを、2 番目のチャンクの出力に NaN を
/// 含めて確認する。
#[test]
fn progress_stops_at_the_failing_chunk() {
    let chunk_size = 2048;
    let length = 3 * chunk_size;
    let mix = [ramp_signal(length, 0.001), ramp_signal(length, -0.002)];
    let mut call_count = 0;
    let mut progress = Vec::new();
    let result = demix(
        &mix,
        chunk_size,
        1,
        |chunk, output| {
            call_count += 1;
            identity(chunk, output)?;
            if call_count == 2 {
                output[0] = f32::NAN;
            }
            Ok(())
        },
        |value| progress.push(value),
    );
    assert!(matches!(result, Err(Error::NonFiniteOutput)));
    assert_eq!(progress, [1.0 / 7.0]);
}

/// フェードが線形で、窓の中央の値が 1 となることを確認する。
#[test]
fn fade_window_is_linear_with_flat_center() {
    assert_eq!(
        fade_window(10, 3),
        [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]
    );
}

/// 波形がチャンクの半分より長ければ反射で埋め、半分以下なら 0 で埋めることを確認する。
#[test]
fn fill_chunk_uses_reflection_only_for_long_segments() {
    let mut target = [9.0; 4];
    fill_chunk(&mut target, &[1.0, 2.0, 3.0]);
    assert_eq!(target, [1.0, 2.0, 3.0, 2.0]);

    let mut target = [9.0; 4];
    fill_chunk(&mut target, &[1.0, 2.0]);
    assert_eq!(target, [1.0, 2.0, 0.0, 0.0]);
}
