use std::borrow::Cow;

use crate::error::{Error, Result};
use crate::stft::{CHANNEL_COUNT, reflect_padded};

#[cfg(test)]
mod tests;

/// 波形全体をチャンクに分割して処理し、結果を重ね合わせてパートごとの波形を計算する。
///
/// # 引数
///
/// - `mix`: チャンネルごとの波形。長さは一致していること。
/// - `chunk_size`: 1 回の処理に渡すサンプル数。20 以上。
/// - `stem_count`: パート数
/// - `process`: `chunk_size` サンプルの波形を受け取り、`[パート, チャンネル, サンプル]` の順の波形を
///   第 2 引数（要素数は `stem_count * CHANNEL_COUNT * chunk_size`）へ出力する関数
/// - `on_progress`: 処理済みのチャンクの割合（0 より大きく 1 以下）を受け取る関数。チャンクの処理が
///   完了するたびに、前回より大きい値で呼ばれる。成功した場合、最後の呼び出しの値は 1.0 である。
///   処理するチャンクがない（`mix` が空の）場合は、1.0 で 1 回だけ呼ばれる。エラーを返す場合は、
///   エラーの原因となったチャンク以降では呼ばれない。
///
/// # 戻り値
///
/// パートごとに、`mix` と同じ長さのチャンネルごとの波形を返す。`process` の出力に NaN または無限大が
/// 含まれるときは [`Error::NonFiniteOutput`] を返す。
pub fn demix(
    mix: &[Vec<f32>; CHANNEL_COUNT],
    chunk_size: usize,
    stem_count: usize,
    mut process: impl FnMut(&[Vec<f32>; CHANNEL_COUNT], &mut [f32]) -> Result<()>,
    mut on_progress: impl FnMut(f32),
) -> Result<Vec<[Vec<f32>; CHANNEL_COUNT]>> {
    let fade_size = chunk_size / 10;
    let step = chunk_size / 2;
    let border = chunk_size - step;
    let original_length = mix[0].len();
    let is_padded = original_length > 2 * border;

    let signal: [Cow<[f32]>; CHANNEL_COUNT] = std::array::from_fn(|channel| {
        if is_padded {
            Cow::Owned(reflect_padded(&mix[channel], border))
        } else {
            Cow::Borrowed(mix[channel].as_slice())
        }
    });
    let total_length = signal[0].len();
    let base_window = fade_window(chunk_size, fade_size);

    let mut stems: Vec<[Vec<f32>; CHANNEL_COUNT]> = (0..stem_count)
        .map(|_| std::array::from_fn(|_| vec![0.0; total_length]))
        .collect();
    let mut counter = vec![0.0f32; total_length];
    let mut chunk: [Vec<f32>; CHANNEL_COUNT] = std::array::from_fn(|_| vec![0.0; chunk_size]);
    let mut window = vec![0.0; chunk_size];
    let mut separated = vec![0.0; stem_count * CHANNEL_COUNT * chunk_size];

    // 末尾のパディング内で開始するチャンクの出力はすべて破棄されるので処理しない。
    let output_end = if is_padded {
        border + original_length
    } else {
        total_length
    };
    let chunk_count = output_end.div_ceil(step);
    for chunk_index in 0..chunk_count {
        let start = chunk_index * step;
        let segment_length = chunk_size.min(total_length - start);
        for (target, source) in chunk.iter_mut().zip(&signal) {
            fill_chunk(target, &source[start..start + segment_length]);
        }

        process(&chunk, &mut separated)?;
        if separated.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::NonFiniteOutput);
        }

        window.copy_from_slice(&base_window);
        if start == 0 {
            window[..fade_size].fill(1.0);
        }
        if start + step >= total_length {
            window[chunk_size - fade_size..].fill(1.0);
        }

        for (index, source) in separated.chunks_exact(chunk_size).enumerate() {
            let (stem, channel) = (index / CHANNEL_COUNT, index % CHANNEL_COUNT);
            let target = &mut stems[stem][channel][start..start + segment_length];
            for ((value, sample), weight) in target.iter_mut().zip(source).zip(&window) {
                *value += sample * weight;
            }
        }
        for (value, weight) in counter[start..start + segment_length]
            .iter_mut()
            .zip(&window)
        {
            *value += weight;
        }

        // 最後のチャンクでは分子と分母が等しいので、値はちょうど 1.0 になる。
        on_progress((chunk_index + 1) as f32 / chunk_count as f32);
    }
    if chunk_count == 0 {
        on_progress(1.0);
    }

    for channels in &mut stems {
        for channel in channels {
            for (value, count) in channel.iter_mut().zip(&counter) {
                *value /= count;
            }
            if is_padded {
                channel.copy_within(border..border + original_length, 0);
                channel.truncate(original_length);
            }
        }
    }

    Ok(stems)
}

/// 両端を線形にフェードする窓を作成する。
///
/// # 引数
///
/// - `size`: 窓の長さ
/// - `fade_size`: フェードの長さ。2 以上。
fn fade_window(size: usize, fade_size: usize) -> Vec<f32> {
    let mut window = vec![1.0; size];
    let denominator = (fade_size - 1) as f32;
    for index in 0..fade_size {
        let rising = index as f32 / denominator;
        window[index] = rising;
        window[size - 1 - index] = rising;
    }
    window
}

/// 波形をチャンクの先頭へコピーし、残りを埋める。
///
/// 波形がチャンクの半分より長いときは反射で埋め、それ以外は 0 で埋める。
///
/// # 引数
///
/// - `target`: 出力先のチャンク
/// - `segment`: チャンクの先頭に配置する波形
fn fill_chunk(target: &mut [f32], segment: &[f32]) {
    let is_reflected = segment.len() > target.len() / 2;
    target[..segment.len()].copy_from_slice(segment);
    let padding = &mut target[segment.len()..];
    if is_reflected {
        for (offset, value) in padding.iter_mut().enumerate() {
            *value = segment[segment.len() - 2 - offset];
        }
    } else {
        padding.fill(0.0);
    }
}
