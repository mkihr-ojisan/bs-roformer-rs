use crate::stft::CHANNEL_COUNT;

/// 音源分離で得られるパート。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stem {
    /// ベース。
    Bass,
    /// ドラム。
    Drums,
    /// その他の楽器。
    Other,
    /// ボーカル。
    Vocals,
    /// ギター。
    Guitar,
    /// ピアノ。
    Piano,
}

impl Stem {
    /// モデルのマスク推定器の順序で並べたパートの一覧。
    pub const ALL: [Stem; 6] = [
        Stem::Bass,
        Stem::Drums,
        Stem::Other,
        Stem::Vocals,
        Stem::Guitar,
        Stem::Piano,
    ];

    /// パートの英小文字の名前を返す。
    pub fn name(self) -> &'static str {
        match self {
            Stem::Bass => "bass",
            Stem::Drums => "drums",
            Stem::Other => "other",
            Stem::Vocals => "vocals",
            Stem::Guitar => "guitar",
            Stem::Piano => "piano",
        }
    }
}

/// 1 パート分の分離結果。
#[derive(Debug, Clone, PartialEq)]
pub struct SeparatedStem {
    /// パートの種類。
    pub stem: Stem,
    /// チャンネルごとのサンプル列。
    pub samples: [Vec<f32>; CHANNEL_COUNT],
}
