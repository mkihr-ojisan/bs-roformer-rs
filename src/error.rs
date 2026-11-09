use burn::store::PytorchStoreError;
use burn::tensor::TensorReadError;

/// このクレートの処理の結果。
pub type Result<T> = std::result::Result<T, Error>;

/// モデルの読み込みと音源分離で発生するエラー。
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// チャンク長がモデルで処理できる長さより短い。
    #[error("chunk size {chunk_size} is shorter than the minimum {minimum}")]
    ChunkSizeTooShort {
        /// 指定されたチャンク長。
        chunk_size: usize,
        /// チャンク長の下限。
        minimum: usize,
    },

    /// チェックポイントのファイルを読み込めない。
    #[error("failed to load the checkpoint")]
    LoadCheckpoint(#[from] PytorchStoreError),

    /// チェックポイントにモデルが使用しないキーがある。
    #[error("the checkpoint has keys that the model does not use: {unexpected_keys:?}")]
    UnexpectedCheckpointKeys {
        /// モデルが使用しないキーの一覧。
        unexpected_keys: Vec<String>,
    },

    /// 推論結果をデバイスから取得できない。
    #[error("failed to read a tensor from the device")]
    ReadTensor(#[from] TensorReadError),

    /// 分離結果に NaN または無限大が含まれる。
    #[error("the separated chunk has a non-finite sample")]
    NonFiniteOutput,
}
