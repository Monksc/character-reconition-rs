use thiserror::Error;

#[derive(Error, Debug)]
pub enum RecognitionError {
    #[error("empty input — no coordinates found")]
    EmptyInput,
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unknown character label: '{0}'")]
    UnknownLabel(char),
    #[error("font error: {0}")]
    Font(String),
    #[error("pca error: {0}")]
    Pca(String),
    #[error("model not found at path: {0}")]
    ModelNotFound(String),
}
