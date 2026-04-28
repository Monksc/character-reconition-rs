pub mod augment;
pub mod error;
pub mod features;
pub mod fonts;
pub mod glyph_db;
pub mod model;
pub mod normalize;
pub mod pca;
pub mod train;

pub use error::RecognitionError;
pub use features::{FeatureVector, FEATURE_DIM};
pub use glyph_db::{CharMatch, GlyphDatabase, GlyphDbConfig};
pub use model::{best_device, Mlp, MlpConfig};
pub use train::LabelEncoder;

use std::path::Path;

use candle_core::Tensor;
use geo::MultiLineString;

/// High-level pipeline: load model from disk, normalize, extract, infer.
pub struct RecognitionPipeline {
    mlp: Mlp,
    encoder: LabelEncoder,
    device: candle_core::Device,
}

impl RecognitionPipeline {
    pub fn load(model_dir: &Path) -> Result<Self, RecognitionError> {
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("weights.safetensors");
        let labels_path = model_dir.join("labels.json");

        if !config_path.exists() || !weights_path.exists() || !labels_path.exists() {
            return Err(RecognitionError::ModelNotFound(model_dir.display().to_string()));
        }

        let config: MlpConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let encoder: LabelEncoder =
            serde_json::from_str(&std::fs::read_to_string(&labels_path)?)?;

        let device = best_device()?;
        let (mlp, _var_map) = Mlp::load(&config, &weights_path, &device)?;

        Ok(Self { mlp, encoder, device })
    }

    /// Normalize, extract features, and infer from raw geometry.
    pub fn predict(
        &self,
        input: &MultiLineString<f64>,
    ) -> Result<(char, Vec<f32>), RecognitionError> {
        let ar = normalize::aspect_ratio(input);
        let normed = normalize::normalize(input)?;
        let fv = features::extract(&normed, ar)?;
        let arr = fv.to_array();

        let tensor = Tensor::from_slice(&arr, (1, FEATURE_DIM), &self.device)?
            .to_dtype(candle_core::DType::F32)?;
        let logits = self.mlp.forward(&tensor, false)?;

        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let probs_vec: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

        let best_idx = probs_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let predicted = self
            .encoder
            .decode(best_idx)
            .ok_or(RecognitionError::UnknownLabel('\0'))?;

        Ok((predicted, probs_vec))
    }

    pub fn predict_char(&self, input: &MultiLineString<f64>) -> Result<char, RecognitionError> {
        self.predict(input).map(|(c, _)| c)
    }
}
