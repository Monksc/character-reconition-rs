use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap, linear};
use serde::{Deserialize, Serialize};

use crate::error::RecognitionError;
use crate::features::FEATURE_DIM;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub dropout_rate: f32,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            input_dim: FEATURE_DIM,
            hidden_dims: vec![128, 128],
            output_dim: 62,
            dropout_rate: 0.3,
        }
    }
}

pub struct Mlp {
    layers: Vec<Linear>,
    pub config: MlpConfig,
}

impl Mlp {
    pub fn new(config: &MlpConfig, vb: VarBuilder) -> Result<Self, candle_core::Error> {
        let mut layers = Vec::new();
        let mut in_dim = config.input_dim;
        for (i, &out_dim) in config.hidden_dims.iter().enumerate() {
            layers.push(linear(in_dim, out_dim, vb.pp(format!("layer{i}")))?);
            in_dim = out_dim;
        }
        layers.push(linear(in_dim, config.output_dim, vb.pp("output"))?);
        Ok(Self {
            layers,
            config: config.clone(),
        })
    }

    /// Forward pass. Returns raw logits [batch, output_dim].
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor, candle_core::Error> {
        let n_hidden = self.layers.len() - 1;
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if i < n_hidden {
                h = h.relu()?;
                if training && self.config.dropout_rate > 0.0 {
                    let mask = Tensor::rand(0.0f32, 1.0f32, h.shape(), h.device())?;
                    let keep = mask.ge(self.config.dropout_rate as f64)?;
                    let scale = 1.0 / (1.0 - self.config.dropout_rate as f64);
                    h = (h.clone() * keep.to_dtype(h.dtype())?)?.affine(scale, 0.0)?;
                }
            }
        }
        Ok(h)
    }

    pub fn save(&self, var_map: &VarMap, weights_path: &Path) -> Result<(), RecognitionError> {
        var_map.save(weights_path)?;
        Ok(())
    }

    pub fn load(
        config: &MlpConfig,
        weights_path: &Path,
        device: &Device,
    ) -> Result<(Self, VarMap), RecognitionError> {
        if !weights_path.exists() {
            return Err(RecognitionError::ModelNotFound(
                weights_path.display().to_string(),
            ));
        }
        let mut var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, candle_core::DType::F32, device);
        let mlp = Self::new(config, vb)?;
        var_map.load(weights_path)?;
        Ok((mlp, var_map))
    }
}

pub fn best_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    return Device::new_cuda(0);
    Ok(Device::Cpu)
}
