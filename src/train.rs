use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use geo::MultiLineString;
use rand::seq::SliceRandom;
use rand::rng;
use serde::{Deserialize, Serialize};

use crate::augment;
use crate::error::RecognitionError;
use crate::features::{self, FEATURE_DIM};
use crate::model::{Mlp, MlpConfig};
use crate::normalize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEncoder {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
}

impl LabelEncoder {
    pub fn from_chars(chars: &[char]) -> Self {
        let mut sorted: Vec<char> = chars.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let char_to_idx = sorted.iter().copied().enumerate().map(|(i, c)| (c, i)).collect();
        Self { char_to_idx, idx_to_char: sorted }
    }

    pub fn encode(&self, c: char) -> Result<usize, RecognitionError> {
        self.char_to_idx.get(&c).copied().ok_or(RecognitionError::UnknownLabel(c))
    }

    pub fn decode(&self, idx: usize) -> Option<char> {
        self.idx_to_char.get(idx).copied()
    }

    pub fn num_classes(&self) -> usize {
        self.idx_to_char.len()
    }

    pub fn chars(&self) -> &[char] {
        &self.idx_to_char
    }
}

pub struct Sample {
    pub features: [f32; FEATURE_DIM],
    pub label_idx: usize,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub chaikin_rounds: usize,
    pub simplify_epsilon: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            learning_rate: 1e-3,
            batch_size: 32,
            chaikin_rounds: 0,
            simplify_epsilon: 0.5,
        }
    }
}

/// Normalize + optionally augment + extract features from labeled geometry.
pub fn preprocess(
    data: &[(MultiLineString<f64>, char)],
    encoder: &LabelEncoder,
    chaikin_rounds: usize,
    simplify_epsilon: Option<f64>,
) -> Result<Vec<Sample>, RecognitionError> {
    let mut samples = Vec::with_capacity(data.len());
    for (mls, ch) in data {
        let label_idx = encoder.encode(*ch)?;
        let processed = if let Some(eps) = simplify_epsilon {
            augment::simplify(mls, eps)
        } else {
            mls.clone()
        };
        let smoothed = if chaikin_rounds > 0 {
            augment::chaikin(&processed, chaikin_rounds)
        } else {
            processed
        };
        let ar = normalize::aspect_ratio(&smoothed);
        let normed = normalize::normalize(&smoothed)?;
        let fv = features::extract(&normed, ar)?;
        samples.push(Sample { features: fv.to_array(), label_idx });
    }
    Ok(samples)
}

pub fn train(
    samples: &[Sample],
    config: &TrainConfig,
    mlp_config: &MlpConfig,
    device: &Device,
) -> Result<(Mlp, VarMap), RecognitionError> {
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
    let mlp = Mlp::new(mlp_config, vb)?;

    let mut opt = AdamW::new(
        var_map.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            ..Default::default()
        },
    )?;

    let mut indices: Vec<usize> = (0..samples.len()).collect();
    let mut rng = rng();

    for epoch in 0..config.epochs {
        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        let mut epoch_correct = 0usize;
        let mut batch_count = 0usize;

        for chunk in indices.chunks(config.batch_size) {
            let feat_data: Vec<f32> = chunk
                .iter()
                .flat_map(|&i| samples[i].features.iter().copied())
                .collect();
            let label_data: Vec<u32> = chunk
                .iter()
                .map(|&i| samples[i].label_idx as u32)
                .collect();

            let features_t =
                Tensor::from_vec(feat_data, (chunk.len(), FEATURE_DIM), device)?
                    .to_dtype(DType::F32)?;
            let labels_t = Tensor::from_vec(label_data, (chunk.len(),), device)?;

            let logits = mlp.forward(&features_t, true)?;
            let batch_loss = loss::cross_entropy(&logits, &labels_t)?;
            opt.backward_step(&batch_loss)?;

            epoch_loss += batch_loss.to_scalar::<f32>()?;
            batch_count += 1;

            let preds = logits.argmax(1)?;
            let correct = preds.eq(&labels_t)?.sum_all()?.to_scalar::<u8>()? as usize;
            epoch_correct += correct;
        }

        let avg_loss = epoch_loss / batch_count as f32;
        let acc = epoch_correct as f32 / samples.len() as f32;
        println!("epoch {}/{}: loss={:.4}  acc={:.3}", epoch + 1, config.epochs, avg_loss, acc);
    }

    Ok((mlp, var_map))
}
