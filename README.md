# character-reconition-rs

A geometric character recognition library for font glyphs. Uses pure feature engineering on `geo::MultiLineString<f64>` stroke geometry fed into a Candle MLP classifier — no images involved.

## Architecture

```
Font file (.ttf/.otf)  or  SVG file
       │                        │
  ttf-parser                 usvg
  lyon_path/lyon_geom       lyon_path
       │                        │
       └──── MultiLineString<f64> ────┐
                                      │
                              augment::simplify()     ← Douglas-Peucker
                              normalize::normalize()  ← origin + height=1.0
                              features::extract()     ← 28-float vector
                                      │
                              Candle MLP (CPU or CUDA)
                                      │
                                predicted char + confidence scores
```

## Feature Vector (30 floats)

| Slots | Feature | Notes |
|-------|---------|-------|
| 0–19 | Angle bins | 20 bins × 18°, length-weighted, sums to 1 |
| 20 | Segment count | Normalized |
| 21 | LineString count | Number of separate strokes |
| 22–23 | Centroid (x, y) | Length-weighted centroid in normalized space |
| 24 | Crossing count | Self-intersections — distinguishes 8, 0, B from A, I |
| 25 | Long straight count | Segments > 25% of character height |
| 26 | Vertical symmetry | +1 = top-heavy, −1 = bottom-heavy (separates M vs W) |
| 27 | Horizontal symmetry | +1 = right-heavy, −1 = left-heavy |
| 28 | Aspect ratio | width/height pre-normalization — W wide (>1), I narrow (<1) |
| 29 | Closed loop count | Normalized — distinguishes O/D/B (loops) from C/L/I (open) |

## CLI Usage

### Train

Discovers all system fonts, renders glyphs for each target character, and trains an MLP:

```bash
# Default: 62 alphanumeric characters, 50 fonts per char, 50 epochs
cargo run --release -- train --output ./model

# Quick run with digits only
cargo run --release -- train \
  --chars "0123456789" \
  --fonts-per-char 20 \
  --epochs 30 \
  --output ./model
```

Saves `model/weights.safetensors`, `model/config.json`, `model/labels.json`.

### Predict

Predicts the character rendered in an SVG file:

```bash
cargo run --release -- predict "glyph.svg" --model ./model
```

### PCA

Renders characters from system fonts, runs PCA on their feature vectors, and saves a 2D scatter plot:

```bash
# Digits only (default)
cargo run --release -- pca --output pca_digits.png

# All alphanumeric
cargo run --release -- pca --chars "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" \
  --fonts-per-char 10 \
  --output pca_all.png
```

## Library Usage

```rust
use character_recognition::RecognitionPipeline;

let pipeline = RecognitionPipeline::load(std::path::Path::new("./model"))?;
let mls = character_recognition::fonts::render_glyph(
    &font_path, 0, 'A', 0.5
)?;
let (predicted, scores) = pipeline.predict(&mls)?;
println!("Predicted: '{}'  confidence: {:.3}", predicted, scores.iter().cloned().fold(0.0f32, f32::max));
```

## Building

```bash
cargo build --release

# GPU support (CUDA)
cargo build --release --features cuda
```

## Modules

| Module | Responsibility |
|--------|---------------|
| `fonts` | System font discovery; glyph → MultiLineString; SVG → MultiLineString |
| `normalize` | Translate to origin, scale height to 1.0 uniformly |
| `augment` | Douglas-Peucker simplification; Chaikin corner smoothing |
| `features` | Extract 28-float FeatureVector from normalized geometry |
| `model` | Candle MLP definition, forward pass, safetensors save/load |
| `train` | LabelEncoder, preprocessing pipeline, AdamW training loop |
| `pca` | 2-component PCA + plotters scatter plot PNG |
| `lib` | `RecognitionPipeline` high-level API |
