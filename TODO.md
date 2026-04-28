# TODO

## Phase 1 — Core Pipeline ✅
- [x] `error.rs` — RecognitionError enum
- [x] `normalize.rs` — bounding box translate + uniform scale
- [x] `augment.rs` — Douglas-Peucker simplification + Chaikin smoothing
- [x] `features.rs` — 28-float FeatureVector extraction (angle bins, centroid, crossings, symmetry)
- [x] `fonts.rs` — system font discovery, glyph → MultiLineString, SVG → MultiLineString
- [x] `model.rs` — Candle MLP, forward pass, safetensors save/load
- [x] `train.rs` — LabelEncoder, preprocessing, AdamW training loop
- [x] `pca.rs` — 2-component PCA + plotters scatter PNG
- [x] `lib.rs` — RecognitionPipeline API
- [x] `main.rs` — `train` / `predict` / `pca` Clap subcommands

## Phase 2 — Validation & Tuning
- [ ] Run `cargo run -- train --chars "0123456789" --fonts-per-char 20 --epochs 30` and check loss curve
- [ ] Run `cargo run -- pca --chars "0123456789"` and inspect whether digits cluster visually
- [ ] Tune `SEGMENT_COUNT_SCALE`, `CROSSING_COUNT_SCALE`, `LONG_STRAIGHT_SCALE` based on real data distributions
- [ ] Tune `hidden_dims` and `dropout_rate` in `MlpConfig`
- [ ] Evaluate confusion matrix (which characters get confused most)
- [ ] Try `--chaikin-rounds 1` augmentation and compare accuracy
- [ ] Add `--chaikin-rounds` flag to `train` subcommand

## Phase 3 — Quality
- [ ] Add `predict` output to show character labels alongside confidence scores (needs encoder saved/loaded)
- [ ] Add `--face-index` flag to support font files with multiple faces (e.g., .ttc collections)
- [ ] Add progress bar to `train` font-loading phase (indicatif crate)
- [ ] Add `--validate-split` flag for hold-out accuracy during training
- [ ] Consider O(n log n) sweep-line intersection algorithm if segment counts exceed ~500
- [ ] Add `export` subcommand: export a character's feature vector to JSON for inspection
- [ ] Consider adding stroke direction as an additional feature (important for some scripts)

## Phase 4 — Extended Characters
- [ ] Test with full printable ASCII (use `--chars` with all 95 printable chars)
- [ ] Consider adding Unicode range support beyond ASCII
- [ ] Benchmark feature extraction time per glyph; target < 1ms per character
