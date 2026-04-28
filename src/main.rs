use std::path::PathBuf;

use character_recognition::augment;
use character_recognition::error::RecognitionError;
use character_recognition::features;
use character_recognition::fonts;
use character_recognition::glyph_db::{GlyphDatabase, GlyphDbConfig};
use character_recognition::model::{best_device, MlpConfig};
use character_recognition::normalize;
use character_recognition::pca;
use character_recognition::train::{self, LabelEncoder, TrainConfig};
use character_recognition::RecognitionPipeline;
use clap::{Parser, Subcommand};
use rand::seq::SliceRandom;
use rand::rng;

const DEFAULT_CHARS: &str =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

#[derive(Parser)]
#[command(name = "charrecog", about = "Geometric font character recognition")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Discover system fonts, render glyphs, and train an MLP classifier
    Train(TrainArgs),
    /// Predict the character in an SVG file using a trained MLP model
    Predict(PredictArgs),
    /// Run PCA on character feature vectors and save a 2D scatter plot
    Pca(PcaArgs),
    /// KNN identify: load all font glyphs into an R-tree and vote on the top-N closest
    Identify(IdentifyArgs),
    /// Render a character from a system font and save it as an SVG file
    Export(ExportArgs),
    /// Show which character pairs are geometrically closest (hardest to distinguish)
    Analyze(AnalyzeArgs),
}

#[derive(clap::Args)]
struct TrainArgs {
    /// Directory to save model weights, config, and labels
    #[arg(long, default_value = "model")]
    output: PathBuf,

    #[arg(long, default_value_t = 50)]
    epochs: usize,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// Characters to include in training
    #[arg(long, default_value = DEFAULT_CHARS)]
    chars: String,

    /// Maximum number of fonts to sample per character
    #[arg(long, default_value_t = 50)]
    fonts_per_char: usize,

    /// Flatten bezier curves to line segments with this tolerance
    #[arg(long, default_value_t = 0.5)]
    flatten_tolerance: f32,

    /// Douglas-Peucker simplification epsilon (0 = disabled)
    #[arg(long, default_value_t = 0.5)]
    simplify_epsilon: f64,
}

#[derive(clap::Args)]
struct PredictArgs {
    /// Path to a trained model directory
    #[arg(long, default_value = "model")]
    model: PathBuf,

    /// SVG file to predict
    input: PathBuf,
}

#[derive(clap::Args)]
struct PcaArgs {
    /// Characters to include in PCA
    #[arg(long, default_value = "0123456789")]
    chars: String,

    /// Number of fonts to sample per character
    #[arg(long, default_value_t = 20)]
    fonts_per_char: usize,

    /// Output PNG file path
    #[arg(long, default_value = "pca.png")]
    output: PathBuf,

    /// Plot dimensions WxH in pixels
    #[arg(long, default_value = "1024x1024")]
    size: String,

    #[arg(long, default_value_t = 0.5)]
    flatten_tolerance: f32,
}

#[derive(clap::Args)]
struct IdentifyArgs {
    /// SVG file to identify
    input: PathBuf,

    /// Number of nearest neighbors to retrieve and vote over
    #[arg(long, default_value_t = 16)]
    top: usize,

    /// Characters to include in the glyph database
    #[arg(long, default_value = DEFAULT_CHARS)]
    chars: String,

    /// Fonts to sample per character when building the database
    #[arg(long, default_value_t = 50)]
    fonts_per_char: usize,

    #[arg(long, default_value_t = 0.5)]
    flatten_tolerance: f32,

    #[arg(long, default_value_t = 0.5)]
    simplify_epsilon: f64,
}

#[derive(clap::Args)]
struct ExportArgs {
    /// Character to render
    char: char,

    /// Output SVG file path
    #[arg(long, default_value = "glyph.svg")]
    output: PathBuf,

    /// Specific font file to use (picks a random system font if omitted)
    #[arg(long)]
    font: Option<PathBuf>,

    #[arg(long, default_value_t = 0.5)]
    flatten_tolerance: f32,
}

fn main() -> Result<(), RecognitionError> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => cmd_train(args),
        Command::Predict(args) => cmd_predict(args),
        Command::Pca(args) => cmd_pca(args),
        Command::Identify(args) => cmd_identify(args),
        Command::Export(args) => cmd_export(args),
        Command::Analyze(args) => cmd_analyze(args),
    }
}

#[derive(clap::Args)]
struct AnalyzeArgs {
    /// Characters to compare
    #[arg(long, default_value = DEFAULT_CHARS)]
    chars: String,

    /// Fonts to sample per character
    #[arg(long, default_value_t = 50)]
    fonts_per_char: usize,

    /// How many closest pairs to show
    #[arg(long, default_value_t = 20)]
    top: usize,
}

fn cmd_analyze(args: AnalyzeArgs) -> Result<(), RecognitionError> {
    let chars: Vec<char> = args.chars.chars().collect();
    let config = GlyphDbConfig {
        chars: chars.clone(),
        fonts_per_char: args.fonts_per_char,
        flatten_tolerance: 0.5,
        simplify_epsilon: 0.5,
    };
    let db = GlyphDatabase::init(&config)?;
    println!("Indexed {} glyphs.\n", db.len());

    // Compute per-class centroid feature vectors
    let centroids: Vec<(char, [f32; character_recognition::FEATURE_DIM])> = {
        let entries = db.entries();
        chars.iter().filter_map(|&ch| {
            let class_entries: Vec<_> = entries.iter()
                .filter(|e| e.character == ch)
                .collect();
            if class_entries.is_empty() { return None; }
            let n = class_entries.len() as f32;
            let mut centroid = [0.0f32; character_recognition::FEATURE_DIM];
            for e in &class_entries {
                for (i, v) in e.point.0.iter().enumerate() {
                    centroid[i] += v / n;
                }
            }
            Some((ch, centroid))
        }).collect()
    };

    // Compute all pairwise distances between centroids
    let mut pairs: Vec<(f32, char, char)> = Vec::new();
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let (ca, a) = &centroids[i];
            let (cb, b) = &centroids[j];
            let dist: f32 = a.iter().zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();
            pairs.push((dist, *ca, *cb));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("Closest character pairs (centroid distance, lower = harder to distinguish):");
    println!("{:<6} {:<10} pair", "dist", "");
    println!("{}", "-".repeat(30));
    for (dist, ca, cb) in pairs.iter().take(args.top) {
        println!("  {:.4}   '{}' ↔ '{}'", dist, ca, cb);
    }

    Ok(())
}

fn cmd_train(args: TrainArgs) -> Result<(), RecognitionError> {
    let chars: Vec<char> = args.chars.chars().collect();
    println!("Discovering system fonts...");
    let all_fonts = fonts::list_system_fonts();
    println!("Found {} font files", all_fonts.len());

    let mut rng = rng();
    let mut labeled_data = Vec::new();

    for &ch in &chars {
        let mut fonts_for_char: Vec<_> = all_fonts.clone();
        fonts_for_char.shuffle(&mut rng);
        fonts_for_char.truncate(args.fonts_per_char);

        let mut count = 0;
        for font_path in &fonts_for_char {
            match fonts::render_glyph(font_path, 0, ch, args.flatten_tolerance) {
                Ok(mls) => {
                    labeled_data.push((mls, ch));
                    count += 1;
                }
                Err(RecognitionError::EmptyInput) => {}
                Err(_) => {}
            }
        }
        println!("  '{}': {} samples", ch, count);
    }

    println!("Total samples: {}", labeled_data.len());
    if labeled_data.is_empty() {
        return Err(RecognitionError::EmptyInput);
    }

    let encoder = LabelEncoder::from_chars(&chars);
    let simplify = if args.simplify_epsilon > 0.0 { Some(args.simplify_epsilon) } else { None };
    let samples = train::preprocess(&labeled_data, &encoder, 0, simplify)?;

    let mlp_config = MlpConfig {
        input_dim: character_recognition::FEATURE_DIM,
        hidden_dims: vec![128, 64],
        output_dim: encoder.num_classes(),
        dropout_rate: 0.3,
    };

    let train_config = TrainConfig {
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        batch_size: args.batch_size,
        simplify_epsilon: args.simplify_epsilon,
        ..Default::default()
    };

    let device = best_device()?;
    let (mlp, var_map) = train::train(&samples, &train_config, &mlp_config, &device)?;

    std::fs::create_dir_all(&args.output)?;
    let weights_path = args.output.join("weights.safetensors");
    let config_path = args.output.join("config.json");
    let labels_path = args.output.join("labels.json");

    mlp.save(&var_map, &weights_path)?;
    std::fs::write(&config_path, serde_json::to_string_pretty(&mlp_config)?)?;
    std::fs::write(&labels_path, serde_json::to_string_pretty(&encoder)?)?;

    println!("Model saved to: {}", args.output.display());
    Ok(())
}

fn cmd_predict(args: PredictArgs) -> Result<(), RecognitionError> {
    let pipeline = RecognitionPipeline::load(&args.model)?;

    let mls = fonts::render_svg(&args.input, 0.5)?;
    let (predicted, scores) = pipeline.predict(&mls)?;

    println!("Predicted: '{}'", predicted);

    // Show top-5 alternatives
    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top predictions:");
    for (idx, score) in indexed.iter().take(5) {
        // We don't have direct encoder access here; print index + score
        println!("  class {}: {:.4}", idx, score);
    }

    Ok(())
}

fn cmd_pca(args: PcaArgs) -> Result<(), RecognitionError> {
    let chars: Vec<char> = args.chars.chars().collect();
    let all_fonts = fonts::list_system_fonts();
    println!("Found {} font files", all_fonts.len());

    let mut rng = rng();
    let mut samples: Vec<([f32; character_recognition::FEATURE_DIM], char)> = Vec::new();

    for &ch in &chars {
        let mut fonts_for_char: Vec<_> = all_fonts.clone();
        fonts_for_char.shuffle(&mut rng);
        fonts_for_char.truncate(args.fonts_per_char);

        for font_path in &fonts_for_char {
            match fonts::render_glyph(font_path, 0, ch, args.flatten_tolerance) {
                Ok(mls) => {
                    let simplified = augment::simplify(&mls, 0.5);
                    let ar = normalize::aspect_ratio(&simplified);
                    match normalize::normalize(&simplified) {
                        Ok(normed) => {
                            if let Ok(fv) = features::extract(&normed, ar) {
                                samples.push((fv.to_array(), ch));
                            }
                        }
                        Err(_) => {}
                    }
                }
                Err(_) => {}
            }
        }
    }

    println!("Extracted {} samples for PCA", samples.len());
    if samples.is_empty() {
        return Err(RecognitionError::EmptyInput);
    }

    let plot_size = parse_size(&args.size).unwrap_or((1024, 1024));
    pca::run_pca_plot(&samples, &args.output, plot_size)?;
    Ok(())
}

fn cmd_identify(args: IdentifyArgs) -> Result<(), RecognitionError> {
    let chars: Vec<char> = args.chars.chars().collect();

    let config = GlyphDbConfig {
        chars,
        fonts_per_char: args.fonts_per_char,
        flatten_tolerance: args.flatten_tolerance,
        simplify_epsilon: args.simplify_epsilon,
    };

    let db = GlyphDatabase::init(&config)?;
    println!("Indexed {} glyphs total.\n", db.len());

    let mls = fonts::render_svg(&args.input, args.flatten_tolerance)?;
    let matches = db.top_matches(&mls, args.top)?;

    if matches.is_empty() {
        println!("No matches found.");
        return Ok(());
    }

    println!("Top {} nearest neighbors:", matches.len());
    for (i, m) in matches.iter().enumerate() {
        println!("  #{:<2}  '{}'  dist={:.4}", i + 1, m.character, m.distance);
    }

    // Vote tally
    let mut tally: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for m in &matches {
        *tally.entry(m.character).or_insert(0) += 1;
    }
    let mut tally_vec: Vec<(char, usize)> = tally.into_iter().collect();
    // Sort by count desc, then by first appearance (lowest distance) as tiebreak
    tally_vec.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nVote tally (top {}):", args.top);
    let winner = tally_vec[0].0;
    for (ch, count) in &tally_vec {
        let marker = if *ch == winner { " ◀ winner" } else { "" };
        println!("  '{}'  {}/{}{}",  ch, count, args.top, marker);
    }

    println!("\nIdentified: '{}'", winner);
    Ok(())
}

fn cmd_export(args: ExportArgs) -> Result<(), RecognitionError> {
    use rand::seq::SliceRandom;

    let font_path = match args.font {
        Some(p) => p,
        None => {
            let all_fonts = fonts::list_system_fonts();
            let mut rng = rng();
            let mut candidates = all_fonts.clone();
            candidates.shuffle(&mut rng);
            // Try fonts until one has a glyph for this character
            candidates
                .into_iter()
                .find(|p| {
                    fonts::render_glyph(p, 0, args.char, args.flatten_tolerance).is_ok()
                })
                .ok_or_else(|| {
                    RecognitionError::Font(format!("no system font has a glyph for '{}'", args.char))
                })?
        }
    };

    let mls = fonts::render_glyph(&font_path, 0, args.char, args.flatten_tolerance)?;
    let svg_content = multilinestring_to_svg(&mls, args.char);
    std::fs::write(&args.output, &svg_content)?;

    println!(
        "Exported '{}' from {} → {}",
        args.char,
        font_path.display(),
        args.output.display()
    );
    Ok(())
}

/// Convert a MultiLineString to a minimal SVG file.
/// Each LineString becomes one <path> element using M/L commands.
fn multilinestring_to_svg(mls: &geo::MultiLineString<f64>, label: char) -> String {
    // Compute bounding box for viewBox
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for ls in &mls.0 {
        for c in &ls.0 {
            if c.x < min_x { min_x = c.x; }
            if c.y < min_y { min_y = c.y; }
            if c.x > max_x { max_x = c.x; }
            if c.y > max_y { max_y = c.y; }
        }
    }
    let w = (max_x - min_x).max(1.0);
    let h = (max_y - min_y).max(1.0);

    let mut paths = String::new();
    for ls in &mls.0 {
        if ls.0.is_empty() { continue; }
        let mut d = String::new();
        for (i, c) in ls.0.iter().enumerate() {
            if i == 0 {
                d.push_str(&format!("M {:.4},{:.4}", c.x - min_x, c.y - min_y));
            } else {
                d.push_str(&format!(" L {:.4},{:.4}", c.x - min_x, c.y - min_y));
            }
        }
        paths.push_str(&format!(
            "  <path d=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"{:.2}\"/>\n",
            d,
            w * 0.02
        ));
    }

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.4} {h:.4}" width="512" height="512">
  <title>{label}</title>
{paths}</svg>
"#,
        w = w, h = h, label = label, paths = paths
    )
}

fn parse_size(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let w = parts[0].parse().ok()?;
    let h = parts[1].parse().ok()?;
    Some((w, h))
}
