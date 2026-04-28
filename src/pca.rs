use std::path::Path;

use linfa::prelude::{Fit, Transformer};
use linfa::Dataset;
use linfa_reduction::Pca;
use ndarray::{Array1, Array2};
use plotters::prelude::*;

use crate::error::RecognitionError;
use crate::features::FEATURE_DIM;

const PALETTE: &[(u8, u8, u8)] = &[
    (228, 26, 28),   (55, 126, 184),  (77, 175, 74),   (152, 78, 163),
    (255, 127, 0),   (166, 86, 40),   (247, 129, 191), (153, 153, 153),
    (255, 255, 51),  (0, 188, 188),   (200, 0, 200),   (0, 128, 64),
    (64, 64, 255),   (255, 64, 64),   (64, 255, 64),   (64, 64, 64),
];

/// Run 2-component PCA on labeled feature vectors and save a scatter plot PNG.
pub fn run_pca_plot(
    samples: &[([f32; FEATURE_DIM], char)],
    output_path: &Path,
    plot_size: (u32, u32),
) -> Result<(), RecognitionError> {
    if samples.is_empty() {
        return Err(RecognitionError::EmptyInput);
    }

    let n = samples.len();
    let raw: Vec<f64> = samples
        .iter()
        .flat_map(|(f, _)| f.iter().map(|&v| v as f64))
        .collect();
    let data = Array2::from_shape_vec((n, FEATURE_DIM), raw)
        .map_err(|e| RecognitionError::Pca(e.to_string()))?;

    let labels: Array1<usize> = Array1::from_vec(
        samples.iter().map(|(_, _)| 0usize).collect(),
    );
    let dataset = Dataset::new(data, labels);

    let pca = Pca::params(2)
        .fit(&dataset)
        .map_err(|e| RecognitionError::Pca(e.to_string()))?;

    let projected_ds = pca.transform(dataset.records().clone().into());
    let projected = projected_ds.records();

    let coords: Vec<(f64, f64, char)> = projected
        .rows()
        .into_iter()
        .zip(samples.iter())
        .map(|(row, (_, ch))| (row[0], row[1], *ch))
        .collect();

    let min_x = coords.iter().map(|&(x, _, _)| x).fold(f64::INFINITY, f64::min);
    let max_x = coords.iter().map(|&(x, _, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let min_y = coords.iter().map(|&(_, y, _)| y).fold(f64::INFINITY, f64::min);
    let max_y = coords.iter().map(|&(_, y, _)| y).fold(f64::NEG_INFINITY, f64::max);

    let pad_x = (max_x - min_x) * 0.05 + 0.01;
    let pad_y = (max_y - min_y) * 0.05 + 0.01;

    let unique_chars: Vec<char> = {
        let mut chars: Vec<char> = samples.iter().map(|(_, c)| *c).collect();
        chars.sort_unstable();
        chars.dedup();
        chars
    };

    let root = BitMapBackend::new(output_path, plot_size).into_drawing_area();
    root.fill(&WHITE).map_err(|e| RecognitionError::Pca(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Character PCA", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (min_x - pad_x)..(max_x + pad_x),
            (min_y - pad_y)..(max_y + pad_y),
        )
        .map_err(|e| RecognitionError::Pca(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc("PC1")
        .y_desc("PC2")
        .draw()
        .map_err(|e| RecognitionError::Pca(e.to_string()))?;

    for (ci, &ch) in unique_chars.iter().enumerate() {
        let (r, g, b) = PALETTE[ci % PALETTE.len()];
        let color = RGBColor(r, g, b);

        let points: Vec<(f64, f64)> = coords
            .iter()
            .filter(|(_, _, c)| *c == ch)
            .map(|&(x, y, _)| (x, y))
            .collect();

        chart
            .draw_series(
                points.iter().map(|&(x, y)| {
                    Circle::new((x, y), 4, color.filled())
                }),
            )
            .map_err(|e| RecognitionError::Pca(e.to_string()))?
            .label(ch.to_string())
            .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));

        // Label each point with its character
        chart
            .draw_series(points.iter().map(|&(x, y)| {
                Text::new(
                    ch.to_string(),
                    (x + pad_x * 0.3, y + pad_y * 0.3),
                    ("sans-serif", 10).into_font().color(&color),
                )
            }))
            .map_err(|e| RecognitionError::Pca(e.to_string()))?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .map_err(|e| RecognitionError::Pca(e.to_string()))?;

    root.present().map_err(|e| RecognitionError::Pca(e.to_string()))?;

    println!("PCA plot saved to: {}", output_path.display());
    Ok(())
}
