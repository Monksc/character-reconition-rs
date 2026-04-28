use geo::{Coord, LineString, MultiLineString};

use crate::error::RecognitionError;

/// Translate so min=(0,0), then scale uniformly so max Y = 1.0.
/// Width scales by the same factor — no distortion.
pub fn normalize(mls: &MultiLineString<f64>) -> Result<MultiLineString<f64>, RecognitionError> {
    let (min_x, min_y, max_x, max_y) = bounding_box(mls).ok_or(RecognitionError::EmptyInput)?;

    let height = max_y - min_y;
    let width = max_x - min_x;
    let scale = if height > f64::EPSILON {
        1.0 / height
    } else if width > f64::EPSILON {
        1.0 / width
    } else {
        return Err(RecognitionError::EmptyInput);
    };

    let result = mls
        .0
        .iter()
        .map(|ls| {
            let coords: Vec<Coord<f64>> = ls
                .0
                .iter()
                .map(|c| Coord {
                    x: (c.x - min_x) * scale,
                    y: (c.y - min_y) * scale,
                })
                .collect();
            LineString(coords)
        })
        .collect();

    Ok(MultiLineString(result))
}

/// Width-to-height ratio of the raw (un-normalized) MLS.
/// Must be called before normalize() since normalization destroys this info.
pub fn aspect_ratio(mls: &MultiLineString<f64>) -> f32 {
    match bounding_box(mls) {
        Some((min_x, min_y, max_x, max_y)) => {
            let h = max_y - min_y;
            let w = max_x - min_x;
            if h < f64::EPSILON { 1.0 } else { (w / h) as f32 }
        }
        None => 1.0,
    }
}

fn bounding_box(mls: &MultiLineString<f64>) -> Option<(f64, f64, f64, f64)> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut found = false;

    for ls in &mls.0 {
        for c in &ls.0 {
            found = true;
            if c.x < min_x { min_x = c.x; }
            if c.y < min_y { min_y = c.y; }
            if c.x > max_x { max_x = c.x; }
            if c.y > max_y { max_y = c.y; }
        }
    }

    if found { Some((min_x, min_y, max_x, max_y)) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::line_string;

    #[test]
    fn normalize_unit_square() {
        let mls = MultiLineString(vec![
            line_string![(x: 10.0, y: 20.0), (x: 30.0, y: 40.0)],
        ]);
        let norm = normalize(&mls).unwrap();
        let coords: Vec<_> = norm.0[0].0.iter().collect();
        assert!((coords[0].x).abs() < 1e-9);
        assert!((coords[0].y).abs() < 1e-9);
        assert!((coords[1].y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_empty_errors() {
        let mls = MultiLineString::<f64>(vec![]);
        assert!(normalize(&mls).is_err());
    }
}
