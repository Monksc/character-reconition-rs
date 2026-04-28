use geo::{Coord, LineString, MultiLineString};

/// Douglas-Peucker simplification — reduces points while preserving shape.
/// Apply before feature extraction to reduce noise.
pub fn simplify(mls: &MultiLineString<f64>, epsilon: f64) -> MultiLineString<f64> {
    MultiLineString(
        mls.0
            .iter()
            .map(|ls| {
                if ls.0.len() < 3 {
                    return ls.clone();
                }
                let pts = douglas_peucker(&ls.0, epsilon);
                LineString(pts)
            })
            .collect(),
    )
}

/// Chaikin corner-cutting smoothing (open-polyline variant).
/// Preserves first and last endpoints of each stroke.
/// Each round doubles the point count (minus 2 preserved endpoints).
pub fn chaikin(mls: &MultiLineString<f64>, rounds: usize) -> MultiLineString<f64> {
    MultiLineString(
        mls.0
            .iter()
            .map(|ls| {
                let mut current = ls.clone();
                for _ in 0..rounds {
                    current = chaikin_linestring(&current);
                }
                current
            })
            .collect(),
    )
}

fn chaikin_linestring(ls: &LineString<f64>) -> LineString<f64> {
    if ls.0.len() < 2 {
        return ls.clone();
    }
    let pts = &ls.0;
    let mut out = Vec::with_capacity(pts.len() * 2);
    out.push(pts[0]);
    for w in pts.windows(2) {
        let a = w[0];
        let b = w[1];
        out.push(Coord {
            x: a.x + 0.25 * (b.x - a.x),
            y: a.y + 0.25 * (b.y - a.y),
        });
        out.push(Coord {
            x: a.x + 0.75 * (b.x - a.x),
            y: a.y + 0.75 * (b.y - a.y),
        });
    }
    out.push(*pts.last().unwrap());
    LineString(out)
}

fn douglas_peucker(points: &[Coord<f64>], epsilon: f64) -> Vec<Coord<f64>> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let first = points[0];
    let last = *points.last().unwrap();

    let (max_dist, max_idx) = points[1..points.len() - 1]
        .iter()
        .enumerate()
        .map(|(i, &p)| (point_to_segment_dist(p, first, last), i + 1))
        .fold((0.0f64, 0), |(d, i), (nd, ni)| {
            if nd > d { (nd, ni) } else { (d, i) }
        });

    if max_dist > epsilon {
        let mut left = douglas_peucker(&points[..=max_idx], epsilon);
        let right = douglas_peucker(&points[max_idx..], epsilon);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

fn point_to_segment_dist(p: Coord<f64>, a: Coord<f64>, b: Coord<f64>) -> f64 {
    let ab = Coord { x: b.x - a.x, y: b.y - a.y };
    let ap = Coord { x: p.x - a.x, y: p.y - a.y };
    let len_sq = ab.x * ab.x + ab.y * ab.y;
    if len_sq < f64::EPSILON {
        return (ap.x * ap.x + ap.y * ap.y).sqrt();
    }
    let t = ((ap.x * ab.x + ap.y * ab.y) / len_sq).clamp(0.0, 1.0);
    let proj = Coord {
        x: a.x + t * ab.x,
        y: a.y + t * ab.y,
    };
    ((p.x - proj.x).powi(2) + (p.y - proj.y).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::line_string;

    #[test]
    fn chaikin_doubles_inner_points() {
        let ls = line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 0.0), (x: 2.0, y: 0.0)];
        let mls = MultiLineString(vec![ls]);
        let result = chaikin(&mls, 1);
        // 2 preserved endpoints + 2 new points per original segment (2 segs) = 6
        assert_eq!(result.0[0].0.len(), 6);
    }

    #[test]
    fn simplify_collinear_removes_midpoints() {
        // 5 collinear points should reduce to 2
        let ls = line_string![
            (x: 0.0, y: 0.0), (x: 1.0, y: 0.0), (x: 2.0, y: 0.0),
            (x: 3.0, y: 0.0), (x: 4.0, y: 0.0)
        ];
        let mls = MultiLineString(vec![ls]);
        let result = simplify(&mls, 0.01);
        assert_eq!(result.0[0].0.len(), 2);
    }
}
