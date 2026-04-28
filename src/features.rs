use geo::{Coord, MultiLineString};
use serde::{Deserialize, Serialize};

use crate::error::RecognitionError;

pub const FEATURE_DIM: usize = 48;

const SEGMENT_COUNT_SCALE: f32 = 200.0;
const LINESTRING_COUNT_SCALE: f32 = 10.0;
const CROSSING_COUNT_SCALE: f32 = 20.0;
const LONG_STRAIGHT_SCALE: f32 = 20.0;
const LONG_STRAIGHT_THRESHOLD: f64 = 0.25;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// 20 bins × 18°, length-weighted, normalized to sum=1
    pub angle_bins: [f32; 20],
    pub segment_count_norm: f32,
    pub linestring_count_norm: f32,
    pub centroid_x: f32,
    pub centroid_y: f32,
    /// Self-intersections (distinguishes 8, 0, 6, B from A, T, I)
    pub crossing_count_norm: f32,
    /// Segments longer than 25% of normalized height
    pub long_straight_count_norm: f32,
    /// (top_len - bottom_len) / total_len  ∈ [-1, 1] — separates M vs W
    pub vertical_symmetry: f32,
    /// (right_len - left_len) / total_len  ∈ [-1, 1]
    pub horizontal_symmetry: f32,
    /// width / height in original (pre-normalization) space — wide chars > 1, narrow < 1
    pub aspect_ratio: f32,
    /// Number of closed loops (LineStrings where first ≈ last point), normalized by 5.0
    pub closed_loop_count_norm: f32,
    /// 3×3 spatial grid — total stroke length in each cell, normalized to sum=1.
    /// Cells are row-major: [top-left, top-center, top-right, mid-left, ..., bot-right]
    pub grid_cells: [f32; 9],
    /// 3×3 spatial grid — count of stroke endpoints (open-loop only) in each cell, normalized to sum=1.
    pub endpoint_grid: [f32; 9],
}

impl FeatureVector {
    pub fn to_array(&self) -> [f32; FEATURE_DIM] {
        let mut out = [0.0f32; FEATURE_DIM];
        out[..20].copy_from_slice(&self.angle_bins);
        out[20] = self.segment_count_norm;
        out[21] = self.linestring_count_norm;
        out[22] = self.centroid_x;
        out[23] = self.centroid_y;
        out[24] = self.crossing_count_norm;
        out[25] = self.long_straight_count_norm;
        out[26] = self.vertical_symmetry;
        out[27] = self.horizontal_symmetry;
        out[28] = self.aspect_ratio;
        out[29] = self.closed_loop_count_norm;
        out[30..39].copy_from_slice(&self.grid_cells);
        out[39..48].copy_from_slice(&self.endpoint_grid);
        out
    }

    pub fn from_slice(s: &[f32]) -> Option<Self> {
        if s.len() < FEATURE_DIM {
            return None;
        }
        let mut angle_bins = [0.0f32; 20];
        angle_bins.copy_from_slice(&s[..20]);
        Some(Self {
            angle_bins,
            segment_count_norm: s[20],
            linestring_count_norm: s[21],
            centroid_x: s[22],
            centroid_y: s[23],
            crossing_count_norm: s[24],
            long_straight_count_norm: s[25],
            vertical_symmetry: s[26],
            horizontal_symmetry: s[27],
            aspect_ratio: s[28],
            closed_loop_count_norm: s[29],
            grid_cells: s[30..39].try_into().unwrap(),
            endpoint_grid: s[39..48].try_into().unwrap(),
        })
    }
}

/// Extract features from a normalized MultiLineString.
/// `aspect_ratio` must be computed from the raw MLS via normalize::aspect_ratio() before calling normalize().
pub fn extract(mls: &MultiLineString<f64>, aspect_ratio: f32) -> Result<FeatureVector, RecognitionError> {
    let segs = segments(mls);
    if segs.is_empty() {
        return Err(RecognitionError::EmptyInput);
    }

    let (angle_bins, total_len) = compute_angle_bins(&segs);
    let (cx, cy) = compute_centroid(&segs, total_len);
    let crossing_count = count_self_intersections(&segs, mls);
    let long_count = count_long_straight(&segs);
    let (vert_sym, horiz_sym) = compute_symmetry(&segs, total_len, cx);
    let closed_loops = count_closed_loops(mls);
    let max_x = segs.iter().map(|(a, b, _, _)| a.x.max(b.x)).fold(0.0f64, f64::max);
    let grid_cells = compute_grid_cells(&segs, max_x);
    let endpoint_grid = compute_endpoint_grid(mls, max_x);

    let ls_count = mls.0.len();
    let seg_count = segs.len();

    Ok(FeatureVector {
        angle_bins,
        segment_count_norm: (seg_count as f32 / SEGMENT_COUNT_SCALE).min(1.0),
        linestring_count_norm: (ls_count as f32 / LINESTRING_COUNT_SCALE).min(1.0),
        centroid_x: cx as f32,
        centroid_y: cy as f32,
        crossing_count_norm: (crossing_count as f32 / CROSSING_COUNT_SCALE).min(1.0),
        long_straight_count_norm: (long_count as f32 / LONG_STRAIGHT_SCALE).min(1.0),
        vertical_symmetry: vert_sym,
        horizontal_symmetry: horiz_sym,
        aspect_ratio,
        closed_loop_count_norm: (closed_loops as f32 / 5.0).min(1.0),
        grid_cells,
        endpoint_grid,
    })
}

/// Flat list of (start, end, linestring_index, segment_index_within_ls) tuples.
type Seg = (Coord<f64>, Coord<f64>, usize, usize);

fn segments(mls: &MultiLineString<f64>) -> Vec<Seg> {
    let mut out = Vec::new();
    for (ls_idx, ls) in mls.0.iter().enumerate() {
        for (seg_idx, w) in ls.0.windows(2).enumerate() {
            out.push((w[0], w[1], ls_idx, seg_idx));
        }
    }
    out
}

fn seg_length(a: Coord<f64>, b: Coord<f64>) -> f64 {
    ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt()
}

fn seg_angle_deg(a: Coord<f64>, b: Coord<f64>) -> f64 {
    let deg = (b.y - a.y).atan2(b.x - a.x).to_degrees();
    (deg + 360.0) % 360.0
}

fn compute_angle_bins(segs: &[Seg]) -> ([f32; 20], f64) {
    let mut bins = [0.0f64; 20];
    let mut total = 0.0f64;
    for &(a, b, _, _) in segs {
        let len = seg_length(a, b);
        if len < f64::EPSILON {
            continue;
        }
        let bin = (seg_angle_deg(a, b) / 18.0) as usize;
        let bin = bin.min(19);
        bins[bin] += len;
        total += len;
    }
    let mut out = [0.0f32; 20];
    if total > f64::EPSILON {
        for (i, &v) in bins.iter().enumerate() {
            out[i] = (v / total) as f32;
        }
    }
    (out, total)
}

fn compute_centroid(segs: &[Seg], total_len: f64) -> (f64, f64) {
    if total_len < f64::EPSILON {
        return (0.5, 0.5);
    }
    let mut cx = 0.0f64;
    let mut cy = 0.0f64;
    for &(a, b, _, _) in segs {
        let len = seg_length(a, b);
        cx += ((a.x + b.x) * 0.5) * len;
        cy += ((a.y + b.y) * 0.5) * len;
    }
    (cx / total_len, cy / total_len)
}

fn count_long_straight(segs: &[Seg]) -> usize {
    segs.iter()
        .filter(|&&(a, b, _, _)| seg_length(a, b) > LONG_STRAIGHT_THRESHOLD)
        .count()
}

fn compute_symmetry(segs: &[Seg], total_len: f64, centroid_x: f64) -> (f32, f32) {
    if total_len < f64::EPSILON {
        return (0.0, 0.0);
    }
    let mut top_len = 0.0f64;
    let mut bot_len = 0.0f64;
    let mut right_len = 0.0f64;
    let mut left_len = 0.0f64;

    for &(a, b, _, _) in segs {
        let len = seg_length(a, b);
        let mid_y = (a.y + b.y) * 0.5;
        let mid_x = (a.x + b.x) * 0.5;
        if mid_y >= 0.5 { top_len += len; } else { bot_len += len; }
        if mid_x >= centroid_x { right_len += len; } else { left_len += len; }
    }

    let vert = ((top_len - bot_len) / total_len) as f32;
    let horiz = ((right_len - left_len) / total_len) as f32;
    (vert, horiz)
}

fn orientation(p: Coord<f64>, q: Coord<f64>, r: Coord<f64>) -> i8 {
    let val = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
    if val.abs() < 1e-10 { 0 } else if val > 0.0 { 1 } else { -1 }
}

fn segments_intersect(a: Coord<f64>, b: Coord<f64>, c: Coord<f64>, d: Coord<f64>) -> bool {
    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);
    o1 != 0 && o2 != 0 && o3 != 0 && o4 != 0 && o1 != o2 && o3 != o4
}

/// Divide the normalized bounding box into a 3×3 grid and compute the total
/// stroke length passing through each cell. Normalized so cells sum to 1.
/// Row-major order: [top-left … top-right, mid-left … mid-right, bot-left … bot-right]
fn compute_grid_cells(segs: &[Seg], max_x: f64) -> [f32; 9] {
    let mut cells = [0.0f64; 9];
    let mut total = 0.0f64;

    let bx = max_x.max(f64::EPSILON);
    let x_splits = [bx / 3.0, 2.0 * bx / 3.0];
    let y_splits = [1.0 / 3.0, 2.0 / 3.0];

    for &(a, b, _, _) in segs {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len < f64::EPSILON {
            continue;
        }

        // Collect all t values where this segment crosses a grid line.
        let mut ts = vec![0.0f64, 1.0];
        for &xb in &x_splits {
            if dx.abs() > f64::EPSILON {
                let t = (xb - a.x) / dx;
                if t > 0.0 && t < 1.0 {
                    ts.push(t);
                }
            }
        }
        for &yb in &y_splits {
            if dy.abs() > f64::EPSILON {
                let t = (yb - a.y) / dy;
                if t > 0.0 && t < 1.0 {
                    ts.push(t);
                }
            }
        }
        ts.sort_by(|p, q| p.partial_cmp(q).unwrap());

        for w in ts.windows(2) {
            let mid_t = (w[0] + w[1]) * 0.5;
            let mid_x = a.x + mid_t * dx;
            let mid_y = a.y + mid_t * dy;

            let col = if mid_x < x_splits[0] { 0usize } else if mid_x < x_splits[1] { 1 } else { 2 };
            // y=0 is bottom in font coords; row 0 = top of visual glyph = high y
            let row = if mid_y >= y_splits[1] { 0usize } else if mid_y >= y_splits[0] { 1 } else { 2 };

            let sub_len = seg_len * (w[1] - w[0]);
            cells[row * 3 + col] += sub_len;
            total += sub_len;
        }
    }

    let mut out = [0.0f32; 9];
    if total > f64::EPSILON {
        for (i, &v) in cells.iter().enumerate() {
            out[i] = (v / total) as f32;
        }
    }
    out
}

/// Count stroke endpoints (first and last point of each open LineString) in a 3×3 grid.
/// Closed loops contribute no endpoints. Normalized to sum=1 (or all-zero if no endpoints).
fn compute_endpoint_grid(mls: &MultiLineString<f64>, max_x: f64) -> [f32; 9] {
    let mut cells = [0.0f64; 9];
    let mut total = 0.0f64;

    let bx = max_x.max(f64::EPSILON);
    let x_splits = [bx / 3.0, 2.0 * bx / 3.0];
    let y_splits = [1.0 / 3.0, 2.0 / 3.0];

    for ls in &mls.0 {
        if ls.0.len() < 2 {
            continue;
        }
        let first = ls.0[0];
        let last = *ls.0.last().unwrap();
        let dx = first.x - last.x;
        let dy = first.y - last.y;
        let is_closed = (dx * dx + dy * dy).sqrt() < 1e-6;
        if is_closed {
            continue;
        }

        for pt in [first, last] {
            let col = if pt.x < x_splits[0] { 0usize } else if pt.x < x_splits[1] { 1 } else { 2 };
            let row = if pt.y >= y_splits[1] { 0usize } else if pt.y >= y_splits[0] { 1 } else { 2 };
            cells[row * 3 + col] += 1.0;
            total += 1.0;
        }
    }

    let mut out = [0.0f32; 9];
    if total > 0.0 {
        for (i, &v) in cells.iter().enumerate() {
            out[i] = (v / total) as f32;
        }
    }
    out
}

fn count_closed_loops(mls: &MultiLineString<f64>) -> usize {
    mls.0
        .iter()
        .filter(|ls| {
            if ls.0.len() < 3 {
                return false;
            }
            let first = ls.0[0];
            let last = *ls.0.last().unwrap();
            let dx = first.x - last.x;
            let dy = first.y - last.y;
            (dx * dx + dy * dy).sqrt() < 1e-6
        })
        .count()
}

fn count_self_intersections(segs: &[Seg], _mls: &MultiLineString<f64>) -> usize {
    let mut count = 0;
    for i in 0..segs.len() {
        for j in (i + 1)..segs.len() {
            let (a, b, ls_i, seg_i) = segs[i];
            let (c, d, ls_j, seg_j) = segs[j];
            // Skip segments that share an endpoint (adjacent in the same linestring)
            if ls_i == ls_j && seg_j == seg_i + 1 {
                continue;
            }
            if segments_intersect(a, b, c, d) {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::line_string;

    fn x_shape() -> MultiLineString<f64> {
        MultiLineString(vec![
            line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)],
            line_string![(x: 1.0, y: 0.0), (x: 0.0, y: 1.0)],
        ])
    }

    #[test]
    fn angle_bins_sum_to_one() {
        let mls = x_shape();
        let segs = segments(&mls);
        let (bins, _) = compute_angle_bins(&segs);
        let sum: f32 = bins.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "bins sum = {}", sum);
    }

    #[test]
    fn x_shape_has_one_crossing() {
        let mls = x_shape();
        let segs = segments(&mls);
        let crossings = count_self_intersections(&segs, &mls);
        assert_eq!(crossings, 1);
    }

    #[test]
    fn extract_returns_48_features() {
        let mls = x_shape();
        let fv = extract(&mls, 1.0).unwrap();
        assert_eq!(fv.to_array().len(), 48);
    }

    #[test]
    fn grid_cells_sum_to_one() {
        let mls = x_shape();
        let fv = extract(&mls, 1.0).unwrap();
        let sum: f32 = fv.grid_cells.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "grid cells sum = {}", sum);
    }

    #[test]
    fn endpoint_grid_sums_to_one() {
        let mls = x_shape();
        let fv = extract(&mls, 1.0).unwrap();
        let sum: f32 = fv.endpoint_grid.iter().sum();
        // x_shape has 4 open endpoints, so sum should be 1
        assert!((sum - 1.0).abs() < 1e-5, "endpoint grid sum = {}", sum);
    }
}
