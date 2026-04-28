use std::path::{Path, PathBuf};

use fontdb::Database;
use geo::{Coord, LineString, MultiLineString};
use lyon_path::geom::point;
use ttf_parser::OutlineBuilder;

use crate::error::RecognitionError;

/// Discover all font files registered in the system font database.
pub fn list_system_fonts() -> Vec<PathBuf> {
    let mut db = Database::new();
    db.load_system_fonts();
    let mut paths: Vec<PathBuf> = db
        .faces()
        .filter_map(|face| {
            if let fontdb::Source::File(p) = &face.source {
                Some(p.clone())
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

/// Render a glyph from a font file as a MultiLineString<f64>.
/// Bezier curves are flattened to line segments with the given tolerance.
pub fn render_glyph(
    font_path: &Path,
    face_index: u32,
    character: char,
    flatten_tolerance: f32,
) -> Result<MultiLineString<f64>, RecognitionError> {
    let data = std::fs::read(font_path)?;
    let face = ttf_parser::Face::parse(&data, face_index)
        .map_err(|e| RecognitionError::Font(e.to_string()))?;

    let glyph_id = face
        .glyph_index(character)
        .ok_or_else(|| RecognitionError::Font(format!("no glyph for '{character}'")))?;

    let mut builder = GlyphOutlineBuilder::new();
    face.outline_glyph(glyph_id, &mut builder)
        .ok_or_else(|| RecognitionError::EmptyInput)?;

    let path = builder.finish();
    let mls = flatten_path_to_multilinestring(&path, flatten_tolerance);

    if mls.0.is_empty() {
        return Err(RecognitionError::EmptyInput);
    }
    Ok(mls)
}

/// Parse an SVG file and extract all path geometry as a MultiLineString.
pub fn render_svg(svg_path: &Path, flatten_tolerance: f32) -> Result<MultiLineString<f64>, RecognitionError> {
    let data = std::fs::read_to_string(svg_path)?;
    let tree = usvg::Tree::from_str(&data, &usvg::Options::default())
        .map_err(|e| RecognitionError::Font(e.to_string()))?;

    let mut all_linestrings: Vec<LineString<f64>> = Vec::new();

    collect_node_paths(tree.root(), flatten_tolerance, &mut all_linestrings);

    if all_linestrings.is_empty() {
        return Err(RecognitionError::EmptyInput);
    }
    Ok(MultiLineString(all_linestrings))
}

fn collect_node_paths(
    node: &usvg::Group,
    tolerance: f32,
    out: &mut Vec<LineString<f64>>,
) {
    for child in node.children() {
        match child {
            usvg::Node::Path(path) => {
                let mls = usvg_path_to_multilinestring(path, tolerance);
                out.extend(mls.0);
            }
            usvg::Node::Group(group) => {
                collect_node_paths(group, tolerance, out);
            }
            _ => {}
        }
    }
}

fn usvg_path_to_multilinestring(path: &usvg::Path, tolerance: f32) -> MultiLineString<f64> {
    let mut builder = lyon_path::Path::builder();
    let data = path.data();

    for segment in data.segments() {
        match segment {
            usvg::tiny_skia_path::PathSegment::MoveTo(p) => {
                builder.begin(point(p.x, p.y));
            }
            usvg::tiny_skia_path::PathSegment::LineTo(p) => {
                builder.line_to(point(p.x, p.y));
            }
            usvg::tiny_skia_path::PathSegment::QuadTo(ctrl, end) => {
                builder.quadratic_bezier_to(point(ctrl.x, ctrl.y), point(end.x, end.y));
            }
            usvg::tiny_skia_path::PathSegment::CubicTo(c1, c2, end) => {
                builder.cubic_bezier_to(
                    point(c1.x, c1.y),
                    point(c2.x, c2.y),
                    point(end.x, end.y),
                );
            }
            usvg::tiny_skia_path::PathSegment::Close => {
                builder.close();
            }
        }
    }

    let lyon_path = builder.build();
    flatten_path_to_multilinestring(&lyon_path, tolerance)
}

fn flatten_path_to_multilinestring(
    path: &lyon_path::Path,
    tolerance: f32,
) -> MultiLineString<f64> {
    use lyon_path::iterator::PathIterator;
    use lyon_path::Event;

    let mut linestrings: Vec<LineString<f64>> = Vec::new();
    let mut current: Vec<Coord<f64>> = Vec::new();

    for event in path.iter().flattened(tolerance) {
        match event {
            Event::Begin { at } => {
                if current.len() >= 2 {
                    linestrings.push(LineString(current.drain(..).collect()));
                } else {
                    current.clear();
                }
                current.push(Coord { x: at.x as f64, y: at.y as f64 });
            }
            Event::Line { to, .. } => {
                current.push(Coord { x: to.x as f64, y: to.y as f64 });
            }
            Event::End { last, first, close } => {
                current.push(Coord { x: last.x as f64, y: last.y as f64 });
                if close {
                    current.push(Coord { x: first.x as f64, y: first.y as f64 });
                }
                if current.len() >= 2 {
                    linestrings.push(LineString(current.drain(..).collect()));
                } else {
                    current.clear();
                }
            }
            _ => {}
        }
    }

    if current.len() >= 2 {
        linestrings.push(LineString(current));
    }

    MultiLineString(linestrings)
}

/// Collects glyph outline events into a lyon Path.
struct GlyphOutlineBuilder {
    builder: lyon_path::path::Builder,
    started: bool,
}

impl GlyphOutlineBuilder {
    fn new() -> Self {
        Self {
            builder: lyon_path::Path::builder(),
            started: false,
        }
    }

    fn finish(self) -> lyon_path::Path {
        self.builder.build()
    }
}

impl OutlineBuilder for GlyphOutlineBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.started {
            self.builder.close();
        }
        self.builder.begin(point(x, y));
        self.started = true;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(point(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(point(x1, y1), point(x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(point(x1, y1), point(x2, y2), point(x, y));
    }

    fn close(&mut self) {
        self.builder.close();
        self.started = false;
    }
}
