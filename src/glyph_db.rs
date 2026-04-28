use std::sync::OnceLock;

use geo::MultiLineString;
use rand::seq::SliceRandom;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

use crate::augment;
use crate::error::RecognitionError;
use crate::features::{self, FEATURE_DIM};
use crate::fonts;
use crate::normalize;

static GLYPH_DB: OnceLock<GlyphDatabase> = OnceLock::new();

const DEFAULT_CHARS: &str =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// ── rstar point type ────────────────────────────────────────────────────────

/// Newtype wrapping the 30-float feature array so we can implement rstar::Point.
#[derive(Clone, Debug, PartialEq)]
pub struct FeaturePoint(pub [f32; FEATURE_DIM]);

impl rstar::Point for FeaturePoint {
    type Scalar = f32;
    const DIMENSIONS: usize = FEATURE_DIM;

    fn generate(mut generator: impl FnMut(usize) -> f32) -> Self {
        Self(std::array::from_fn(|i| generator(i)))
    }

    fn nth(&self, index: usize) -> f32 {
        self.0[index]
    }

    fn nth_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

// ── R-tree entry ─────────────────────────────────────────────────────────────

pub struct GlyphEntry {
    pub point: FeaturePoint,
    pub character: char,
}

impl RTreeObject for GlyphEntry {
    type Envelope = AABB<FeaturePoint>;

    fn envelope(&self) -> AABB<FeaturePoint> {
        AABB::from_point(self.point.clone())
    }
}

impl PointDistance for GlyphEntry {
    fn distance_2(&self, point: &FeaturePoint) -> f32 {
        self.point
            .0
            .iter()
            .zip(point.0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }
}

// ── public types ─────────────────────────────────────────────────────────────

pub struct GlyphDbConfig {
    pub chars: Vec<char>,
    pub fonts_per_char: usize,
    pub flatten_tolerance: f32,
    pub simplify_epsilon: f64,
}

impl Default for GlyphDbConfig {
    fn default() -> Self {
        Self {
            chars: DEFAULT_CHARS.chars().collect(),
            fonts_per_char: 50,
            flatten_tolerance: 0.5,
            simplify_epsilon: 0.5,
        }
    }
}

pub struct CharMatch {
    pub character: char,
    /// Euclidean distance in 30-dimensional feature space.
    pub distance: f32,
}

// ── singleton ─────────────────────────────────────────────────────────────────

pub struct GlyphDatabase {
    tree: RTree<GlyphEntry>,
    total: usize,
}

impl GlyphDatabase {
    /// Build and install the singleton. Idempotent — a second call returns
    /// the already-built database without rebuilding.
    pub fn init(config: &GlyphDbConfig) -> Result<&'static GlyphDatabase, RecognitionError> {
        if let Some(db) = GLYPH_DB.get() {
            return Ok(db);
        }
        let db = Self::build(config)?;
        Ok(GLYPH_DB.get_or_init(|| db))
    }

    /// Returns the singleton if already initialized.
    pub fn global() -> Option<&'static GlyphDatabase> {
        GLYPH_DB.get()
    }

    /// Number of glyphs indexed.
    pub fn len(&self) -> usize {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    pub fn entries(&self) -> Vec<&GlyphEntry> {
        self.tree.iter().collect()
    }

    /// Find the `top_n` nearest glyphs in feature space.
    /// Returns matches sorted ascending by Euclidean distance.
    pub fn top_matches(
        &self,
        mls: &MultiLineString<f64>,
        top_n: usize,
    ) -> Result<Vec<CharMatch>, RecognitionError> {
        let ar = normalize::aspect_ratio(mls);
        let normed = normalize::normalize(mls)?;
        let fv = features::extract(&normed, ar)?;
        let query = FeaturePoint(fv.to_array());

        let matches = self
            .tree
            .nearest_neighbor_iter_with_distance_2(&query)
            .take(top_n)
            .map(|(entry, dist_sq)| CharMatch {
                character: entry.character,
                distance: dist_sq.sqrt(),
            })
            .collect();

        Ok(matches)
    }

    fn build(config: &GlyphDbConfig) -> Result<GlyphDatabase, RecognitionError> {
        let all_fonts = fonts::list_system_fonts();
        println!(
            "Building glyph database ({} chars × {} fonts, {} total font files found)...",
            config.chars.len(),
            config.fonts_per_char,
            all_fonts.len()
        );

        let mut rng = rand::rng();
        let mut entries: Vec<GlyphEntry> = Vec::new();

        for &ch in &config.chars {
            let mut font_list = all_fonts.clone();
            font_list.shuffle(&mut rng);
            font_list.truncate(config.fonts_per_char);

            let before = entries.len();
            for font_path in &font_list {
                let Ok(mls) = fonts::render_glyph(font_path, 0, ch, config.flatten_tolerance)
                else {
                    continue;
                };
                let simplified = augment::simplify(&mls, config.simplify_epsilon);
                let ar = normalize::aspect_ratio(&simplified);
                let Ok(normed) = normalize::normalize(&simplified) else { continue };
                let Ok(fv) = features::extract(&normed, ar) else { continue };
                entries.push(GlyphEntry {
                    point: FeaturePoint(fv.to_array()),
                    character: ch,
                });
            }
            println!("  '{}': {} glyphs", ch, entries.len() - before);
        }

        if entries.is_empty() {
            return Err(RecognitionError::EmptyInput);
        }

        let total = entries.len();
        println!("Building R-tree from {} glyphs (bulk load)...", total);
        // bulk_load builds an optimally packed tree in one pass — faster than N inserts.
        let tree = RTree::bulk_load(entries);
        println!("R-tree ready.");

        Ok(GlyphDatabase { tree, total })
    }
}
