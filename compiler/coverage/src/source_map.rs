use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CoverageRegionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourcePosition {
    pub line: u32,
    pub column: u32,
    pub byte: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub file: PathBuf,
    pub start: SourcePosition,
    pub end: SourcePosition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageRegion {
    pub id: CoverageRegionId,
    pub function: String,
    pub span: SourceSpan,
    pub kind: CoverageRegionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoverageRegionKind {
    Function,
    Statement,
    Branch,
    Condition,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageSourceMap {
    regions: BTreeMap<CoverageRegionId, CoverageRegion>,
}

impl CoverageSourceMap {
    pub fn insert(&mut self, region: CoverageRegion) -> Option<CoverageRegion> {
        self.regions.insert(region.id, region)
    }

    pub fn region(&self, id: CoverageRegionId) -> Option<&CoverageRegion> {
        self.regions.get(&id)
    }

    pub fn regions(&self) -> impl Iterator<Item = &CoverageRegion> {
        self.regions.values()
    }

    pub fn extend(&mut self, other: Self) {
        self.regions.extend(other.regions);
    }

    pub fn regions_for_file<'a>(
        &'a self,
        file: &'a Path,
    ) -> impl Iterator<Item = &'a CoverageRegion> + 'a {
        self.regions
            .values()
            .filter(move |region| region.span.file == file)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }

    pub fn load_json(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

/// Stable region id based on source identity rather than traversal order.
///
/// FNV-1a is used here because it is tiny, deterministic across Rust versions,
/// and this id is for coverage mapping rather than cryptographic integrity.
pub fn stable_region_id(
    function: &str,
    span: &SourceSpan,
    kind: CoverageRegionKind,
) -> CoverageRegionId {
    let mut hash = 0xcbf29ce484222325u64;

    fn add(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }

    add(&mut hash, function.as_bytes());
    add(&mut hash, span.file.to_string_lossy().as_bytes());
    add(&mut hash, &span.start.byte.to_le_bytes());
    add(&mut hash, &span.end.byte.to_le_bytes());
    add(&mut hash, &[kind as u8]);

    CoverageRegionId(hash)
}
