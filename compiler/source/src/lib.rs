#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub source: SourceId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn new(source: SourceId, start: u32, end: u32) -> Self {
        Self { source, start, end }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub id: SourceId,
    pub path: PathBuf,
    pub text: String,
}

impl SourceFile {
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        Ok(Self {
            id: SourceId(0),
            path: path.to_owned(),
            text: std::fs::read_to_string(path)?,
        })
    }

    pub fn virtual_source(name: impl Into<PathBuf>, text: impl Into<String>) -> Self {
        Self {
            id: SourceId(0),
            path: name.into(),
            text: text.into(),
        }
    }
}
