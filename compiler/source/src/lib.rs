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

/// Owns the sources for one compilation and assigns each one a distinct,
/// deterministic identity in insertion order.
#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub const fn new() -> Self {
        Self { files: Vec::new() }
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> std::io::Result<SourceId> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        Ok(self.insert(path.to_owned(), text))
    }

    pub fn add_virtual(&mut self, name: impl Into<PathBuf>, text: impl Into<String>) -> SourceId {
        self.insert(name.into(), text.into())
    }

    pub fn get(&self, id: SourceId) -> Option<&SourceFile> {
        self.files.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    fn insert(&mut self, path: PathBuf, text: String) -> SourceId {
        let index = u32::try_from(self.files.len()).expect("source map exceeds u32 identities");
        let id = SourceId(index);
        self.files.push(SourceFile { id, path, text });
        id
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_ids_are_unique() {
        let mut sources = SourceMap::new();
        let first = sources.add_virtual("first.sev", "first := 1\n");
        let second = sources.add_virtual("second.sev", "second := 2\n");

        assert_ne!(first, second);
        assert_ne!(
            Span::new(first, 0, 1).source,
            Span::new(second, 0, 1).source
        );
        assert_eq!(sources.get(first).unwrap().path, Path::new("first.sev"));
        assert_eq!(sources.get(second).unwrap().path, Path::new("second.sev"));
    }
}
