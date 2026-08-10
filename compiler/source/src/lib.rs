#![forbid(unsafe_code)]

use severian_ast::Span;
use std::{fmt, path::{Path, PathBuf}};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceSpan {
    pub file: FileId,
    pub bytes: ByteRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    pub line: u32,
    pub column: u32,
    pub byte: usize,
}

#[derive(Debug, Clone)]
pub struct SourceFile {
    path: PathBuf,
    source: String,
    line_starts: Vec<usize>,
}

impl SourceFile {
    fn new(path: PathBuf, source: String) -> Self {
        let mut line_starts = vec![0];
        line_starts.extend(
            source.bytes().enumerate().filter_map(|(index, byte)|
                (byte == b'\n').then_some(index + 1)
            ),
        );
        Self { path, source, line_starts }
    }

    pub fn path(&self) -> &Path { &self.path }

    pub fn source(&self) -> &str { &self.source }

    pub fn line_count(&self) -> usize { self.line_starts.len() }

    pub fn line(&self, line: u32) -> Option<&str> {
        let index = usize::try_from(line.checked_sub(1)?).ok()?;
        let start = *self.line_starts.get(index)?;
        let end = self.line_starts.get(index + 1).copied().unwrap_or(self.source.len());
        Some(self.source[start..end].trim_end_matches(['\n', '\r']))
    }

    pub fn location(&self, byte: usize) -> Option<SourceLocation> {
        if byte > self.source.len() || !self.source.is_char_boundary(byte) { return None; }
        let line_index = self.line_starts.partition_point(|start| *start <= byte).saturating_sub(1);
        let line_start = self.line_starts[line_index];
        Some(SourceLocation {
            line: line_index as u32 + 1,
            column: self.source[line_start..byte].chars().count() as u32 + 1,
            byte,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> Self { Self::default() }

    pub fn add(&mut self, path: impl Into<PathBuf>, source: impl Into<String>) -> FileId {
        let id = FileId(self.files.len());
        self.files.push(SourceFile::new(path.into(), source.into()));
        id
    }

    pub fn file(&self, id: FileId) -> Option<&SourceFile> { self.files.get(id.0) }

    pub fn file_by_path(&self, path: &Path) -> Option<&SourceFile> {
        self.files.iter().find(|file| file.path() == path)
    }

    pub fn from_ast_span(&self, file: FileId, span: Span) -> Result<SourceSpan, SourceMapError> {
        let source = self.file(file).ok_or(SourceMapError::UnknownFile(file))?;
        if span.start > span.end || span.end > source.source().len() {
            return Err(SourceMapError::InvalidSpan { file, start: span.start, end: span.end });
        }
        Ok(SourceSpan { file, bytes: ByteRange { start: span.start, end: span.end } })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceMapError {
    UnknownFile(FileId),
    InvalidSpan { file: FileId, start: usize, end: usize },
}

impl fmt::Display for SourceMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFile(file) => write!(formatter, "unknown source file {}", file.0),
            Self::InvalidSpan { file, start, end } =>
                write!(formatter, "invalid source span {start}..{end} for file {}", file.0),
        }
    }
}

impl std::error::Error for SourceMapError {}
