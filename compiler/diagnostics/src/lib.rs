#![forbid(unsafe_code)]

pub mod coverage;
pub mod dead_code;
pub mod doctor;
pub mod explain;
pub mod lint;
pub mod naming;
pub mod render;
pub mod verify;

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Allow,
    Note,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiagnosticCode(pub String);

impl From<&str> for DiagnosticCode {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRange {
    pub file: PathBuf,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: Option<u32>,
    pub start_column: Option<u32>,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: DiagnosticCode,
    pub message: String,
    pub help: Option<String>,
    pub source: Option<SourceRange>,
    pub related: Vec<Diagnostic>,
}

impl Diagnostic {
    pub fn warning(code: impl Into<DiagnosticCode>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            help: None,
            source: None,
            related: Vec::new(),
        }
    }

    pub fn error(code: impl Into<DiagnosticCode>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            help: None,
            source: None,
            related: Vec::new(),
        }
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn at(mut self, source: SourceRange) -> Self {
        self.source = Some(source);
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct DiagnosticBag {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBag {
    pub fn push(&mut self, diagnostic: Diagnostic) {
        if diagnostic.severity != Severity::Allow {
            self.diagnostics.push(diagnostic);
        }
    }

    pub fn extend(&mut self, diagnostics: impl IntoIterator<Item = Diagnostic>) {
        self.diagnostics.extend(
            diagnostics
                .into_iter()
                .filter(|diagnostic| diagnostic.severity != Severity::Allow),
        );
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity >= Severity::Error)
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Warning)
            .count()
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Error)
            .count()
    }
}
