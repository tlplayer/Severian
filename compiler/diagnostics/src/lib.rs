#![forbid(unsafe_code)]

use severian_source::{SourceFile, Span};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLabel {
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub message: String,
    pub span: Option<Span>,
    pub labels: Vec<DiagnosticLabel>,
    pub notes: Vec<String>,
    pub help: Option<String>,
    pub sources: Vec<SourceFile>,
    pub additional: Vec<Diagnostic>,
}

impl Diagnostic {
    pub fn new(code: &'static str, message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            labels: Vec::new(),
            notes: Vec::new(),
            help: None,
            sources: Vec::new(),
            additional: Vec::new(),
        }
    }

    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel {
            span,
            message: message.into(),
        });
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn with_source(mut self, source: SourceFile) -> Self {
        if !self.sources.iter().any(|known| known.id == source.id) {
            self.sources.push(source.clone());
        }
        for diagnostic in &mut self.additional {
            if !diagnostic.sources.iter().any(|known| known.id == source.id) {
                diagnostic.sources.push(source.clone());
            }
        }
        self
    }

    pub fn with_sources(mut self, sources: impl IntoIterator<Item = SourceFile>) -> Self {
        for source in sources {
            if !self.sources.iter().any(|known| known.id == source.id) {
                self.sources.push(source.clone());
            }
            for diagnostic in &mut self.additional {
                if !diagnostic.sources.iter().any(|known| known.id == source.id) {
                    diagnostic.sources.push(source.clone());
                }
            }
        }
        self
    }

    pub fn with_additional(mut self, diagnostics: impl IntoIterator<Item = Diagnostic>) -> Self {
        self.additional.extend(diagnostics);
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)?;
        if let Some(span) = self.span {
            if let Some(source) = self.sources.iter().find(|source| source.id == span.source) {
                if let (Some(start), Some(end)) =
                    (source.location(span.start), source.location(span.end))
                {
                    write!(
                        formatter,
                        "\n --> {}:{}:{}\n     |",
                        source.path.display(),
                        start.line,
                        start.column
                    )?;
                    if let Some(line) = source.line(start.line) {
                        let width = start.line.to_string().len().max(4);
                        let marker_start = start.column.saturating_sub(1) as usize;
                        let marker_len = if start.line == end.line {
                            end.column.saturating_sub(start.column).max(1) as usize
                        } else {
                            1
                        };
                        let label = self
                            .labels
                            .iter()
                            .find(|label| label.span == span)
                            .map(|label| label.message.as_str());
                        write!(
                            formatter,
                            "\n{line_number:>width$} | {line}\n     | {indent}{markers}{label}",
                            line_number = start.line,
                            indent = " ".repeat(marker_start),
                            markers = "^".repeat(marker_len),
                            label = label.map_or(String::new(), |label| format!(" {label}")),
                        )?;
                    }
                }
            }
        }
        for note in &self.notes {
            write!(formatter, "\n note: {note}")?;
        }
        if let Some(help) = &self.help {
            write!(formatter, "\n help: {help}")?;
        }
        for diagnostic in &self.additional {
            write!(formatter, "\n\n{diagnostic}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Diagnostic {}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_source::{SourceFile, SourceId};

    #[test]
    fn display_renders_source_labels_notes_and_help() {
        let source = SourceFile::virtual_source("example.sev", "value = .\n");
        let span = Span::new(SourceId(0), 8, 9);
        let rendered = Diagnostic::new("E000111", "expected an expression", Some(span))
            .with_label(span, "expression starts here")
            .with_note("a value is required")
            .with_help("remove the dot")
            .with_source(source)
            .to_string();
        assert!(rendered.contains("--> example.sev:1:9"));
        assert!(rendered.contains("1 | value = ."));
        assert!(rendered.contains("^ expression starts here"));
        assert!(rendered.contains("note: a value is required"));
        assert!(rendered.contains("help: remove the dot"));
    }
}
