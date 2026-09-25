#![forbid(unsafe_code)]

use severian_source::{SourceFile, Span};
use std::fmt;

/// Context required at a diagnostic's reporting boundary. Source errors need a
/// primary span; operation errors explicitly opt out of source coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticContext {
    pub package: String,
    pub stage: String,
    source_required: bool,
}

impl DiagnosticContext {
    pub fn source(package: impl Into<String>, stage: impl Into<String>) -> Self {
        Self { package: package.into(), stage: stage.into(), source_required: true }
    }

    pub fn operation(package: impl Into<String>, stage: impl Into<String>) -> Self {
        Self { package: package.into(), stage: stage.into(), source_required: false }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticContractError(pub Vec<String>);

impl fmt::Display for DiagnosticContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "incomplete diagnostic: {}", self.0.join("; "))
    }
}

impl std::error::Error for DiagnosticContractError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticLabel {
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub context: Option<DiagnosticContext>,
    pub code: &'static str,
    pub message: String,
    pub span: Option<Span>,
    pub labels: Box<Vec<DiagnosticLabel>>,
    pub notes: Box<Vec<String>>,
    pub help: Option<String>,
    pub sources: Box<Vec<SourceFile>>,
    pub additional: Box<Vec<Diagnostic>>,
}

impl Diagnostic {
    pub fn new(code: &'static str, message: impl Into<String>, span: Option<Span>) -> Self {
        Self {
            context: None,
            code,
            message: message.into(),
            span,
            labels: Box::default(),
            notes: Box::default(),
            help: None,
            sources: Box::default(),
            additional: Box::default(),
        }
    }

    /// Construct a source report with all mandatory information supplied.
    pub fn source_error(
        code: &'static str, message: impl Into<String>, span: Span,
        source: SourceFile, package: impl Into<String>, stage: impl Into<String>,
        help: impl Into<String>,
    ) -> Self {
        Self::new(code, message, Some(span))
            .with_context(DiagnosticContext::source(package, stage))
            .with_source(source).with_help(help)
    }

    pub fn operation_error(
        code: &'static str, message: impl Into<String>, package: impl Into<String>,
        stage: impl Into<String>, help: impl Into<String>,
    ) -> Self {
        Self::new(code, message, None)
            .with_context(DiagnosticContext::operation(package, stage)).with_help(help)
    }

    pub fn with_context(mut self, context: DiagnosticContext) -> Self {
        self.context = Some(context);
        self
    }

    /// Validate before serializing or displaying a report. Never discard the
    /// original failure merely because its producer violated this contract.
    pub fn validate(&self) -> Result<(), DiagnosticContractError> {
        let mut missing = Vec::new();
        if self.code.trim().is_empty() { missing.push("diagnostic code is empty".into()); }
        if self.message.trim().is_empty() { missing.push("cause is empty".into()); }
        match &self.context {
            Some(context) => {
                if context.package.trim().is_empty() { missing.push("package is empty".into()); }
                if context.stage.trim().is_empty() { missing.push("stage is empty".into()); }
                if context.source_required && self.span.is_none() { missing.push("primary source span is missing".into()); }
            }
            None => missing.push("package/stage context is missing".into()),
        }
        if self.help.as_ref().is_none_or(|help| help.trim().is_empty()) {
            missing.push("actionable help is missing".into());
        }
        for span in self.span.iter().chain(self.labels.iter().map(|label| &label.span)) {
            let candidates: Vec<_> = self.sources.iter().filter(|source| source.id == span.source).collect();
            if candidates.len() > 1 {
                missing.push(format!("source {} identifies conflicting snapshots: {}", span.source.0,
                    candidates.iter().map(|source| source.path.display().to_string()).collect::<Vec<_>>().join(", ")));
            }
            match self.source_for(*span) {
                Some(source) if span.start <= span.end && source.location(span.start).is_some()
                    && source.location(span.end).is_some() && !source.path.as_os_str().is_empty() => {}
                _ => missing.push(format!("source {} bytes {}..{} has no valid file/span mapping", span.source.0, span.start, span.end)),
            }
        }
        for label in self.labels.iter() {
            if label.message.trim().is_empty() { missing.push("related-location explanation is empty".into()); }
        }
        for (index, additional) in self.additional.iter().enumerate() {
            if let Err(error) = additional.validate() {
                missing.push(format!("additional diagnostic {index}: {error}"));
            }
        }
        if missing.is_empty() { Ok(()) } else { Err(DiagnosticContractError(missing)) }
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
        self.attach_source(&source);
        self
    }

    fn attach_source(&mut self, source: &SourceFile) {
        // Equal IDs with different snapshots are a producer error. Retain both
        // so validation can expose the collision instead of showing the wrong file.
        if !self.sources.iter().any(|known| known == source) { self.sources.push(source.clone()); }
        for diagnostic in self.additional.iter_mut() { diagnostic.attach_source(source); }
    }

    fn source_for(&self, span: Span) -> Option<&SourceFile> {
        let mut matches = self.sources.iter().filter(|source| source.id == span.source);
        let source = matches.next()?;
        if matches.next().is_some() || span.start > span.end { return None; }
        Some(source)
    }

    pub fn with_sources(mut self, sources: impl IntoIterator<Item = SourceFile>) -> Self {
        for source in sources {
            self.attach_source(&source);
        }
        self
    }

    pub fn with_additional(mut self, diagnostics: impl IntoIterator<Item = Diagnostic>) -> Self {
        for diagnostic in diagnostics {
            self.additional.push(diagnostic.with_sources(self.sources.iter().cloned()));
        }
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)?;
        if let Some(context) = &self.context {
            write!(formatter, "\n package: {}\n stage: {}", context.package, context.stage)?;
            if !context.source_required && self.span.is_none() {
                write!(formatter, "\n location: not applicable (operation failure)")?;
            }
        }
        if let Some(span) = self.span {
            if let Some(source) = self.source_for(span) {
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
                        let labels = self
                            .labels
                            .iter()
                            .filter(|label| label.span == span)
                            .map(|label| label.message.as_str())
                            .collect::<Vec<_>>();
                        write!(
                            formatter,
                            "\n{line_number:>width$} | {line}\n     | {indent}{markers}{label}",
                            line_number = start.line,
                            indent = " ".repeat(marker_start),
                            markers = "^".repeat(marker_len),
                            label = if labels.is_empty() { String::new() } else { format!(" {}", labels.join("; ")) },
                        )?;
                    }
                }
            }
        }
        // Related locations explain conflicts and must not disappear merely
        // because they differ from the primary span (or belong to another file).
        for label in self.labels.iter().filter(|label| Some(label.span) != self.span) {
            if let Some(source) = self.source_for(label.span) {
                if let Some(location) = source.location(label.span.start) {
                    write!(formatter, "\n related: {}:{}:{}: {}", source.path.display(), location.line, location.column, label.message)?;
                    if let Some(line) = source.line(location.line) { write!(formatter, "\n     | {line}")?; }
                    continue;
                }
            }
            write!(formatter, "\n related: source {} bytes {}..{}: {}", label.span.source.0, label.span.start, label.span.end, label.message)?;
        }
        if let Some(span) = self.span {
            if !self.source_for(span).is_some_and(|source| source.location(span.start).is_some() && source.location(span.end).is_some()) {
                write!(formatter, "\n --> source {} bytes {}..{} (source unavailable)", span.source.0, span.start, span.end)?;
            }
        }
        for note in self.notes.iter() {
            write!(formatter, "\n note: {note}")?;
        }
        if let Some(help) = &self.help {
            write!(formatter, "\n help: {help}")?;
        }
        if let Err(problem) = self.validate() {
            write!(formatter, "\n diagnostic contract: {problem}")?;
        }
        for diagnostic in self.additional.iter() {
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

    #[test]
    fn source_contract_keeps_both_conflicting_files_and_actionable_context() {
        let mut sources = severian_source::SourceMap::new();
        let first = sources.add_virtual("first.sev", "hashing = 1\n");
        let second = sources.add_virtual("second.sev", "hashing = 2\n");
        let error = Diagnostic::source_error("E000203", "duplicate hashing", Span::new(second, 0, 7),
            sources.get(second).unwrap().clone(), "syntax@0.1.0", "import planning", "remove or alias the duplicate import")
            .with_label(Span::new(first, 0, 7), "first introduced here")
            .with_source(sources.get(first).unwrap().clone());
        assert!(error.validate().is_ok());
        let rendered = error.to_string();
        assert!(rendered.contains("package: syntax@0.1.0"));
        assert!(rendered.contains("stage: import planning"));
        assert!(rendered.contains("--> second.sev:1:1"));
        assert!(rendered.contains("related: first.sev:1:1: first introduced here"));
        assert!(rendered.contains("help: remove or alias"));
        assert!(!rendered.contains("diagnostic contract:"));
    }

    #[test]
    fn missing_sources_and_invalid_spans_are_reported_without_losing_the_cause() {
        let error = Diagnostic::new("E000203", "duplicate hashing", Some(Span::new(SourceId(42), 3, 8)));
        assert!(error.validate().is_err());
        let rendered = error.to_string();
        assert!(rendered.contains("E000203: duplicate hashing"));
        assert!(rendered.contains("source 42 bytes 3..8"));
        assert!(rendered.contains("diagnostic contract:"));
        let source = SourceFile::virtual_source("bad.sev", "abc");
        let invalid = Diagnostic::source_error("E000203", "invalid span", Span::new(source.id, 2, 1),
            source, "example", "parse", "correct the span producer");
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn operation_errors_explicitly_have_no_source_location() {
        let error = Diagnostic::operation_error("E000001", "cannot open package manifest", "example", "package loading", "check the manifest path and permissions");
        assert!(error.validate().is_ok());
        assert!(error.to_string().contains("location: not applicable (operation failure)"));
    }

    #[test]
    fn conflicting_source_ids_do_not_misidentify_the_error_file() {
        let first = SourceFile::virtual_source("first.sev", "x\n");
        let second = SourceFile::virtual_source("second.sev", "y\n");
        let error = Diagnostic::source_error("E000203", "duplicate x", Span::new(first.id, 0, 1),
            first, "example", "resolve", "rename x").with_source(second);
        assert!(error.validate().is_err());
        let rendered = error.to_string();
        assert!(!rendered.contains("--> first.sev"));
        assert!(!rendered.contains("--> second.sev"));
        assert!(rendered.contains("source 0 bytes 0..1"));
        assert!(rendered.contains("diagnostic contract:"));
    }

    #[test]
    fn source_attachment_reaches_nested_errors_regardless_of_attachment_order() {
        let source = SourceFile::virtual_source("nested.sev", "x\n");
        let leaf = Diagnostic::new("E000203", "duplicate x", Some(Span::new(source.id, 0, 1)))
            .with_context(DiagnosticContext::source("example", "resolve")).with_help("rename x");
        let middle = Diagnostic::operation_error("E000001", "module failed", "example", "resolve", "fix the nested error")
            .with_additional([leaf]);
        let outer = Diagnostic::operation_error("E000001", "build failed", "example", "build", "fix the nested error")
            .with_source(source.clone()).with_additional([middle.clone()]);
        assert!(outer.validate().is_ok());
        assert!(outer.to_string().contains("--> nested.sev:1:1"));
        let reverse = Diagnostic::operation_error("E000001", "build failed", "example", "build", "fix the nested error")
            .with_additional([middle]).with_source(source);
        assert!(reverse.validate().is_ok());
        assert!(reverse.to_string().contains("--> nested.sev:1:1"));
    }
}
