use severian_ast::Span;
use severian_diagnostics::{
    Applicability, Diagnostic, DiagnosticLabel, DiagnosticSuggestion, SourceRange, TextEdit,
};
use severian_source::SourceMap;
use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum CompileError {
    Io(std::io::Error),
    Frontend {
        stage: &'static str,
        span: Span,
        message: String,
        source_path: PathBuf,
        source: String,
    },
    Ownership(String),
    Optimization(String),
    Verification(String),
    Package(String),
    Execution(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Io(error) => error.fmt(formatter),
            CompileError::Frontend {
                stage,
                span,
                message,
                source_path,
                source,
            } => formatter.write_str(&render_frontend(
                *stage,
                *span,
                message,
                source_path,
                source,
            )),
            CompileError::Ownership(message) => write!(formatter, "ownership error: {message}"),
            CompileError::Optimization(message) => {
                write!(formatter, "optimization error: {message}")
            }
            CompileError::Verification(message) => {
                write!(formatter, "compiler IR verification error: {message}")
            }
            CompileError::Package(message) => write!(formatter, "package error: {message}"),
            CompileError::Execution(message) => write!(formatter, "execution error: {message}"),
        }
    }
}

impl CompileError {
    /// Returns the source-level diagnostic when this error originated in the
    /// frontend. Build clients use the same structure as the terminal renderer
    /// so labels and suggested edits are not lost in JSON output.
    pub fn diagnostic(&self) -> Option<Diagnostic> {
        let Self::Frontend {
            stage,
            span,
            message,
            source_path,
            source,
        } = self
        else {
            return None;
        };
        Some(frontend_diagnostic(
            stage,
            *span,
            message,
            source_path,
            source,
        ))
    }
}

fn render_frontend(
    stage: &str,
    span: Span,
    raw_message: &str,
    source_path: &std::path::Path,
    source: &str,
) -> String {
    let diagnostic = frontend_diagnostic(stage, span, raw_message, source_path, source);
    let code = diagnostic.code.0.clone();
    let mut sources = SourceMap::new();
    sources.add(source_path, source);
    let mut rendered = severian_diagnostics::render::render(
        &diagnostic,
        Some(&sources),
        &severian_diagnostics::render::RenderOptions {
            color: false,
            context_lines: 0,
            show_codes: true,
        },
    );
    if severian_diagnostics::explain::explain(&code).is_some() {
        rendered.push_str(&format!(
            "\n\nFor more information:\n    sev explain {code}"
        ));
    }
    rendered
}

fn frontend_diagnostic(
    stage: &str,
    span: Span,
    raw_message: &str,
    source_path: &std::path::Path,
    source: &str,
) -> Diagnostic {
    let (code, raw_message) = frontend_code(stage, raw_message);
    let missing_argument = (code == "E000203")
        .then(|| missing_argument_details(&raw_message))
        .flatten();
    let tensor_dimensions = (code == "E002401")
        .then(|| tensor_dimension_details(&raw_message))
        .flatten();
    let message = if let Some((name, _)) = &missing_argument {
        format!("missing argument `{name}`")
    } else if tensor_dimensions.is_some() {
        "incompatible tensor dimensions".into()
    } else {
        raw_message.clone()
    };
    let mut range = source_range(source_path, source, span);
    if code == "E000104" {
        range.end_byte = range.start_byte;
        range.end_line = range.start_line;
        range.end_column = range.start_column;
    }
    let label = primary_label(&code, &message);
    let mut diagnostic = Diagnostic::error(code.clone(), message.clone())
        .with_label(DiagnosticLabel::primary(range.clone(), label));

    if code == "E000202" {
        if let Some((expected, actual)) = expected_and_actual(&message) {
            diagnostic = diagnostic.with_note(format!(
                "this expression has type `{actual}`, but this boundary requires `{expected}`"
            ));
            if expected == "int" && actual == "string" {
                let expression = source.get(span.start..span.end).unwrap_or("value");
                diagnostic = diagnostic.with_suggestion(DiagnosticSuggestion {
                    message: "convert the value with `int(...)`".into(),
                    applicability: Applicability::MaybeIncorrect,
                    edits: vec![TextEdit {
                        source: range.clone(),
                        replacement: format!("int({expression})"),
                    }],
                });
            }
        }
    } else if code == "E000203" {
        if let Some((name, expected)) = missing_argument {
            diagnostic = diagnostic.with_note(format!(
                "this call does not supply required parameter `{name}: {expected}`"
            ));
            if let Some(declaration) = parameter_declaration(source_path, source, &name) {
                diagnostic = diagnostic.with_label(DiagnosticLabel::secondary(
                    declaration,
                    format!("`{name}` is declared without a default"),
                ));
            }
            if let (Some((insertion, prefix)), Some(placeholder)) = (
                call_argument_insertion(source_path, source, span),
                placeholder(&expected),
            ) {
                diagnostic = diagnostic.with_suggestion(DiagnosticSuggestion {
                    message: format!("add `{name}`"),
                    applicability: Applicability::MaybeIncorrect,
                    edits: vec![TextEdit {
                        source: insertion,
                        replacement: format!("{prefix}{name} = {placeholder}"),
                    }],
                });
            }
        }
    } else if code == "E002401" {
        if let Some((left, right, requirement)) = tensor_dimensions {
            diagnostic = diagnostic
                .with_note(format!("left operand:  `{left}`"))
                .with_note(format!("right operand: `{right}`"))
                .with_note(format!("matrix multiplication requires `{requirement}`"))
                .with_help(
                    "reshape, transpose, or replace an operand so the contracting dimensions match",
                );
        }
    } else if code == "E000104" && message.contains("`:`") {
        let insertion = SourceRange {
            start_byte: span.start,
            end_byte: span.start,
            ..range.clone()
        };
        diagnostic = diagnostic.with_suggestion(DiagnosticSuggestion::machine_applicable(
            "insert `:`",
            insertion,
            ":",
        ));
    }

    diagnostic
}

fn missing_argument_details(message: &str) -> Option<(String, String)> {
    let rest = message.strip_prefix("missing argument `")?;
    let (name, rest) = rest.split_once("`; expected `")?;
    let expected = rest.strip_suffix('`')?;
    Some((name.to_owned(), expected.to_owned()))
}

fn tensor_dimension_details(message: &str) -> Option<(String, String, String)> {
    let rest = message.strip_prefix("incompatible tensor dimensions; left is `")?;
    let (left, rest) = rest.split_once("`; right is `")?;
    let (right, rest) = rest.split_once("`; requires `")?;
    let requirement = rest.strip_suffix('`')?;
    Some((left.into(), right.into(), requirement.into()))
}

fn parameter_declaration(
    path: &std::path::Path,
    source: &str,
    parameter: &str,
) -> Option<SourceRange> {
    source.match_indices(parameter).find_map(|(start, _)| {
        let end = start + parameter.len();
        let follows_annotation = source
            .get(end..)?
            .trim_start_matches([' ', '\t'])
            .starts_with(':');
        follows_annotation.then(|| source_range(path, source, Span::new(start, end)))
    })
}

fn call_argument_insertion(
    path: &std::path::Path,
    source: &str,
    span: Span,
) -> Option<(SourceRange, &'static str)> {
    let call = source.get(span.start..span.end)?;
    let close = call.rfind(')')?;
    let insertion = span.start + close;
    let open = call.find('(')?;
    let prefix = if call[open + 1..close].trim().is_empty() {
        ""
    } else {
        ", "
    };
    Some((source_range(path, source, Span::empty(insertion)), prefix))
}

fn placeholder(expected: &str) -> Option<&'static str> {
    match expected {
        "string" => Some("\"\""),
        "int" => Some("0"),
        "float" => Some("0.0"),
        "bool" => Some("false"),
        "list" => Some("[]"),
        "map" => Some("{}"),
        _ => None,
    }
}

fn frontend_code(stage: &str, message: &str) -> (String, String) {
    if message.len() > 8
        && message.starts_with('E')
        && message.as_bytes()[1..7].iter().all(u8::is_ascii_digit)
        && message.as_bytes()[7] == b':'
    {
        return (
            message[..7].to_owned(),
            message[8..].trim_start().to_owned(),
        );
    }
    let code = match stage {
        "lexer" => "E000100",
        "parser" if message.starts_with("expected `") => "E000104",
        "parser" => "E000100",
        "semantic" => "E000200",
        "ownership" => "E000300",
        _ => "E009900",
    };
    (code.into(), message.to_owned())
}

fn primary_label<'a>(code: &str, message: &'a str) -> &'a str {
    match code {
        "E000202" => "incompatible value",
        "E000203" => "required argument is absent",
        "E002401" => "contracting dimensions do not match",
        "E000104" => "syntax needs another token here",
        _ => message,
    }
}

fn expected_and_actual(message: &str) -> Option<(String, String)> {
    let message = message
        .strip_prefix("mismatched types: ")
        .unwrap_or(message);
    let values = message.strip_prefix("expected ")?;
    let (expected, actual) = values.split_once(", found ")?;
    Some((
        expected.trim_matches('`').to_ascii_lowercase(),
        actual.trim_matches('`').to_ascii_lowercase(),
    ))
}

fn source_range(path: &std::path::Path, source: &str, span: Span) -> SourceRange {
    let start = span.start.min(source.len());
    let end = span.end.min(source.len()).max(start);
    let prefix = source.get(..start).unwrap_or("");
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
    let column = source.get(line_start..start).unwrap_or("").chars().count() + 1;
    let end_prefix = source.get(..end).unwrap_or(prefix);
    let end_line = end_prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let end_line_start = end_prefix.rfind('\n').map_or(0, |index| index + 1);
    let end_column = source
        .get(end_line_start..end)
        .unwrap_or("")
        .chars()
        .count()
        + 1;
    SourceRange {
        file: path.to_path_buf(),
        start_byte: start,
        end_byte: end,
        start_line: Some(line as u32),
        start_column: Some(column as u32),
        end_line: Some(end_line as u32),
        end_column: Some(end_column as u32),
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
