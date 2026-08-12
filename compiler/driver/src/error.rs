use severian_ast::Span;
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
            } => {
                let (line, column, text, marker) = source_location(source, *span);
                write!(
                    formatter,
                    "{stage} error: {message}\n --> {}:{line}:{column}\n{line:>4} | {text}\n     | {marker}",
                    source_path.display()
                )
            }
            CompileError::Ownership(message) => write!(formatter, "ownership error: {message}"),
            CompileError::Optimization(message) => {
                write!(formatter, "optimization error: {message}")
            }
            CompileError::Package(message) => write!(formatter, "package error: {message}"),
            CompileError::Execution(message) => write!(formatter, "execution error: {message}"),
        }
    }
}

fn source_location(source: &str, span: Span) -> (usize, usize, &str, String) {
    let start = span.start.min(source.len());
    let end = span.end.min(source.len()).max(start);
    let prefix = source.get(..start).unwrap_or("");
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let line_start = prefix.rfind('\n').map_or(0, |index| index + 1);
    let line_end = source[start..]
        .find('\n')
        .map_or(source.len(), |length| start + length);
    let text = source.get(line_start..line_end).unwrap_or("");
    let column = source.get(line_start..start).unwrap_or("").chars().count() + 1;
    let marker_width = source
        .get(start..end.min(line_end))
        .unwrap_or("")
        .chars()
        .count()
        .max(1);
    let marker = format!("{}{}", " ".repeat(column - 1), "^".repeat(marker_width));
    (line, column, text, marker)
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
