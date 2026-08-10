use severian_ast::Span;
use std::fmt;

#[derive(Debug)]
pub enum CompileError {
    Io(std::io::Error),
    Frontend {
        stage: &'static str,
        span: Span,
        message: String,
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
            } => write!(
                formatter,
                "{stage} error at bytes {}..{}: {message}",
                span.start, span.end
            ),
            CompileError::Ownership(message) => write!(formatter, "ownership error: {message}"),
            CompileError::Optimization(message) => {
                write!(formatter, "optimization error: {message}")
            }
            CompileError::Package(message) => write!(formatter, "package error: {message}"),
            CompileError::Execution(message) => write!(formatter, "execution error: {message}"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
