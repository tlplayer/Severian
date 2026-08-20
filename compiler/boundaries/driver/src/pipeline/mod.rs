use severian_backend::{Artifact, BackendError};
use severian_diagnostics::Diagnostic;
use severian_source::SourceFile;
use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum CompileError {
    Diagnostic(Diagnostic),
    Backend(BackendError),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Diagnostic(diagnostic) => diagnostic.fmt(formatter),
            Self::Backend(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompileError {}

pub fn compile_source(source: &SourceFile, output: &Path) -> Result<Artifact, CompileError> {
    let tokens = severian_lexer::scan(source).map_err(CompileError::Diagnostic)?;
    let ast = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
    let hir = severian_semantic::analyze(&ast).map_err(CompileError::Diagnostic)?;
    severian_ownership::validate(&hir).map_err(CompileError::Diagnostic)?;
    let mir = severian_mir::build(&hir);
    let lowered = severian_lowering::lower(&mir, &hir.types)
        .map_err(|message| CompileError::Diagnostic(Diagnostic::new("E000950", message, None)))?;
    severian_backend::emit_executable(&lowered, output).map_err(CompileError::Backend)
}

pub fn compile_file(source: &Path, output: &Path) -> Result<Artifact, CompileError> {
    let source = SourceFile::load(source).map_err(|error| {
        CompileError::Diagnostic(Diagnostic::new(
            "E000001",
            format!("could not read source: {error}"),
            None,
        ))
    })?;
    compile_source(&source, output)
}
