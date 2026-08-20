use severian_backend::{Artifact, BackendError};
use severian_diagnostics::Diagnostic;
use severian_source::SourceFile;
use severian_universal::{TargetSpec, UniversalContext};
use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum CompileError {
    Bootstrap(severian_bootstrap::BootstrapError),
    Diagnostic(Diagnostic),
    Lowering(severian_lowering::LoweringError),
    Backend(BackendError),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bootstrap(error) => write!(formatter, "primitive bootstrap failed: {error}"),
            Self::Diagnostic(diagnostic) => diagnostic.fmt(formatter),
            Self::Lowering(error) => write!(formatter, "lowering failed: {error}"),
            Self::Backend(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompileError {}

pub struct Compiler {
    context: UniversalContext,
}

impl Compiler {
    pub fn new(target: TargetSpec) -> Result<Self, CompileError> {
        let context = severian_bootstrap::load(target).map_err(CompileError::Bootstrap)?;
        Ok(Self { context })
    }

    pub fn context(&self) -> &UniversalContext {
        &self.context
    }

    pub fn compile_source(
        &self,
        source: &SourceFile,
        output: &Path,
    ) -> Result<Artifact, CompileError> {
        let tokens = severian_lexer::scan(source).map_err(CompileError::Diagnostic)?;
        let ast = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
        let hir = severian_semantic::analyze(&ast, &self.context.types)
            .map_err(CompileError::Diagnostic)?;
        severian_ownership::validate(&hir).map_err(CompileError::Diagnostic)?;
        let mir = severian_mir::build(&hir);
        let lir = severian_lowering::lower(&mir, &self.context.types, &self.context.target)
            .map_err(CompileError::Lowering)?;
        severian_backend::emit_executable(&lir, output).map_err(CompileError::Backend)
    }

    pub fn compile_file(&self, source: &Path, output: &Path) -> Result<Artifact, CompileError> {
        let source = SourceFile::load(source).map_err(|error| {
            CompileError::Diagnostic(Diagnostic::new(
                "E000001",
                format!("could not read source: {error}"),
                None,
            ))
        })?;
        self.compile_source(&source, output)
    }
}

pub fn compile_source(source: &SourceFile, output: &Path) -> Result<Artifact, CompileError> {
    Compiler::new(TargetSpec::host())?.compile_source(source, output)
}

pub fn compile_file(source: &Path, output: &Path) -> Result<Artifact, CompileError> {
    Compiler::new(TargetSpec::host())?.compile_file(source, output)
}
