use severian_abi::Target;
use severian_backend::{Artifact, BackendError};
use severian_diagnostics::Diagnostic;
use severian_source::SourceFile;
use severian_universal::UniversalContext;
use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum CompileError {
    Bootstrap(severian_bootstrap::BootstrapError),
    Diagnostic(Diagnostic),
    Lowering(severian_lowering::LoweringError),
    Backend(BackendError),
    External(severian_xxi::XxiError),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bootstrap(error) => write!(formatter, "primitive bootstrap failed: {error}"),
            Self::Diagnostic(diagnostic) => diagnostic.fmt(formatter),
            Self::Lowering(error) => write!(formatter, "lowering failed: {error}"),
            Self::Backend(error) => error.fmt(formatter),
            Self::External(error) => write!(formatter, "external interface failed: {error}"),
        }
    }
}

impl std::error::Error for CompileError {}

pub struct Compiler {
    context: UniversalContext,
    abi_target: Target,
}

impl Compiler {
    pub fn new(abi_target: Target) -> Result<Self, CompileError> {
        let context = severian_bootstrap::load().map_err(CompileError::Bootstrap)?;
        Ok(Self {
            context,
            abi_target,
        })
    }

    pub fn context(&self) -> &UniversalContext {
        &self.context
    }

    pub fn abi_target(&self) -> &Target {
        &self.abi_target
    }

    pub fn compile_source(
        &self,
        source: &SourceFile,
        output: &Path,
    ) -> Result<Artifact, CompileError> {
        let tokens = severian_lexer::scan(source).map_err(CompileError::Diagnostic)?;
        let ast = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
        severian_xxi::resolve(&ast, &self.context.types, &self.abi_target)
            .map_err(CompileError::External)?;
        let hir = severian_semantic::analyze(&ast, &self.context.types)
            .map_err(CompileError::Diagnostic)?;
        severian_ownership::validate(&hir).map_err(CompileError::Diagnostic)?;
        let mir = severian_mir::build(&hir);
        let lir = severian_lowering::lower(&mir, &self.context.types, &self.abi_target)
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
    Compiler::new(Target::host())?.compile_source(source, output)
}

pub fn compile_file(source: &Path, output: &Path) -> Result<Artifact, CompileError> {
    Compiler::new(Target::host())?.compile_file(source, output)
}
