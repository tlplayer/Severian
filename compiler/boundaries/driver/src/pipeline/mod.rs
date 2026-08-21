use severian_backend::{Artifact, BackendError};
use severian_compile::{CompileContext, CompileHandler, CompilePlan, CompilerRegistry};
use severian_diagnostics::Diagnostic;
use severian_mir::Module as MirModule;
use severian_source::SourceFile;
use severian_target::TargetSpec;
use severian_universal::{CompilerId, UniversalContext};
use std::fmt;
use std::path::Path;

#[derive(Debug)]
pub enum CompileError {
    Bootstrap(severian_bootstrap::BootstrapError),
    Diagnostic(Diagnostic),
    Compile(severian_compile::CompileError),
    Lowering(severian_lowering::LoweringError),
    Mlir(severian_mlir::MlirError),
    Backend(BackendError),
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bootstrap(error) => write!(formatter, "primitive bootstrap failed: {error}"),
            Self::Diagnostic(diagnostic) => diagnostic.fmt(formatter),
            Self::Compile(error) => write!(formatter, "CompileType dispatch failed: {error}"),
            Self::Lowering(error) => write!(formatter, "lowering failed: {error}"),
            Self::Mlir(error) => write!(formatter, "MLIR generation failed: {error}"),
            Self::Backend(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompileError {}

pub struct Compiler {
    context: UniversalContext,
    target: TargetSpec,
    compile_handlers: CompilerRegistry,
}

impl Compiler {
    pub fn new(target: TargetSpec) -> Result<Self, CompileError> {
        let context = severian_bootstrap::load().map_err(CompileError::Bootstrap)?;
        Ok(Self::with_context(context, target))
    }

    pub fn with_context(context: UniversalContext, target: TargetSpec) -> Self {
        Self {
            context,
            target,
            compile_handlers: CompilerRegistry::new(),
        }
    }

    pub fn context(&self) -> &UniversalContext {
        &self.context
    }

    pub fn target(&self) -> &TargetSpec {
        &self.target
    }

    pub fn register_compile_handler(
        &mut self,
        compiler: CompilerId,
        handler: impl CompileHandler + 'static,
    ) -> Result<(), CompileError> {
        self.compile_handlers
            .register(compiler, handler)
            .map_err(CompileError::Compile)
    }

    /// Orchestrates both MIR routes and returns one verified MLIR module.
    /// Custom handlers never depend on or invoke ordinary lowering directly.
    pub fn compile_mir_to_mlir(&self, mir: &MirModule) -> Result<String, CompileError> {
        let plan =
            severian_compile::plan(mir, &self.context.types).map_err(CompileError::Compile)?;
        self.compile_plan_to_mlir(&plan)
    }

    fn compile_plan_to_mlir(&self, plan: &CompilePlan) -> Result<String, CompileError> {
        let artifacts = self
            .compile_handlers
            .compile(
                plan,
                &CompileContext {
                    types: &self.context.types,
                    target: &self.target,
                },
            )
            .map_err(CompileError::Compile)?;
        let resumed = plan.resumed_mir();
        let lir = severian_lowering::lower(&resumed, &self.context.types, &self.target)
            .map_err(CompileError::Lowering)?;
        let ordinary = severian_mlir::render(&lir).map_err(CompileError::Mlir)?;
        severian_mlir::compose(&ordinary, &artifacts, &self.target).map_err(CompileError::Mlir)
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
        let plan =
            severian_compile::plan(&mir, &self.context.types).map_err(CompileError::Compile)?;
        if !plan.has_custom_regions() {
            let resumed = plan.resumed_mir();
            let lir = severian_lowering::lower(&resumed, &self.context.types, &self.target)
                .map_err(CompileError::Lowering)?;
            if severian_backend::supports_direct_lir(&lir) {
                return severian_backend::emit_executable(&lir, output)
                    .map_err(CompileError::Backend);
            }
        }
        let mlir = self.compile_plan_to_mlir(&plan)?;
        severian_backend::emit_mlir_executable(&mlir, &self.target.triple, output)
            .map_err(CompileError::Backend)
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
