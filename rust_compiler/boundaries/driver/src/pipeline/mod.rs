use severian_backend::{Artifact, BackendError};
use severian_compile::{
    CompileContext, CompileHandler, CompilePlan, CompilerRegistry, VerifiedCompiledRegionArtifact,
};
use severian_diagnostics::Diagnostic;
use severian_mir::{CfgStatement, Module as MirModule};
use severian_source::{SourceFile, SourceId};
use severian_target::TargetSpec;
use severian_universal::{CompilerId, UniversalContext};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub enum CompileError {
    Bootstrap(severian_bootstrap::BootstrapError),
    Diagnostic(Diagnostic),
    Compile(severian_compile::CompileError),
    MirVerify(severian_mir::VerifyError),
    MirPass(severian_mir::PassError),
    Lowering(severian_lowering::LoweringError),
    Mlir(severian_mlir::MlirError),
    Backend(BackendError),
    Component(String),
    AgentIr(String),
    NativeLink(String),
}

/// A compiler representation that can be inspected without producing or
/// executing a native artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitStage {
    Ast,
    Hir,
    Mir,
    Lir,
    Mlir,
    AgentIr,
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bootstrap(error) => write!(formatter, "primitive bootstrap failed: {error}"),
            Self::Diagnostic(diagnostic) => diagnostic.fmt(formatter),
            Self::Compile(error) => write!(formatter, "CompileType dispatch failed: {error}"),
            Self::MirVerify(error) => write!(formatter, "MIR verification failed: {error}"),
            Self::MirPass(error) => error.fmt(formatter),
            Self::Lowering(error) => write!(formatter, "lowering failed: {error}"),
            Self::Mlir(error) => write!(formatter, "MLIR generation failed: {error}"),
            Self::Backend(error) => error.fmt(formatter),
            Self::Component(error) => write!(formatter, "component provisioning failed: {error}"),
            Self::AgentIr(error) => write!(formatter, "Agent IR emission failed: {error}"),
            Self::NativeLink(error) => formatter.write_str(error),
        }
    }
}

impl std::error::Error for CompileError {}

impl CompileError {
    fn with_source(self, source: SourceFile) -> Self {
        match self {
            Self::Diagnostic(diagnostic) => Self::Diagnostic(diagnostic.with_source(source)),
            error => error,
        }
    }
}

pub struct Compiler {
    context: UniversalContext,
    target: TargetSpec,
    compile_handlers: CompilerRegistry,
    packages: Option<severian_modules::PackageGraph>,
    coverage: bool,
    max_errors: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTest {
    pub name: String,
    pub modes: Vec<severian_mir::TestMode>,
    pub execution: TestExecution,
    pub expectations: Vec<severian_mir::TestExpectation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoutedProgram {
    pub host_mlir: String,
    pub gpu_kernels: Vec<severian_compile::VerifiedGpuKernelBundle>,
    pub tensor_jit_source: String,
    pub tensor_jit_requires_gpu: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestExecution {
    Executable(Artifact),
    Compiler { failure: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompileMode {
    Build,
    Test,
}

impl Compiler {
    pub fn new(target: TargetSpec) -> Result<Self, CompileError> {
        let context = severian_bootstrap::load().map_err(CompileError::Bootstrap)?;
        Ok(Self::with_context(context, target))
    }

    pub fn with_context(context: UniversalContext, target: TargetSpec) -> Self {
        let mut compile_handlers = CompilerRegistry::new();
        compile_handlers
            .register(
                severian_universal::tensor::compiler_id(),
                severian_tensor_compiler::TensorCompiler,
            )
            .expect("the built-in tensor compiler is registered once");
        Self {
            context,
            target,
            compile_handlers,
            packages: None,
            coverage: false,
            max_errors: 5,
        }
    }

    pub fn with_packages(mut self, packages: severian_modules::PackageGraph) -> Self {
        self.packages = Some(packages);
        self
    }

    pub fn with_coverage(mut self) -> Self {
        self.coverage = true;
        self
    }

    pub fn with_max_errors(mut self, max_errors: usize) -> Self {
        self.max_errors = max_errors.max(1);
        self
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
        self.compile_mir_to_routed_program(mir)
            .map(|program| program.host_mlir)
    }

    pub fn compile_mir_to_routed_program(
        &self,
        mir: &MirModule,
    ) -> Result<RoutedProgram, CompileError> {
        let plan =
            severian_compile::plan(mir, self.types_for(mir)).map_err(CompileError::Compile)?;
        self.compile_plan(&plan)
    }

    fn compile_plan(&self, plan: &CompilePlan) -> Result<RoutedProgram, CompileError> {
        let types = self.types_for(&plan.source);
        let target = crate::components::ensure_for_plan(plan, &self.target)
            .map_err(CompileError::Component)?;
        let artifacts = self
            .compile_handlers
            .compile(
                plan,
                &CompileContext {
                    types,
                    target: &target,
                },
            )
            .map_err(CompileError::Compile)?;
        let resumed = plan.resumed_mir();
        let lir =
            severian_lowering::lower(&resumed, types, &target).map_err(CompileError::Lowering)?;
        let ordinary = severian_mlir::render(&lir).map_err(CompileError::Mlir)?;
        compose_region_artifacts(&ordinary, artifacts, &target)
    }

    fn types_for<'a>(&'a self, mir: &'a MirModule) -> &'a severian_universal::TypeContext {
        mir.types.as_ref().unwrap_or(&self.context.types)
    }

    pub fn compile_source(
        &self,
        source: &SourceFile,
        output: &Path,
    ) -> Result<Artifact, CompileError> {
        let mir = self.check_source_to_mir(source)?;
        self.compile_mir(&mir, output, None)
    }

    fn check_source_to_mir(&self, source: &SourceFile) -> Result<MirModule, CompileError> {
        let tokens = severian_lexer::scan(source).map_err(|diagnostic| {
            CompileError::Diagnostic(diagnostic.with_source(source.clone()))
        })?;
        let ast = severian_parser::parse(&tokens).map_err(|diagnostic| {
            CompileError::Diagnostic(diagnostic.with_source(source.clone()))
        })?;
        let mut mir = self
            .check_ast_to_mir(&ast, CompileMode::Build, &module_name(&source.path))
            .map_err(|error| error.with_source(source.clone()))?;
        attach_assertion_locations(&mut mir, source);
        if !self.coverage {
            remove_module_coverage(&mut mir);
        }
        Ok(mir)
    }

    fn check_ast_to_mir(
        &self,
        ast: &severian_ast::Module,
        mode: CompileMode,
        module_name: &str,
    ) -> Result<MirModule, CompileError> {
        let source = SourceFile::virtual_source(format!("{module_name}.sev"), "");
        let graph = severian_modules::ModuleGraph {
            modules: vec![severian_modules::ResolvedModule {
                id: severian_modules::ModuleId(1),
                path: PathBuf::from(format!("{module_name}.sev")),
                source,
                package: severian_modules::PackageId(0),
                ast: ast.clone(),
                imports: Vec::new(),
            }],
        };
        self.check_graph_to_mir(graph, mode)
    }

    pub fn check_source(&self, source: &SourceFile) -> Result<(), CompileError> {
        self.check_source_to_mir(source).map(|_| ())
    }

    pub fn compile_file(&self, source: &Path, output: &Path) -> Result<Artifact, CompileError> {
        let mir = self.check_file_to_mir(source, CompileMode::Build)?;
        self.compile_mir(&mir, output, Some(source))
    }

    /// Run the normal checked compilation pipeline through `stage` and return
    /// a deterministic textual representation suitable for diagnostics and
    /// compiler tests.
    pub fn emit_file(&self, source: &Path, stage: EmitStage) -> Result<String, CompileError> {
        match stage {
            EmitStage::Ast => {
                let graph = self.resolve_modules(source)?;
                let mut output = String::new();
                for module in graph.modules {
                    output.push_str(&format!(
                        "// module {}\n{:#?}\n",
                        module.path.display(),
                        module.ast
                    ));
                }
                Ok(output)
            }
            EmitStage::Hir => {
                let (hir, _, _) = self.check_file_to_hir(source, CompileMode::Build)?;
                Ok(format!("{hir:#?}\n"))
            }
            EmitStage::Mir => {
                let mir = self.check_file_to_mir(source, CompileMode::Build)?;
                Ok(format!("{mir:#?}\n"))
            }
            EmitStage::Lir => {
                let mir = self.check_file_to_mir(source, CompileMode::Build)?;
                let types = self.types_for(&mir);
                let plan = severian_compile::plan(&mir, types).map_err(CompileError::Compile)?;
                let lir = severian_lowering::lower(&plan.resumed_mir(), types, &self.target)
                    .map_err(CompileError::Lowering)?;
                Ok(format!("{lir:#?}\n"))
            }
            EmitStage::Mlir => {
                let mir = self.check_file_to_mir(source, CompileMode::Build)?;
                let types = self.types_for(&mir);
                let plan = severian_compile::plan(&mir, types).map_err(CompileError::Compile)?;
                let artifacts = self
                    .compile_handlers
                    .compile(
                        &plan,
                        &CompileContext {
                            types,
                            target: &self.target,
                        },
                    )
                    .map_err(CompileError::Compile)?;
                let lir = severian_lowering::lower(&plan.resumed_mir(), types, &self.target)
                    .map_err(CompileError::Lowering)?;
                let ordinary = severian_mlir::render(&lir).map_err(CompileError::Mlir)?;
                let text = compose_region_artifacts(&ordinary, artifacts, &self.target)?.host_mlir;
                Ok(format!("{}\n", text.trim_end()))
            }
            EmitStage::AgentIr => Err(CompileError::AgentIr(
                "Agent IR is a directory artifact; use `sev build --emit agent-ir`".into(),
            )),
        }
    }

    /// Emits the compiler's existing semantic graph without introducing a
    /// second frontend or re-parsing a lower-level representation.
    pub fn emit_agent_ir(
        &self,
        source: &Path,
        root: &Path,
        output: &Path,
        package: &str,
    ) -> Result<(), CompileError> {
        let graph = self.resolve_modules(source)?;
        // Test mode preserves test declarations in the semantic graph. It uses
        // the same frontend, HIR, and MIR pipeline as `sev test`; Agent IR only
        // serializes those existing compiler representations.
        let (hir, sources, types) = self.check_graph_to_hir(graph.clone(), CompileMode::Test)?;
        let mir = self.check_hir_to_mir(hir.clone(), sources, types.clone())?;
        crate::agent_ir::write(output, package, root, &graph, &hir, &mir, &types)
            .map_err(CompileError::AgentIr)
    }

    pub fn check_file(&self, source: &Path) -> Result<(), CompileError> {
        self.check_file_to_mir(source, CompileMode::Build)
            .map(|_| ())
    }

    pub fn file_has_entry(&self, source: &Path) -> Result<bool, CompileError> {
        let root = std::fs::canonicalize(source).map_err(|error| {
            CompileError::Diagnostic(Diagnostic::new(
                "E000001",
                format!("could not read {}: {error}", source.display()),
                None,
            ))
        })?;
        let graph = self.resolve_modules(source)?;
        Ok(graph.modules.iter().any(|module| {
            module.path == root
                && module.ast.items.iter().any(|item| {
                    matches!(item, severian_ast::Item::Function(function) if function.name == "main")
                })
        }))
    }

    pub fn file_has_asserting_tests(&self, source: &Path) -> Result<bool, CompileError> {
        fn has_assert(statements: &[severian_ast::Statement]) -> bool {
            statements.iter().any(|statement| match statement {
                severian_ast::Statement::Assert { .. } => true,
                severian_ast::Statement::If {
                    then_block,
                    else_block,
                    ..
                } => has_assert(then_block) || has_assert(else_block),
                severian_ast::Statement::Match { cases, .. } => {
                    cases.iter().any(|case| has_assert(&case.body))
                }
                _ => false,
            })
        }
        let root = std::fs::canonicalize(source).map_err(|error| {
            CompileError::Diagnostic(Diagnostic::new(
                "E000001",
                format!("could not read {}: {error}", source.display()),
                None,
            ))
        })?;
        let graph = self.resolve_modules(source)?;
        Ok(graph.modules.iter().any(|module| {
            module.path == root
                && module.ast.items.iter().any(
                    |item| matches!(item, severian_ast::Item::Test(test) if has_assert(&test.body)),
                )
        }))
    }

    pub fn resolved_module_paths(
        &self,
        source: &Path,
    ) -> Result<BTreeSet<std::path::PathBuf>, CompileError> {
        Ok(self
            .resolve_modules(source)?
            .modules
            .into_iter()
            .map(|module| module.path)
            .collect())
    }

    pub fn routes_file(
        &self,
        source: &Path,
        tests: bool,
    ) -> Result<BTreeSet<String>, CompileError> {
        let mode = if tests {
            CompileMode::Test
        } else {
            CompileMode::Build
        };
        let mir = self.check_file_to_mir(source, mode)?;
        let plan =
            severian_compile::plan(&mir, self.types_for(&mir)).map_err(CompileError::Compile)?;
        Ok(self.routes(&plan))
    }

    pub fn coverage_points_file(
        &self,
        source: &Path,
        tests: bool,
    ) -> Result<BTreeSet<severian_mir::CoveragePoint>, CompileError> {
        let mode = if tests {
            CompileMode::Test
        } else {
            CompileMode::Build
        };
        let mir = self.check_file_to_mir(source, mode)?;
        let mut points = BTreeSet::new();
        collect_coverage_points(&mir.initializer, &mut points);
        for body in mir
            .functions
            .iter()
            .filter_map(|function| function.body.as_ref())
        {
            collect_coverage_points(body, &mut points);
        }
        Ok(points)
    }

    pub fn compile_tests_file(
        &self,
        source: &Path,
        output_directory: &Path,
    ) -> Result<Vec<CompiledTest>, CompileError> {
        let graph = self.resolve_modules(source)?;
        self.compile_tests_graph(graph, output_directory)
    }

    /// Resolves a source root to the same parsed module graph used by the
    /// ordinary test pipeline. Mutation tooling clones this graph and changes
    /// one AST node without ever rewriting a source file.
    pub fn resolve_test_graph(
        &self,
        source: &Path,
    ) -> Result<severian_modules::ModuleGraph, CompileError> {
        self.resolve_modules(source)
    }

    /// Compiles tests from an already-resolved module graph through the normal
    /// semantic, MIR, lowering, and backend pipeline.
    pub fn compile_tests_graph(
        &self,
        graph: severian_modules::ModuleGraph,
        output_directory: &Path,
    ) -> Result<Vec<CompiledTest>, CompileError> {
        let native_root = graph.modules.last().map(|module| module.path.clone());
        let compiler_results = self.compiler_test_results(&graph)?;
        let mir = self.check_graph_to_mir(graph, CompileMode::Test)?;
        let mut compiler_results = compiler_results.into_iter();
        mir.tests
            .iter()
            .enumerate()
            .map(|(index, test)| {
                if test.modes.contains(&severian_mir::TestMode::Compiler) {
                    return Ok(CompiledTest {
                        name: test.name.clone(),
                        modes: test.modes.clone(),
                        execution: TestExecution::Compiler {
                            failure: compiler_results
                                .next()
                                .expect("every compiler test has an evaluation result"),
                        },
                        expectations: test.expectations.clone(),
                    });
                }
                let selected_function =
                    test.expectations
                        .iter()
                        .find_map(|expectation| match expectation {
                            severian_mir::TestExpectation::Panics { function, .. } => {
                                Some(function)
                            }
                            _ => None,
                        });
                let selected_id = if let Some(name) = selected_function {
                    mir.functions
                        .iter()
                        .find(|function| function.name == *name)
                        .map(|function| function.id)
                        .ok_or_else(|| {
                            CompileError::Diagnostic(Diagnostic::new(
                                "E000217",
                                format!("panic test references unknown function `{name}`"),
                                None,
                            ))
                        })?
                } else {
                    test.function
                };
                let selected = select_test(&mir, selected_id);
                let artifact = self.compile_mir(
                    &selected,
                    &output_directory.join(format!("test-{index}")),
                    native_root.as_deref(),
                )?;
                Ok(CompiledTest {
                    name: test.name.clone(),
                    modes: test.modes.clone(),
                    execution: TestExecution::Executable(artifact),
                    expectations: test.expectations.clone(),
                })
            })
            .collect()
    }

    fn compiler_test_results(
        &self,
        graph: &severian_modules::ModuleGraph,
    ) -> Result<Vec<Option<String>>, CompileError> {
        let root_package = graph
            .modules
            .last()
            .map(|module| module.package)
            .expect("a resolved module graph contains its root");
        let mut results = Vec::new();
        for module in graph
            .modules
            .iter()
            .filter(|module| module.package == root_package)
        {
            let declarations = module
                .ast
                .items
                .iter()
                .filter(|item| !matches!(item, severian_ast::Item::Test(_)))
                .cloned()
                .collect::<Vec<_>>();
            for test in module.ast.items.iter().filter_map(|item| match item {
                severian_ast::Item::Test(test)
                    if test.modes.iter().any(|mode| mode == "compiler") =>
                {
                    Some(test)
                }
                _ => None,
            }) {
                let mut failure = None;
                for case in &test.compiler_cases {
                    let mut ast = severian_ast::Module {
                        items: declarations.clone(),
                    };
                    ast.items.extend(case.items.clone());
                    let mut body = test.body.clone();
                    body.extend(case.body.clone());
                    ast.items
                        .push(severian_ast::Item::Test(severian_ast::TestDeclaration {
                            name: Some("compiler case".into()),
                            parameters: Vec::new(),
                            cases: Vec::new(),
                            matrix: false,
                            modes: Vec::new(),
                            contracts: Vec::new(),
                            body,
                            compiler_cases: Vec::new(),
                            span: case.span,
                        }));
                    let mut case_graph = graph.clone();
                    let case_module = case_graph
                        .modules
                        .iter_mut()
                        .find(|candidate| candidate.id == module.id)
                        .expect("compiler test module remains in its resolved graph");
                    case_module.ast = ast;
                    let result = self.check_graph_to_mir(case_graph, CompileMode::Test);
                    let matched = match case.expectation {
                        severian_ast::CompilerExpectation::Accept => result.is_ok(),
                        severian_ast::CompilerExpectation::Reject => result.is_err(),
                    };
                    if !matched {
                        failure = Some(match case.expectation {
                            severian_ast::CompilerExpectation::Accept => format!(
                                "expected compiler acceptance, got: {}",
                                result.expect_err("failed acceptance has an error")
                            ),
                            severian_ast::CompilerExpectation::Reject => {
                                "expected compiler rejection, but the fragment was accepted".into()
                            }
                        });
                        break;
                    }
                }
                results.push(failure);
            }
        }
        Ok(results)
    }

    fn check_file_to_hir(
        &self,
        source: &Path,
        mode: CompileMode,
    ) -> Result<
        (
            severian_hir::Program,
            Vec<SourceFile>,
            severian_universal::TypeContext,
        ),
        CompileError,
    > {
        let graph = self.resolve_modules(source)?;
        self.check_graph_to_hir(graph, mode)
    }

    fn check_graph_to_hir(
        &self,
        mut graph: severian_modules::ModuleGraph,
        mode: CompileMode,
    ) -> Result<
        (
            severian_hir::Program,
            Vec<SourceFile>,
            severian_universal::TypeContext,
        ),
        CompileError,
    > {
        let root_package = graph
            .modules
            .last()
            .map(|module| module.package)
            .expect("a resolved module graph contains its root");
        let mut sources = Vec::new();
        let mut external = Vec::new();
        for module in &mut graph.modules {
            let source = module.source.clone();
            module.ast = with_core_prelude(&module.ast, &self.context.types)?;
            external.push(
                severian_xxi::resolve(
                    &module.ast,
                    &self.context.types,
                    &severian_abi::AbiTarget::derive(&self.target),
                )
                .map_err(|error| {
                    CompileError::Diagnostic(Diagnostic::new("E000701", error.to_string(), None))
                })?,
            );
            sources.push(source);
        }
        let mut typed = severian_semantic::analyze_package_with_context(
            &graph,
            &self.context,
            severian_semantic::PackageAnalysisContext {
                test_package: (mode == CompileMode::Test).then_some(root_package),
            },
        )
        .map_err(|diagnostic| {
            CompileError::Diagnostic(diagnostic.with_sources(sources.iter().cloned()))
        })?;
        for ((module, source_module), resolved) in typed
            .hir
            .modules
            .iter_mut()
            .zip(&graph.modules)
            .zip(&external)
        {
            apply_external_calls_to_module(
                &source_module.ast,
                resolved,
                module,
                Some((&typed.index, source_module.id)),
            )?;
        }
        severian_ownership::validate(&typed.hir).map_err(|diagnostic| {
            CompileError::Diagnostic(diagnostic.with_sources(sources.iter().cloned()))
        })?;
        Ok((typed.hir, sources, typed.types))
    }

    fn check_file_to_mir(
        &self,
        source: &Path,
        mode: CompileMode,
    ) -> Result<MirModule, CompileError> {
        let graph = self.resolve_modules(source)?;
        self.check_graph_to_mir(graph, mode)
    }

    fn check_graph_to_mir(
        &self,
        graph: severian_modules::ModuleGraph,
        mode: CompileMode,
    ) -> Result<MirModule, CompileError> {
        let root_package = graph
            .modules
            .last()
            .map(|module| module.package.0)
            .expect("a resolved module graph contains its root");
        let (hir, sources, types) = self.check_graph_to_hir(graph, mode)?;
        let mut mir = self.check_hir_to_mir(hir, sources, types)?;
        retain_package_exports_and_dependencies(&mut mir, u128::from(root_package));
        Ok(mir)
    }

    fn check_hir_to_mir(
        &self,
        hir: severian_hir::Program,
        sources: Vec<SourceFile>,
        types: severian_universal::TypeContext,
    ) -> Result<MirModule, CompileError> {
        let mut merged = severian_mir::build(&hir).map_err(CompileError::MirVerify)?;
        merged.types = Some(types);
        let mut context = self.context.clone();
        context.types = merged
            .types
            .clone()
            .expect("the source pipeline installed its structural type catalog");
        severian_mir::run_required_pipeline(&mut merged, &context)
            .map_err(CompileError::MirPass)?;
        if let Some(source) = sources.last() {
            attach_assertion_locations(&mut merged, source);
        }
        if !self.coverage {
            remove_module_coverage(&mut merged);
        }
        Ok(merged)
    }

    fn compile_mir(
        &self,
        mir: &MirModule,
        output: &Path,
        source: Option<&Path>,
    ) -> Result<Artifact, CompileError> {
        let linker_arguments = source
            .map(|source| self.native_linker_arguments(source, output))
            .transpose()?
            .unwrap_or_default();
        let types = self.types_for(mir);
        let plan = severian_compile::plan(mir, types).map_err(CompileError::Compile)?;
        if linker_arguments.is_empty() && !plan.has_custom_regions() {
            let resumed = plan.resumed_mir();
            let lir = severian_lowering::lower(&resumed, types, &self.target)
                .map_err(CompileError::Lowering)?;
            if severian_backend::supports_direct_lir(&lir) {
                return severian_backend::emit_executable(&lir, output)
                    .map_err(CompileError::Backend);
            }
        }
        let program = self.compile_plan(&plan)?;
        let mut linker_arguments = linker_arguments;
        if !program.tensor_jit_source.is_empty() {
            stage_tensor_jit_provider(output)?;
            if program.tensor_jit_requires_gpu {
                stage_triton_bridge(output)?;
            }
            let source = output.with_extension("tensor-jit.c");
            std::fs::write(&source, &program.tensor_jit_source).map_err(|error| {
                CompileError::NativeLink(format!(
                    "could not write Tensor-JIT launcher {}: {error}",
                    source.display()
                ))
            })?;
            linker_arguments.push(source.to_string_lossy().into_owned());
        }
        if program.host_mlir.contains("__sev_tokenizer_")
            || linker_arguments.iter().any(|argument| {
                Path::new(argument)
                    .file_name()
                    .is_some_and(|name| name == "tokenizer.c")
            })
        {
            stage_tokenizer_provider(output)?;
        }
        severian_backend::emit_mlir_executable_with_linker_arguments(
            &program.host_mlir,
            &self.target.triple,
            output,
            &linker_arguments,
        )
        .map_err(CompileError::Backend)
    }

    fn native_linker_arguments(
        &self,
        source: &Path,
        output: &Path,
    ) -> Result<Vec<String>, CompileError> {
        let Some(providers) = NativeProviderSources::discover(source, self.packages.as_ref())?
        else {
            return Ok(Vec::new());
        };
        let mut arguments = providers
            .c
            .iter()
            .map(|source| source.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        arguments.extend(
            providers
                .include
                .iter()
                .map(|path| format!("-I{}", path.display())),
        );
        arguments.extend(
            providers
                .libraries
                .iter()
                .map(|library| format!("-l{library}")),
        );

        for (index, source) in providers.rust.iter().enumerate() {
            let archive = output.with_extension(format!("ffi-rust-{index}.a"));
            let rustc = std::env::var("SEVERIAN_RUSTC").unwrap_or_else(|_| "rustc".into());
            let result = Command::new(&rustc)
                .arg(source)
                .args(["--crate-type", "staticlib"])
                .arg("-o")
                .arg(&archive)
                .output()
                .map_err(|error| {
                    CompileError::NativeLink(format!(
                        "could not start Rust FFI compiler `{rustc}`: {error}"
                    ))
                })?;
            if !result.status.success() {
                return Err(CompileError::NativeLink(format!(
                    "Rust FFI compilation failed for {}:\n{}",
                    source.display(),
                    String::from_utf8_lossy(&result.stderr).trim()
                )));
            }
            arguments.extend([
                "-Wl,--whole-archive".into(),
                "-Xlinker".into(),
                archive.to_string_lossy().into_owned(),
                "-Wl,--no-whole-archive".into(),
            ]);
        }

        if !providers.python.is_empty() {
            if providers.python.len() != 1 {
                return Err(CompileError::NativeLink(
                    "Python FFI currently requires exactly one source module per package".into(),
                ));
            }
            let bridge = output.with_extension("ffi-python.c");
            let bridge_source = self.python_bridge(source, &providers.python[0])?;
            std::fs::write(&bridge, bridge_source).map_err(|error| {
                CompileError::NativeLink(format!(
                    "could not write Python FFI bridge {}: {error}",
                    bridge.display()
                ))
            })?;
            arguments.push(bridge.to_string_lossy().into_owned());
            let python_config =
                std::env::var("SEVERIAN_PYTHON_CONFIG").unwrap_or_else(|_| "python3-config".into());
            for option in [["--embed", "--cflags"], ["--embed", "--ldflags"]] {
                let result = Command::new(&python_config)
                    .args(option)
                    .output()
                    .map_err(|error| {
                        CompileError::NativeLink(format!(
                            "could not start Python FFI configuration tool `{python_config}`: {error}"
                        ))
                    })?;
                if !result.status.success() {
                    return Err(CompileError::NativeLink(format!(
                        "Python FFI configuration failed:\n{}",
                        String::from_utf8_lossy(&result.stderr).trim()
                    )));
                }
                arguments.extend(
                    String::from_utf8_lossy(&result.stdout)
                        .split_whitespace()
                        .map(str::to_owned),
                );
            }
        }
        Ok(arguments)
    }

    fn python_bridge(&self, source: &Path, python: &Path) -> Result<String, CompileError> {
        let source_file = SourceFile::load(source).map_err(|error| {
            CompileError::NativeLink(format!("could not read {}: {error}", source.display()))
        })?;
        let tokens = severian_lexer::scan(&source_file).map_err(CompileError::Diagnostic)?;
        let ast = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
        let external = severian_xxi::resolve(
            &ast,
            &self.context.types,
            &severian_abi::AbiTarget::derive(&self.target),
        )
        .map_err(|error| CompileError::NativeLink(error.to_string()))?;
        render_python_bridge(python, &external.plans).map_err(CompileError::NativeLink)
    }

    fn resolve_modules(
        &self,
        source: &Path,
    ) -> Result<severian_modules::ModuleGraph, CompileError> {
        let packages = self.standard_package_graph(source)?;
        let initial = severian_modules::resolve_with_packages_and_max_errors(
            source,
            &packages,
            self.max_errors,
        )
        .map_err(CompileError::Diagnostic)?;
        let imports_file = initial.modules.iter().any(|module| {
            module.ast.items.iter().any(|item| {
                let severian_ast::Item::Import(import) = item else {
                    return false;
                };
                import.source.as_deref() == Some("file")
                    || (import.source.is_none()
                        && matches!(&import.subject, severian_ast::ImportSubject::Name(name) if name == "file"))
            })
        });
        if !imports_file {
            return Ok(initial);
        }
        let root = packages
            .packages
            .get(&packages.root)
            .expect("a package graph contains its root");
        let registry_roots = if ["data", "file", "csv", "json", "yaml"]
            .iter()
            .all(|name| root.dependencies.contains_key(*name))
        {
            ["csv", "json", "yaml"]
                .iter()
                .filter_map(|name| root.dependencies.get(*name))
                .map(|id| {
                    let package = &packages.packages[id];
                    (package.library.clone(), package.id)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        severian_modules::resolve_with_packages_and_additional_roots(
            source,
            &packages,
            &registry_roots,
            self.max_errors,
        )
        .map_err(CompileError::Diagnostic)
    }

    fn standard_package_graph(
        &self,
        source: &Path,
    ) -> Result<severian_modules::PackageGraph, CompileError> {
        let mut packages = self.packages.clone().unwrap_or_else(|| {
            let root = severian_modules::PackageId(0);
            severian_modules::PackageGraph {
                root,
                packages: BTreeMap::from([(
                    root,
                    severian_modules::ResolvedPackage {
                        id: root,
                        root: source.parent().unwrap_or_else(|| Path::new(".")).to_owned(),
                        library: source.to_owned(),
                        dependencies: BTreeMap::new(),
                    },
                )]),
            }
        });
        let library = crate::runtime_paths::library_root();
        let standard = [
            ("abi", library.join("interop/abi")),
            ("cli", library.join("system/cli")),
            ("csv", library.join("data/csv")),
            ("data", library.join("data")),
            ("device", library.join("system/device")),
            ("driver", library.join("system/driver")),
            ("environment", library.join("system/environment")),
            ("ffi", library.join("interop/ffi")),
            ("file", library.join("system/file")),
            ("io", library.join("system/io")),
            ("json", library.join("data/json")),
            ("math", library.join("core/math")),
            ("model", library.join("model")),
            ("os", library.join("system/os")),
            ("parallel", library.join("compute/parallel")),
            ("path", library.join("system/path")),
            ("platform", library.join("system/platform")),
            ("process", library.join("system/process")),
            ("tensor", library.join("compute/tensor")),
            ("yaml", library.join("data/yaml")),
        ];
        let mut next = packages
            .packages
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let catalog = crate::config::Catalog::load().map_err(CompileError::Component)?;
        let mut standard_ids = BTreeMap::new();
        for (name, root) in standard {
            let manifest_path = root.join("package.toml");
            let manifest =
                crate::config::Manifest::load(&manifest_path, &catalog).map_err(|error| {
                    CompileError::Diagnostic(Diagnostic::new(
                        "C001001",
                        format!("compiler standard package `{name}` could not resolve: {error}"),
                        None,
                    ))
                })?;
            let declared = manifest.module_graph(false);
            let id = merge_package_graph(&mut packages, declared, &mut next);
            standard_ids.insert(name.to_owned(), id);
        }
        for package in packages.packages.values_mut() {
            for (name, id) in &standard_ids {
                if package.id != *id {
                    package.dependencies.entry(name.clone()).or_insert(*id);
                }
            }
        }
        Ok(packages)
    }

    fn routes(&self, plan: &CompilePlan) -> BTreeSet<String> {
        fn block(
            block: &severian_compile::PlannedBlock,
            output: &mut BTreeSet<Option<CompilerId>>,
        ) {
            for segment in &block.segments {
                match segment {
                    severian_compile::PlanSegment::Standard(_) => {
                        output.insert(None);
                    }
                    severian_compile::PlanSegment::Compiler(region) => {
                        output.insert(Some(region.compiler));
                    }
                }
            }
        }
        let mut routes = BTreeSet::new();
        block(&plan.initializer, &mut routes);
        for function in &plan.functions {
            if let Some(body) = &function.body {
                block(body, &mut routes);
            }
        }
        routes
            .into_iter()
            .map(|route| match route {
                None => "standard".into(),
                Some(compiler) => self
                    .context
                    .types
                    .definitions()
                    .find(|definition| definition.declaration == compiler.declaration())
                    .map(|definition| definition.path.clone())
                    .unwrap_or_else(|| format!("compiler:{compiler}")),
            })
            .collect()
    }
}

fn compose_region_artifacts(
    ordinary: &str,
    artifacts: Vec<VerifiedCompiledRegionArtifact>,
    target: &TargetSpec,
) -> Result<RoutedProgram, CompileError> {
    let mut cpu = Vec::new();
    let gpu_kernels = Vec::new();
    let mut tensor_jit = Vec::new();
    for artifact in artifacts {
        match artifact {
            VerifiedCompiledRegionArtifact::CpuMlir(artifact) => cpu.push(artifact),
            VerifiedCompiledRegionArtifact::GpuKernel(artifact) => {
                let id = artifact.id;
                let generated =
                    severian_tensor_compiler::lower_gpu_bundle_to_mlir(&artifact.bundle)
                        .map_err(CompileError::Compile)?;
                let verified = severian_mlir::verify_artifact(id, generated, target)
                    .map_err(CompileError::Mlir)?;
                cpu.push(verified);
            }
            VerifiedCompiledRegionArtifact::TensorJit(artifact) => tensor_jit.push(artifact),
        }
    }
    let host_mlir = severian_mlir::compose(ordinary, &cpu, target).map_err(CompileError::Mlir)?;
    let tensor_jit_requires_gpu = tensor_jit.iter().any(|artifact| {
        artifact.bundle.placement == Some(severian_universal::ExecutionPlacement::Gpu)
    });
    let tensor_jit_source = render_tensor_jit_launchers(&tensor_jit)?;
    Ok(RoutedProgram {
        host_mlir,
        gpu_kernels,
        tensor_jit_source,
        tensor_jit_requires_gpu,
    })
}

fn render_tensor_jit_launchers(
    launchers: &[severian_compile::VerifiedTensorJitBundle],
) -> Result<String, CompileError> {
    if launchers.is_empty() {
        return Ok(String::new());
    }
    let header = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("driver crate is nested below the repository root")
        .join("runtime/native/tensor_jit.h");
    let mut source = format!(
        "#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n#include \"{}\"\n\ntypedef struct {{ int64_t rank; void *descriptor; }} sev_unranked_memref;\nstatic sev_jit_storage_view_abi __sev_jit_memref_view(int64_t rank, void *descriptor, uint32_t kind, uint32_t bits, uint32_t float_format) {{\n  int64_t *fields = (int64_t *)descriptor;\n  sev_jit_storage_view_abi view;\n  memset(&view, 0, sizeof(view));\n  view.magic = SEV_STORAGE_VIEW_ABI_MAGIC; view.abi_version = SEV_STORAGE_VIEW_ABI_VERSION; view.byte_size = sizeof(view);\n  view.owner = ((void **)descriptor)[0]; view.data = ((const uint8_t **)descriptor)[1]; view.rank = (uint64_t)rank; view.offset = fields[2];\n  view.dimensions = fields + 3; view.strides = fields + 3 + rank;\n  view.element.abi_version = 1; view.element.byte_size = sizeof(view.element); view.element.kind = kind; view.element.bits = bits; view.element.float_format = float_format;\n  uint64_t elements = 1; for (int64_t axis = 0; axis < rank; ++axis) elements *= (uint64_t)view.dimensions[axis];\n  view.byte_length = elements * ((bits + 7) / 8);\n  return view;\n}}\nstatic sev_unranked_memref __sev_jit_unranked_result(sev_jit_storage_view_abi *view) {{\n  size_t words = (size_t)(3 + 2 * view->rank);\n  int64_t *descriptor = (int64_t *)malloc(words * sizeof(int64_t));\n  if (descriptor == NULL) abort();\n  ((void **)descriptor)[0] = view->owner != NULL ? view->owner : (void *)view->data;\n  ((void **)descriptor)[1] = (void *)view->data; descriptor[2] = view->offset;\n  memcpy(descriptor + 3, view->dimensions, view->rank * sizeof(int64_t));\n  memcpy(descriptor + 3 + view->rank, view->strides, view->rank * sizeof(int64_t));\n  sev_unranked_memref result = {{(int64_t)view->rank, descriptor}}; return result;\n}}\n",
        header.display()
    );
    source = source.replace(
        "sev_unranked_memref result = {(int64_t)view->rank, descriptor}; return result;",
        "sev_unranked_memref result = {(int64_t)view->rank, descriptor}; free(view); return result;",
    );
    let ranked_abis = launchers
        .iter()
        .flat_map(|launcher| {
            launcher
                .bundle
                .inputs
                .iter()
                .chain(&launcher.bundle.outputs)
        })
        .filter_map(tensor_jit_rank)
        .collect::<BTreeSet<_>>();
    for rank in ranked_abis {
        source.push_str(&format!(
            "typedef struct {{ void *allocated; void *aligned; int64_t offset; int64_t sizes[{rank}]; int64_t strides[{rank}]; }} sev_ranked_memref_{rank};\nstatic sev_ranked_memref_{rank} __sev_jit_ranked_result_{rank}(sev_jit_storage_view_abi *view) {{\n  sev_ranked_memref_{rank} result; result.allocated = view->owner != NULL ? view->owner : (void *)view->data; result.aligned = (void *)view->data; result.offset = view->offset;\n  memcpy(result.sizes, view->dimensions, sizeof(result.sizes)); memcpy(result.strides, view->strides, sizeof(result.strides)); free(view); return result;\n}}\n"
        ));
    }
    for launcher in launchers {
        let bundle = &launcher.bundle;
        if bundle.outputs.is_empty() {
            return Err(CompileError::NativeLink(format!(
                "Tensor-JIT artifact {} has no outputs",
                launcher.id.index()
            )));
        }
        let target = match bundle.placement {
            Some(severian_universal::ExecutionPlacement::Gpu)
                if bundle.architecture.starts_with("gfx") =>
            {
                1u32
            }
            Some(severian_universal::ExecutionPlacement::Gpu) => 2u32,
            _ => 0u32,
        };
        let program = severian_runtime::tensor_jit::TensorJitProgram {
            target: match target {
                1 => severian_runtime::tensor_jit::TensorJitTarget::Amd,
                2 => severian_runtime::tensor_jit::TensorJitTarget::Nvidia,
                _ => severian_runtime::tensor_jit::TensorJitTarget::Cpu,
            },
            architecture: bundle.architecture.clone(),
            nodes: bundle.graph.nodes().to_vec(),
            inputs: bundle.input_nodes.clone(),
            outputs: bundle.output_nodes.clone(),
        }
        .encode()
        .map_err(|error| CompileError::NativeLink(error.to_string()))?;
        source.push_str(&render_tensor_jit_launcher(launcher, &program, target)?);
    }
    Ok(source)
}

fn stage_tensor_jit_provider(output: &Path) -> Result<(), CompileError> {
    #[cfg(target_os = "linux")]
    const PROVIDER_NAME: &str = "libseverian_tensor_jit_provider.so";
    #[cfg(target_os = "macos")]
    const PROVIDER_NAME: &str = "libseverian_tensor_jit_provider.dylib";
    #[cfg(target_os = "windows")]
    const PROVIDER_NAME: &str = "severian_tensor_jit_provider.dll";

    let executable = std::env::current_exe().map_err(|error| {
        CompileError::NativeLink(format!(
            "could not locate the Severian driver while staging the Tensor-JIT provider: {error}"
        ))
    })?;
    let executable_directory = executable.parent().ok_or_else(|| {
        CompileError::NativeLink(format!(
            "Severian driver path {} has no parent directory",
            executable.display()
        ))
    })?;
    let provider = [
        executable_directory.join(PROVIDER_NAME),
        executable_directory
            .parent()
            .unwrap_or(executable_directory)
            .join(PROVIDER_NAME),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
    .ok_or_else(|| {
        CompileError::NativeLink(format!(
            "Tensor-JIT provider {PROVIDER_NAME} was not built beside the Severian driver; build the severian-tensor-jit-provider workspace target"
        ))
    })?;
    let destination_directory = output.parent().unwrap_or_else(|| Path::new("."));
    let destination = destination_directory.join(PROVIDER_NAME);
    if provider != destination {
        std::fs::copy(&provider, &destination).map_err(|error| {
            CompileError::NativeLink(format!(
                "could not stage Tensor-JIT provider {} at {}: {error}",
                provider.display(),
                destination.display()
            ))
        })?;
    }
    Ok(())
}

fn stage_triton_bridge(output: &Path) -> Result<(), CompileError> {
    #[cfg(target_os = "linux")]
    const BRIDGE_NAME: &str = "libseverian_triton_bridge.so";
    #[cfg(target_os = "macos")]
    const BRIDGE_NAME: &str = "libseverian_triton_bridge.dylib";
    #[cfg(target_os = "windows")]
    const BRIDGE_NAME: &str = "severian_triton_bridge.dll";

    let driver = std::env::current_exe().map_err(|error| {
        CompileError::NativeLink(format!(
            "could not locate the Severian driver while staging the Triton bridge: {error}"
        ))
    })?;
    let driver_directory = driver.parent().ok_or_else(|| {
        CompileError::NativeLink("Severian driver has no parent directory".into())
    })?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("driver crate is nested below the repository root");
    let configured = std::env::var_os("SEVERIAN_TRITON_BRIDGE_LIBRARY").map(PathBuf::from);
    let candidates = configured.into_iter().chain([
        driver_directory.join(BRIDGE_NAME),
        driver_directory
            .parent()
            .unwrap_or(driver_directory)
            .join(BRIDGE_NAME),
        repository
            .join("target/severian-triton-native-v5")
            .join(BRIDGE_NAME),
        repository
            .join("target/severian-triton-native")
            .join(BRIDGE_NAME),
    ]);
    let source = candidates.into_iter().find(|candidate| candidate.is_file()).ok_or_else(|| {
        CompileError::NativeLink(format!(
            "GPU Tensor-JIT requires {BRIDGE_NAME}; build compiler/boundaries/triton/native/build-native.sh once or set SEVERIAN_TRITON_BRIDGE_LIBRARY"
        ))
    })?;
    let destination = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(BRIDGE_NAME);
    if source != destination {
        std::fs::copy(&source, &destination).map_err(|error| {
            CompileError::NativeLink(format!(
                "could not stage Triton bridge {} at {}: {error}",
                source.display(),
                destination.display()
            ))
        })?;
    }
    Ok(())
}

fn stage_tokenizer_provider(output: &Path) -> Result<(), CompileError> {
    #[cfg(target_os = "linux")]
    const PROVIDER_NAME: &str = "libseverian_tokenizer_provider.so";
    #[cfg(target_os = "macos")]
    const PROVIDER_NAME: &str = "libseverian_tokenizer_provider.dylib";
    #[cfg(target_os = "windows")]
    const PROVIDER_NAME: &str = "severian_tokenizer_provider.dll";

    let driver = std::env::current_exe().map_err(|error| {
        CompileError::NativeLink(format!(
            "could not locate the Severian driver while staging the tokenizer provider: {error}"
        ))
    })?;
    let executable_directory = driver.parent().ok_or_else(|| {
        CompileError::NativeLink("Severian driver has no parent directory".into())
    })?;
    let candidates = [
        executable_directory.join(PROVIDER_NAME),
        executable_directory
            .parent()
            .map(|parent| parent.join(PROVIDER_NAME))
            .unwrap_or_default(),
    ];
    let source = candidates.iter().find(|candidate| candidate.is_file()).ok_or_else(|| {
        CompileError::NativeLink(format!(
            "tokenizer provider {PROVIDER_NAME} (ABI {}) was not built beside the Severian driver; build the severian-tokenizer-provider workspace target",
            severian_tokenizer_provider::ABI_VERSION,
        ))
    })?;
    let destination_directory = output.parent().unwrap_or_else(|| Path::new("."));
    let destination = destination_directory.join(PROVIDER_NAME);
    if source != &destination {
        std::fs::copy(source, &destination).map_err(|error| {
            CompileError::NativeLink(format!(
                "could not stage tokenizer provider {} at {}: {error}",
                source.display(),
                destination.display()
            ))
        })?;
    }
    Ok(())
}

fn render_tensor_jit_launcher(
    launcher: &severian_compile::VerifiedTensorJitBundle,
    program: &[u8],
    target: u32,
) -> Result<String, CompileError> {
    let id = launcher.id.index();
    let bundle = &launcher.bundle;
    let parameters = bundle
        .inputs
        .iter()
        .enumerate()
        .map(|(index, ty)| c_tensor_jit_parameter(index, ty))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let result_types = bundle
        .outputs
        .iter()
        .map(c_tensor_jit_type)
        .collect::<Result<Vec<_>, _>>()?;
    let result = if result_types.len() == 1 {
        result_types[0].clone()
    } else {
        format!("sev_jit_results_{id}")
    };
    let bytes = program
        .iter()
        .map(|byte| byte.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let graph_hash = stable_tensor_jit_hash(program);
    let mut compiler_identity = bundle.architecture.as_bytes().to_vec();
    compiler_identity.extend_from_slice(b"|severian-tensor-jit-v1|triton-");
    compiler_identity.extend_from_slice(severian_triton::DONOR_REVISION.as_bytes());
    compiler_identity.extend_from_slice(b"|provider-");
    compiler_identity.extend_from_slice(
        severian_tensor_jit_provider::PROVIDER_ABI_VERSION
            .to_string()
            .as_bytes(),
    );
    let compiler_hash = stable_tensor_jit_hash(&compiler_identity);
    let result_declaration = if result_types.len() == 1 {
        String::new()
    } else {
        format!(
            "typedef struct {{ {} }} sev_jit_results_{id};\n",
            result_types
                .iter()
                .enumerate()
                .map(|(index, ty)| format!("{ty} field{index};"))
                .collect::<Vec<_>>()
                .join(" ")
        )
    };
    let mut body = format!(
        "\n{result_declaration}static const uint8_t __sev_jit_program_{id}[] = {{{bytes}}};\nstatic const sev_tensor_jit_region_abi __sev_jit_region_{id} = {{\n  SEV_TENSOR_JIT_ABI_MAGIC, SEV_TENSOR_JIT_ABI_VERSION, sizeof(sev_tensor_jit_region_abi),\n  {{{},{},{},{}}}, {{{},{},{},{}}},\n  __sev_jit_program_{id}, sizeof(__sev_jit_program_{id}), {target}, {}, {}, 0\n}};\n{result} __sev_artifact_{id}({parameters}) {{\n  sev_tensor_jit_value_abi inputs[{}] = {{0}};\n  sev_tensor_jit_value_abi outputs[{}] = {{0}};\n",
        graph_hash[0], graph_hash[1], graph_hash[2], graph_hash[3],
        compiler_hash[0], compiler_hash[1], compiler_hash[2], compiler_hash[3],
        bundle.inputs.len(), bundle.outputs.len(), bundle.inputs.len(), bundle.outputs.len()
    );
    for (index, (ty, node)) in bundle.inputs.iter().zip(&bundle.input_nodes).enumerate() {
        let graph_node = bundle.graph.node(*node);
        let runtime_list = bundle.graph.nodes().iter().any(|consumer| {
            consumer
                .inputs
                .iter()
                .zip(&consumer.operand_roles)
                .any(|(input, role)| input == node && *role != severian_fusion::OperandRole::Data)
        });
        body.push_str(&render_tensor_jit_input(
            index,
            ty,
            graph_node.shape.element_kind,
            runtime_list,
        )?);
    }
    for (index, (ty, node)) in bundle.outputs.iter().zip(&bundle.output_nodes).enumerate() {
        let graph_node = bundle.graph.node(*node);
        let kind = if graph_node.kind == severian_fusion::NodeKind::StorageView
            && matches!(graph_node.operation.as_str(), "shape" | "strides")
        {
            "SEV_TENSOR_JIT_VALUE_LIST_I64"
        } else {
            tensor_jit_value_kind(ty, graph_node.shape.element_kind)?
        };
        body.push_str(&format!(
            "  outputs[{index}].abi_version = SEV_TENSOR_JIT_ABI_VERSION;\n  outputs[{index}].byte_size = sizeof(sev_tensor_jit_value_abi);\n  outputs[{index}].kind = {};\n",
            kind
        ));
    }
    body.push_str(&format!(
        "  int32_t status = __sev_tensor_jit_launch_v1(&__sev_jit_region_{id}, inputs, {}, outputs, {});\n  if (status != SEV_TENSOR_JIT_OK) abort();\n",
        bundle.inputs.len(), bundle.outputs.len()
    ));
    if bundle.outputs.len() == 1 {
        body.push_str(&format!(
            "  return {};\n",
            tensor_jit_result_expression(&bundle.outputs[0], 0)?
        ));
    } else {
        body.push_str(&format!("  sev_jit_results_{id} result;\n"));
        for (index, ty) in bundle.outputs.iter().enumerate() {
            body.push_str(&format!(
                "  result.field{index} = {};\n",
                tensor_jit_result_expression(ty, index)?
            ));
        }
        body.push_str("  return result;\n");
    }
    body.push_str("}\n");
    Ok(body)
}

fn c_tensor_jit_type(ty: &severian_mlir::LoweredType) -> Result<String, CompileError> {
    use severian_mlir::{LoweredFloatFormat, LoweredTensorShape, LoweredType};
    match ty {
        LoweredType::Tensor {
            shape: LoweredTensorShape::Unranked,
            ..
        } => Ok("sev_unranked_memref".into()),
        LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(dimensions),
            ..
        } => Ok(format!("sev_ranked_memref_{}", dimensions.len())),
        LoweredType::Bytes | LoweredType::String => Ok("void *".into()),
        LoweredType::Integer {
            bits: 1..=8,
            signed: true,
        } => Ok("int8_t".into()),
        LoweredType::Integer {
            bits: 1..=8,
            signed: false,
        }
        | LoweredType::Boolean => Ok("uint8_t".into()),
        LoweredType::Integer {
            bits: 9..=16,
            signed: true,
        } => Ok("int16_t".into()),
        LoweredType::Integer {
            bits: 9..=16,
            signed: false,
        } => Ok("uint16_t".into()),
        LoweredType::Integer {
            bits: 17..=32,
            signed: true,
        } => Ok("int32_t".into()),
        LoweredType::Integer {
            bits: 17..=32,
            signed: false,
        } => Ok("uint32_t".into()),
        LoweredType::Integer {
            bits: 33..=64,
            signed: true,
        } => Ok("int64_t".into()),
        LoweredType::Integer {
            bits: 33..=64,
            signed: false,
        } => Ok("uint64_t".into()),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(32),
        } => Ok("float".into()),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(64),
        } => Ok("double".into()),
        unsupported => Err(CompileError::NativeLink(format!(
            "Tensor-JIT host ABI does not support {unsupported:?}"
        ))),
    }
}

fn c_tensor_jit_parameter(
    index: usize,
    ty: &severian_mlir::LoweredType,
) -> Result<String, CompileError> {
    if matches!(
        ty,
        severian_mlir::LoweredType::Tensor {
            shape: severian_mlir::LoweredTensorShape::Unranked,
            ..
        }
    ) {
        Ok(format!(
            "int64_t arg{index}_rank, void *arg{index}_descriptor"
        ))
    } else if let severian_mlir::LoweredType::Tensor {
        shape: severian_mlir::LoweredTensorShape::Ranked(dimensions),
        ..
    } = ty
    {
        let mut fields = vec![
            format!("void *arg{index}_allocated"),
            format!("void *arg{index}_aligned"),
            format!("int64_t arg{index}_offset"),
        ];
        fields.extend((0..dimensions.len()).map(|axis| format!("int64_t arg{index}_size{axis}")));
        fields.extend((0..dimensions.len()).map(|axis| format!("int64_t arg{index}_stride{axis}")));
        Ok(fields.join(", "))
    } else {
        c_tensor_jit_type(ty).map(|ty| format!("{ty} arg{index}"))
    }
}

fn tensor_jit_value_kind(
    ty: &severian_mlir::LoweredType,
    element: severian_fusion::ElementKind,
) -> Result<&'static str, CompileError> {
    use severian_mlir::LoweredType;
    Ok(match ty {
        LoweredType::Tensor { .. } => "SEV_TENSOR_JIT_VALUE_STORAGE",
        LoweredType::Bytes | LoweredType::String
            if element != severian_fusion::ElementKind::Opaque =>
        {
            "SEV_TENSOR_JIT_VALUE_STORAGE"
        }
        LoweredType::Bytes | LoweredType::String => "SEV_TENSOR_JIT_VALUE_POINTER",
        LoweredType::Integer { signed: true, .. } => "SEV_TENSOR_JIT_VALUE_SIGNED",
        LoweredType::Integer { signed: false, .. } | LoweredType::Boolean => {
            "SEV_TENSOR_JIT_VALUE_UNSIGNED"
        }
        LoweredType::Float { .. } => "SEV_TENSOR_JIT_VALUE_FLOAT",
        unsupported => {
            return Err(CompileError::NativeLink(format!(
                "unsupported Tensor-JIT value {unsupported:?}"
            )))
        }
    })
}

fn render_tensor_jit_input(
    index: usize,
    ty: &severian_mlir::LoweredType,
    element: severian_fusion::ElementKind,
    runtime_list: bool,
) -> Result<String, CompileError> {
    use severian_mlir::LoweredType;
    let kind = if runtime_list {
        "SEV_TENSOR_JIT_VALUE_LIST_I64"
    } else {
        tensor_jit_value_kind(ty, element)?
    };
    if let LoweredType::Tensor {
        element: tensor_element,
        shape: severian_mlir::LoweredTensorShape::Unranked,
    } = ty
    {
        let (element_kind, bits, float_format) = tensor_element_abi(*tensor_element)?;
        return Ok(format!(
            "  sev_jit_storage_view_abi arg{index}_view = __sev_jit_memref_view(arg{index}_rank, arg{index}_descriptor, {element_kind}, {bits}, {float_format});\n  inputs[{index}].abi_version = SEV_TENSOR_JIT_ABI_VERSION;\n  inputs[{index}].byte_size = sizeof(sev_tensor_jit_value_abi);\n  inputs[{index}].kind = SEV_TENSOR_JIT_VALUE_STORAGE;\n  inputs[{index}].bits = {bits};\n  inputs[{index}].value.storage = &arg{index}_view;\n"
        ));
    }
    if let LoweredType::Tensor {
        element: tensor_element,
        shape: severian_mlir::LoweredTensorShape::Ranked(dimensions),
    } = ty
    {
        let (element_kind, bits, float_format) = tensor_element_abi(*tensor_element)?;
        let rank = dimensions.len();
        let sizes = (0..rank)
            .map(|axis| format!("arg{index}_size{axis}"))
            .collect::<Vec<_>>()
            .join(",");
        let strides = (0..rank)
            .map(|axis| format!("arg{index}_stride{axis}"))
            .collect::<Vec<_>>()
            .join(",");
        return Ok(format!(
            "  int64_t arg{index}_dimensions[{rank}] = {{{sizes}}};\n  int64_t arg{index}_strides[{rank}] = {{{strides}}};\n  sev_jit_storage_view_abi arg{index}_view; memset(&arg{index}_view, 0, sizeof(arg{index}_view));\n  arg{index}_view.magic = SEV_STORAGE_VIEW_ABI_MAGIC; arg{index}_view.abi_version = SEV_STORAGE_VIEW_ABI_VERSION; arg{index}_view.byte_size = sizeof(arg{index}_view);\n  arg{index}_view.owner = arg{index}_allocated; arg{index}_view.data = (const uint8_t *)arg{index}_aligned; arg{index}_view.rank = {rank}; arg{index}_view.offset = arg{index}_offset; arg{index}_view.dimensions = arg{index}_dimensions; arg{index}_view.strides = arg{index}_strides;\n  arg{index}_view.byte_length = ({bits} + 7) / 8; for (uint64_t axis = 0; axis < {rank}; ++axis) arg{index}_view.byte_length *= (uint64_t)arg{index}_dimensions[axis];\n  arg{index}_view.element.abi_version = 1; arg{index}_view.element.byte_size = sizeof(arg{index}_view.element); arg{index}_view.element.kind = {element_kind}; arg{index}_view.element.bits = {bits}; arg{index}_view.element.float_format = {float_format};\n  inputs[{index}].abi_version = SEV_TENSOR_JIT_ABI_VERSION; inputs[{index}].byte_size = sizeof(sev_tensor_jit_value_abi); inputs[{index}].kind = SEV_TENSOR_JIT_VALUE_STORAGE; inputs[{index}].bits = {bits}; inputs[{index}].value.storage = &arg{index}_view;\n"
        ));
    }
    let field = match ty {
        LoweredType::Bytes | LoweredType::String
            if element != severian_fusion::ElementKind::Opaque =>
        {
            "storage"
        }
        LoweredType::Bytes | LoweredType::String => "pointer",
        LoweredType::Integer { signed: true, .. } => "signed_integer",
        LoweredType::Integer { signed: false, .. } | LoweredType::Boolean => "unsigned_integer",
        LoweredType::Float { .. } => "floating",
        _ => unreachable!("validated by tensor_jit_value_kind"),
    };
    let cast = if field == "storage" {
        "(sev_jit_storage_view_abi *)"
    } else {
        ""
    };
    Ok(format!(
        "  inputs[{index}].abi_version = SEV_TENSOR_JIT_ABI_VERSION;\n  inputs[{index}].byte_size = sizeof(sev_tensor_jit_value_abi);\n  inputs[{index}].kind = {kind};\n  inputs[{index}].bits = {};\n  inputs[{index}].value.{field} = {cast}arg{index};\n",
        tensor_jit_bits(ty)
    ))
}

fn tensor_jit_bits(ty: &severian_mlir::LoweredType) -> u16 {
    match ty {
        severian_mlir::LoweredType::Integer { bits, .. } => *bits,
        severian_mlir::LoweredType::Float {
            format: severian_mlir::LoweredFloatFormat::Ieee(bits),
        } => *bits,
        severian_mlir::LoweredType::Boolean => 1,
        severian_mlir::LoweredType::Tensor { element, .. } => tensor_element_abi(*element)
            .map(|(_, bits, _)| bits as u16)
            .unwrap_or(0),
        _ => 0,
    }
}

fn tensor_jit_result_expression(
    ty: &severian_mlir::LoweredType,
    index: usize,
) -> Result<String, CompileError> {
    use severian_mlir::LoweredType;
    Ok(match ty {
        LoweredType::Tensor {
            shape: severian_mlir::LoweredTensorShape::Unranked,
            ..
        } => format!("__sev_jit_unranked_result(outputs[{index}].value.storage)"),
        LoweredType::Tensor {
            shape: severian_mlir::LoweredTensorShape::Ranked(dimensions),
            ..
        } => {
            let rank = dimensions.len();
            format!("__sev_jit_ranked_result_{rank}(outputs[{index}].value.storage)")
        }
        LoweredType::Bytes | LoweredType::String => format!("outputs[{index}].value.pointer"),
        LoweredType::Integer { signed: true, .. } => {
            format!("outputs[{index}].value.signed_integer")
        }
        LoweredType::Integer { signed: false, .. } | LoweredType::Boolean => {
            format!("outputs[{index}].value.unsigned_integer")
        }
        LoweredType::Float { .. } => format!("outputs[{index}].value.floating"),
        unsupported => {
            return Err(CompileError::NativeLink(format!(
                "unsupported Tensor-JIT result {unsupported:?}"
            )))
        }
    })
}

fn tensor_jit_rank(ty: &severian_mlir::LoweredType) -> Option<usize> {
    match ty {
        severian_mlir::LoweredType::Tensor {
            shape: severian_mlir::LoweredTensorShape::Ranked(dimensions),
            ..
        } => Some(dimensions.len()),
        _ => None,
    }
}

fn tensor_element_abi(
    element: severian_mlir::LoweredTensorElement,
) -> Result<(u32, u32, u32), CompileError> {
    use severian_mlir::{LoweredFloatFormat, LoweredTensorElement};
    Ok(match element {
        LoweredTensorElement::Integer { bits, signed: true } => (1, u32::from(bits), 0),
        LoweredTensorElement::Integer {
            bits,
            signed: false,
        } => (2, u32::from(bits), 0),
        LoweredTensorElement::Boolean => (2, 1, 0),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Ieee(bits),
        } => (3, u32::from(bits), 1),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::BrainFloat16,
        } => (3, 16, 2),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E4M3Fn,
        } => (3, 8, 3),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E5M2,
        } => (3, 8, 4),
    })
}

fn stable_tensor_jit_hash(bytes: &[u8]) -> [u64; 4] {
    let mut hash = [
        1469598103934665603u64,
        7809847782465536322,
        9659303129496669493,
        2870177450012600261,
    ];
    let primes = [
        1099511628211u64,
        14029467366897019727,
        1609587929392839161,
        9650029242287828579,
    ];
    for byte in bytes {
        for lane in 0..4 {
            hash[lane] ^= u64::from(*byte) + lane as u64 * 0x9d;
            hash[lane] = hash[lane].wrapping_mul(primes[lane]);
        }
    }
    hash
}

fn attach_assertion_locations(module: &mut MirModule, source: &SourceFile) {
    attach_block_assertion_locations(&mut module.initializer, source);
    for function in &mut module.functions {
        if let Some(body) = &mut function.body {
            attach_block_assertion_locations(body, source);
        }
    }
}

fn attach_block_assertion_locations(body: &mut severian_mir::CfgBody, source: &SourceFile) {
    for block in &mut body.blocks {
        let mut index = 0;
        while index < block.statements.len() {
            if matches!(&block.statements[index], CfgStatement::Coverage(point) if point.source != source.id)
            {
                block.statements.remove(index);
                block.statement_spans.remove(index);
                continue;
            }
            let statement = &mut block.statements[index];
            match statement {
                CfgStatement::Coverage(point) => {
                    let before = source
                        .text
                        .get(..usize::try_from(point.span_start).unwrap_or(0))
                        .unwrap_or("");
                    let line =
                        u32::try_from(before.bytes().filter(|byte| *byte == b'\n').count() + 1)
                            .unwrap_or(u32::MAX);
                    let kind = match point.kind {
                        severian_mir::CoverageKind::Line => "line",
                        severian_mir::CoverageKind::Branch => "branch",
                    };
                    let file = source.path.display().to_string();
                    point.key = Some(format!(
                        "{file}|{kind}|{line}|{}|{}",
                        point.span_start, point.ordinal
                    ));
                    point.file = Some(file);
                    point.line = Some(line);
                }
                CfgStatement::Assert { origin, .. } => {
                    origin.location = assertion_location(origin, source);
                }
                _ => {}
            }
            index += 1;
        }
    }
}

fn remove_coverage(body: &mut severian_mir::CfgBody) {
    for block in &mut body.blocks {
        let statements = std::mem::take(&mut block.statements);
        let spans = std::mem::take(&mut block.statement_spans);
        for (statement, span) in statements.into_iter().zip(spans) {
            if !matches!(statement, CfgStatement::Coverage(_)) {
                block.statements.push(statement);
                block.statement_spans.push(span);
            }
        }
    }
}

fn remove_module_coverage(module: &mut MirModule) {
    remove_coverage(&mut module.initializer);
    for body in module
        .functions
        .iter_mut()
        .filter_map(|function| function.body.as_mut())
    {
        remove_coverage(body);
    }
}

fn collect_coverage_points(
    body: &severian_mir::CfgBody,
    output: &mut BTreeSet<severian_mir::CoveragePoint>,
) {
    for block in &body.blocks {
        for statement in &block.statements {
            if let CfgStatement::Coverage(point) = statement {
                output.insert(point.clone());
            }
        }
    }
}

fn assertion_location(
    origin: &severian_mir::AssertionOrigin,
    source: &SourceFile,
) -> Option<severian_mir::AssertionLocation> {
    let statement_start = usize::try_from(origin.statement_start).ok()?;
    let condition_start = usize::try_from(origin.condition_start).ok()?;
    let condition_end = usize::try_from(origin.condition_end).ok()?;
    let before = source.text.get(..statement_start)?;
    let line = u32::try_from(before.bytes().filter(|byte| *byte == b'\n').count() + 1).ok()?;
    let line_start = before.rfind('\n').map_or(0, |offset| offset + 1);
    let column = u32::try_from(
        source
            .text
            .get(line_start..statement_start)?
            .chars()
            .count()
            + 1,
    )
    .ok()?;
    let expression = source
        .text
        .get(condition_start..condition_end)?
        .trim()
        .to_owned();
    Some(severian_mir::AssertionLocation {
        file: source.path.display().to_string(),
        line,
        column,
        expression,
    })
}

#[derive(Debug, Default)]
struct NativeProviderSources {
    c: Vec<PathBuf>,
    rust: Vec<PathBuf>,
    python: Vec<PathBuf>,
    include: Vec<PathBuf>,
    libraries: Vec<String>,
}

impl NativeProviderSources {
    fn discover(
        source: &Path,
        packages: Option<&severian_modules::PackageGraph>,
    ) -> Result<Option<Self>, CompileError> {
        let Some(root) = source.parent().and_then(|directory| {
            directory
                .ancestors()
                .find(|ancestor| ancestor.join("package.toml").is_file())
        }) else {
            return Ok(None);
        };
        let mut roots = BTreeSet::from([root.to_owned()]);
        if let Some(packages) = packages {
            roots.extend(
                packages
                    .packages
                    .values()
                    .map(|package| package.root.clone()),
            );
        }
        let mut providers = Self::default();
        for root in roots {
            providers.discover_manifest(&root)?;
        }
        providers.c.sort();
        providers.c.dedup();
        providers.rust.sort();
        providers.rust.dedup();
        providers.python.sort();
        providers.python.dedup();
        providers.include.sort();
        providers.include.dedup();
        providers.libraries.sort();
        providers.libraries.dedup();
        if providers.c.is_empty() && providers.rust.is_empty() && providers.python.is_empty() {
            Ok(None)
        } else {
            Ok(Some(providers))
        }
    }

    fn discover_manifest(&mut self, root: &Path) -> Result<(), CompileError> {
        let manifest_path = root.join("package.toml");
        if !manifest_path.is_file() {
            return Ok(());
        }
        let manifest = std::fs::read_to_string(&manifest_path).map_err(|error| {
            CompileError::NativeLink(format!(
                "could not read FFI manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let document = manifest.parse::<toml::Value>().map_err(|error| {
            CompileError::NativeLink(format!(
                "invalid FFI manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        for (language, output) in [
            ("c", &mut self.c),
            ("rust", &mut self.rust),
            ("python", &mut self.python),
        ] {
            let Some(sources) = document
                .get("xxi")
                .and_then(|xxi| xxi.get(language))
                .and_then(|provider| provider.get("sources"))
                .and_then(toml::Value::as_array)
            else {
                continue;
            };
            for declared in sources {
                let declared = declared.as_str().ok_or_else(|| {
                    CompileError::NativeLink(format!(
                        "[xxi.{language}].sources entries must be paths"
                    ))
                })?;
                let path = root.join(declared);
                if !path.is_file() {
                    return Err(CompileError::NativeLink(format!(
                        "FFI source {} does not exist",
                        path.display()
                    )));
                }
                output.push(path);
            }
        }

        if let Some(entries) = document.get("ffi").and_then(toml::Value::as_array) {
            for entry in entries {
                let Some(table) = entry.as_table() else {
                    return Err(CompileError::NativeLink(format!(
                        "[[ffi]] entries in {} must be tables",
                        manifest_path.display()
                    )));
                };
                let abi = table
                    .get("abi")
                    .and_then(toml::Value::as_str)
                    .unwrap_or("c-v1");
                if abi != "c-v1" {
                    continue;
                }
                append_paths(root, &manifest_path, table.get("sources"), &mut self.c)?;
                append_paths(
                    root,
                    &manifest_path,
                    table.get("include"),
                    &mut self.include,
                )?;
                if let Some(libraries) = table.get("libraries") {
                    for library in libraries.as_array().ok_or_else(|| {
                        CompileError::NativeLink(format!(
                            "ffi libraries in {} must be an array",
                            manifest_path.display()
                        ))
                    })? {
                        self.libraries.push(
                            library
                                .as_str()
                                .ok_or_else(|| {
                                    CompileError::NativeLink(format!(
                                        "ffi libraries in {} must be strings",
                                        manifest_path.display()
                                    ))
                                })?
                                .to_owned(),
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

fn append_paths(
    root: &Path,
    manifest: &Path,
    values: Option<&toml::Value>,
    output: &mut Vec<PathBuf>,
) -> Result<(), CompileError> {
    let Some(values) = values else {
        return Ok(());
    };
    for value in values.as_array().ok_or_else(|| {
        CompileError::NativeLink(format!(
            "ffi paths in {} must be an array",
            manifest.display()
        ))
    })? {
        let relative = value.as_str().ok_or_else(|| {
            CompileError::NativeLink(format!(
                "ffi paths in {} must be strings",
                manifest.display()
            ))
        })?;
        let path = root.join(relative);
        if !path.exists() {
            return Err(CompileError::NativeLink(format!(
                "FFI path {} does not exist",
                path.display()
            )));
        }
        output.push(path);
    }
    Ok(())
}

fn render_python_bridge(
    source: &Path,
    plans: &[severian_ffi::BoundaryPlan],
) -> Result<String, String> {
    use severian_abi::{AbiType, ScalarType};
    use severian_ffi::Conversion;

    let directory = source
        .parent()
        .ok_or_else(|| format!("Python FFI source {} has no parent", source.display()))?;
    let module = source
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Python FFI source {} has no module name", source.display()))?;
    let mut output = format!(
        "#include <Python.h>\n#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n\nstatic PyObject *sev_python_module;\n\nstatic void sev_python_fail(void) {{\n    PyErr_Print();\n    abort();\n}}\n\nstatic void sev_python_initialize(void) {{\n    if (sev_python_module != NULL) return;\n    if (!Py_IsInitialized()) Py_Initialize();\n    PyObject *path = PyUnicode_FromString(\"{}\");\n    if (path == NULL || PyList_Insert(PySys_GetObject(\"path\"), 0, path) != 0) sev_python_fail();\n    Py_DECREF(path);\n    sev_python_module = PyImport_ImportModule(\"{}\");\n    if (sev_python_module == NULL) sev_python_fail();\n}}\n",
        c_string_contents(&directory.to_string_lossy()),
        c_string_contents(module),
    );
    for plan in plans {
        let symbol = plan.symbol.name.as_str();
        if !c_identifier(symbol) {
            return Err(format!(
                "Python FFI symbol `{symbol}` cannot be represented as a native C symbol"
            ));
        }
        let result_type = python_c_type(&plan.result_type, &plan.result_conversion)?;
        let parameters = plan
            .parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| {
                python_c_type(&parameter.abi_type, &parameter.conversion)
                    .map(|ty| format!("{ty} argument_{index}"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        output.push_str(&format!(
            "\n{result_type} {symbol}({}) {{\n    sev_python_initialize();\n    PyObject *callable = PyObject_GetAttrString(sev_python_module, \"{}\");\n    if (callable == NULL) sev_python_fail();\n    PyObject *arguments = PyTuple_New({});\n    if (arguments == NULL) sev_python_fail();\n",
            if parameters.is_empty() { "void" } else { &parameters },
            c_string_contents(symbol),
            plan.parameters.len(),
        ));
        for (index, parameter) in plan.parameters.iter().enumerate() {
            let conversion = match (&parameter.conversion, &parameter.abi_type) {
                (Conversion::Utf8View, _) => {
                    format!("PyUnicode_FromString(argument_{index})")
                }
                (_, AbiType::Scalar(ScalarType::Integer { signed: true, .. })) => {
                    format!("PyLong_FromLongLong((long long)argument_{index})")
                }
                (_, AbiType::Scalar(ScalarType::Integer { signed: false, .. })) => {
                    format!("PyLong_FromUnsignedLongLong((unsigned long long)argument_{index})")
                }
                (_, AbiType::Scalar(ScalarType::Float { .. })) => {
                    format!("PyFloat_FromDouble((double)argument_{index})")
                }
                (_, AbiType::Scalar(ScalarType::Boolean)) => {
                    format!("PyBool_FromLong(argument_{index} != 0)")
                }
                _ => {
                    return Err(format!(
                        "Python FFI parameter `{}` uses an unsupported ABI conversion",
                        parameter.name
                    ))
                }
            };
            output.push_str(&format!(
                "    PyObject *python_{index} = {conversion};\n    if (python_{index} == NULL) sev_python_fail();\n    PyTuple_SET_ITEM(arguments, {index}, python_{index});\n"
            ));
        }
        output.push_str(
            "    PyObject *result = PyObject_CallObject(callable, arguments);\n    Py_DECREF(arguments);\n    Py_DECREF(callable);\n    if (result == NULL) sev_python_fail();\n",
        );
        match (&plan.result_conversion, &plan.result_type) {
            (_, AbiType::Void) => output.push_str("    Py_DECREF(result);\n    return;\n"),
            (Conversion::Utf8View, _) => output.push_str(
                "    const char *text = PyUnicode_AsUTF8(result);\n    if (text == NULL) sev_python_fail();\n    size_t length = strlen(text);\n    char *copy = malloc(length + 1);\n    if (copy == NULL) abort();\n    memcpy(copy, text, length + 1);\n    Py_DECREF(result);\n    return copy;\n",
            ),
            (_, AbiType::Scalar(ScalarType::Integer { signed: true, .. })) => output.push_str(
                "    long long value = PyLong_AsLongLong(result);\n    if (value == -1 && PyErr_Occurred()) sev_python_fail();\n    Py_DECREF(result);\n    return value;\n",
            ),
            (_, AbiType::Scalar(ScalarType::Integer { signed: false, .. })) => output.push_str(
                "    unsigned long long value = PyLong_AsUnsignedLongLong(result);\n    if (value == (unsigned long long)-1 && PyErr_Occurred()) sev_python_fail();\n    Py_DECREF(result);\n    return value;\n",
            ),
            (_, AbiType::Scalar(ScalarType::Float { .. })) => output.push_str(
                "    double value = PyFloat_AsDouble(result);\n    if (value == -1.0 && PyErr_Occurred()) sev_python_fail();\n    Py_DECREF(result);\n    return value;\n",
            ),
            (_, AbiType::Scalar(ScalarType::Boolean)) => output.push_str(
                "    int value = PyObject_IsTrue(result);\n    if (value < 0) sev_python_fail();\n    Py_DECREF(result);\n    return value != 0;\n",
            ),
            _ => {
                return Err(format!(
                    "Python FFI symbol `{symbol}` uses an unsupported result conversion"
                ))
            }
        }
        output.push_str("}\n");
    }
    Ok(output)
}

fn python_c_type(
    ty: &severian_abi::AbiType,
    conversion: &severian_ffi::Conversion,
) -> Result<&'static str, String> {
    use severian_abi::{AbiFloatFormat, AbiType, ScalarType};
    if matches!(conversion, severian_ffi::Conversion::Utf8View) {
        return Ok("const char *");
    }
    match ty {
        AbiType::Void => Ok("void"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 8,
            signed: true,
        }) => Ok("int8_t"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 16,
            signed: true,
        }) => Ok("int16_t"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 32,
            signed: true,
        }) => Ok("int32_t"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 64,
            signed: true,
        }) => Ok("int64_t"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 8,
            signed: false,
        }) => Ok("uint8_t"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 16,
            signed: false,
        }) => Ok("uint16_t"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 32,
            signed: false,
        }) => Ok("uint32_t"),
        AbiType::Scalar(ScalarType::Integer {
            bits: 64,
            signed: false,
        }) => Ok("uint64_t"),
        AbiType::Scalar(ScalarType::Float {
            format: AbiFloatFormat::Ieee(32),
        }) => Ok("float"),
        AbiType::Scalar(ScalarType::Float {
            format: AbiFloatFormat::Ieee(64),
        }) => Ok("double"),
        AbiType::Scalar(ScalarType::Boolean) => Ok("_Bool"),
        _ => Err("Python FFI currently supports scalar and string ABI values".into()),
    }
}

fn c_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte == b'_' || byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

fn c_string_contents(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| match character {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            character => vec![character],
        })
        .collect()
}

fn module_name(path: &Path) -> String {
    let package_root = path
        .ancestors()
        .skip(1)
        .find(|directory| directory.join("package.toml").is_file())
        .or_else(|| path.parent())
        .unwrap_or_else(|| Path::new(""));
    let package = package_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("package");
    let relative = path.strip_prefix(package_root).unwrap_or(path);
    let relative = relative.with_extension("");
    format!("{package}_{}", relative.display())
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn merge_package_graph(
    destination: &mut severian_modules::PackageGraph,
    source: severian_modules::PackageGraph,
    next_id: &mut u32,
) -> severian_modules::PackageId {
    let mut translated = BTreeMap::new();
    for (source_id, source_package) in &source.packages {
        let canonical_root = std::fs::canonicalize(&source_package.root)
            .unwrap_or_else(|_| source_package.root.clone());
        let destination_id = destination
            .packages
            .values()
            .find(|package| {
                std::fs::canonicalize(&package.root).unwrap_or_else(|_| package.root.clone())
                    == canonical_root
            })
            .map(|package| package.id)
            .unwrap_or_else(|| {
                let id = severian_modules::PackageId(*next_id);
                *next_id = next_id.saturating_add(1);
                destination.packages.insert(
                    id,
                    severian_modules::ResolvedPackage {
                        id,
                        root: source_package.root.clone(),
                        library: source_package.library.clone(),
                        dependencies: BTreeMap::new(),
                    },
                );
                id
            });
        translated.insert(*source_id, destination_id);
    }

    for (source_id, source_package) in source.packages {
        let destination_id = translated[&source_id];
        let dependencies = source_package
            .dependencies
            .into_iter()
            .map(|(name, dependency)| (name, translated[&dependency]));
        destination
            .packages
            .get_mut(&destination_id)
            .expect("translated package is present in the destination graph")
            .dependencies
            .extend(dependencies);
    }

    translated[&source.root]
}

fn select_test(module: &MirModule, selected: severian_mir::FunctionId) -> MirModule {
    let mut module = module.clone();
    module.entry = Some(selected);
    let test_functions = module
        .tests
        .iter()
        .map(|test| test.function)
        .collect::<BTreeSet<_>>();
    module
        .functions
        .retain(|function| !test_functions.contains(&function.id) || function.id == selected);
    module.tests.clear();
    retain_reachable_functions(&mut module, [selected]);
    module
}

fn retain_package_exports_and_dependencies(module: &mut MirModule, root_package: u128) {
    let roots = module
        .functions
        .iter()
        .filter(|function| function.definition.package == root_package)
        .map(|function| function.id)
        .chain(module.entry)
        .chain(module.tests.iter().map(|test| test.function))
        .collect::<BTreeSet<_>>();
    retain_reachable_functions(module, roots);
}

fn retain_reachable_functions(
    module: &mut MirModule,
    roots: impl IntoIterator<Item = severian_mir::FunctionId>,
) {
    let functions_by_definition = module.functions.iter().fold(
        BTreeMap::<severian_universal::DefId, Vec<severian_mir::FunctionId>>::new(),
        |mut functions, function| {
            functions
                .entry(function.definition)
                .or_default()
                .push(function.id);
            functions
        },
    );
    let bodies = module
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref().map(|body| (function.id, body)))
        .collect::<BTreeMap<_, _>>();
    let mut reachable = roots.into_iter().collect::<BTreeSet<_>>();
    let mut queue = std::collections::VecDeque::from_iter(reachable.iter().copied());

    let mut initial_instances = BTreeSet::new();
    let mut initial_definitions = BTreeSet::new();
    collect_function_references(
        &module.initializer,
        &mut initial_instances,
        &mut initial_definitions,
    );
    enqueue_function_references(
        initial_instances,
        initial_definitions,
        &functions_by_definition,
        &mut reachable,
        &mut queue,
    );

    while let Some(function) = queue.pop_front() {
        let Some(body) = bodies.get(&function) else {
            continue;
        };
        let mut instances = BTreeSet::new();
        let mut definitions = BTreeSet::new();
        collect_function_references(body, &mut instances, &mut definitions);
        enqueue_function_references(
            instances,
            definitions,
            &functions_by_definition,
            &mut reachable,
            &mut queue,
        );
    }

    module
        .functions
        .retain(|function| reachable.contains(&function.id));
}

fn enqueue_function_references(
    instances: BTreeSet<severian_mir::FunctionId>,
    definitions: BTreeSet<severian_universal::DefId>,
    functions_by_definition: &BTreeMap<severian_universal::DefId, Vec<severian_mir::FunctionId>>,
    reachable: &mut BTreeSet<severian_mir::FunctionId>,
    queue: &mut std::collections::VecDeque<severian_mir::FunctionId>,
) {
    for function in instances.into_iter().chain(
        definitions
            .into_iter()
            .flat_map(|definition| functions_by_definition.get(&definition))
            .flatten()
            .copied(),
    ) {
        if reachable.insert(function) {
            queue.push_back(function);
        }
    }
}

fn collect_function_references(
    body: &severian_mir::CfgBody,
    instances: &mut BTreeSet<severian_mir::FunctionId>,
    definitions: &mut BTreeSet<severian_universal::DefId>,
) {
    for block in &body.blocks {
        for statement in &block.statements {
            match statement {
                severian_mir::CfgStatement::Assign(_, value) => {
                    collect_rvalue_functions(value, definitions)
                }
                severian_mir::CfgStatement::Assert {
                    condition, message, ..
                } => {
                    collect_operand_function(condition, definitions);
                    if let Some(message) = message {
                        collect_operand_function(message, definitions);
                    }
                }
                severian_mir::CfgStatement::Operation { operands, .. } => {
                    for operand in operands {
                        collect_operand_function(operand, definitions);
                    }
                }
                severian_mir::CfgStatement::Drop(_)
                | severian_mir::CfgStatement::StorageLive(_)
                | severian_mir::CfgStatement::StorageDead(_)
                | severian_mir::CfgStatement::Coverage(_) => {}
            }
        }
        match &block.terminator {
            severian_mir::Terminator::Call {
                callee, arguments, ..
            }
            | severian_mir::Terminator::Spawn {
                callee, arguments, ..
            } => {
                collect_callee_functions(callee, instances, definitions);
                for argument in arguments {
                    collect_operand_function(argument, definitions);
                }
            }
            severian_mir::Terminator::Goto(_, operands) => {
                for operand in operands {
                    collect_operand_function(operand, definitions);
                }
            }
            severian_mir::Terminator::Branch { condition, .. } => {
                collect_operand_function(condition, definitions)
            }
            severian_mir::Terminator::Switch { discriminant, .. } => {
                collect_operand_function(discriminant, definitions)
            }
            severian_mir::Terminator::SpawnFieldUpdate { value, .. }
            | severian_mir::Terminator::Throw(value) => {
                collect_operand_function(value, definitions)
            }
            severian_mir::Terminator::Return(value) => {
                if let Some(value) = value {
                    collect_operand_function(value, definitions);
                }
            }
            severian_mir::Terminator::Unreachable => {}
        }
    }
}

fn collect_callee_functions(
    callee: &severian_mir::Callee,
    instances: &mut BTreeSet<severian_mir::FunctionId>,
    definitions: &mut BTreeSet<severian_universal::DefId>,
) {
    match callee {
        severian_mir::Callee::Direct {
            instance, function, ..
        } => {
            if let Some(instance) = instance {
                instances.insert(*instance);
            } else {
                definitions.insert(*function);
            }
        }
        severian_mir::Callee::Method {
            implementation,
            receiver,
            ..
        } => {
            definitions.insert(*implementation);
            collect_operand_function(receiver, definitions);
        }
        severian_mir::Callee::FunctionValue(operand) => {
            collect_operand_function(operand, definitions)
        }
        severian_mir::Callee::Constructor { .. } | severian_mir::Callee::Intrinsic(_) => {}
    }
}

fn collect_rvalue_functions(
    value: &severian_mir::Rvalue,
    definitions: &mut BTreeSet<severian_universal::DefId>,
) {
    match value {
        severian_mir::Rvalue::Use(operand)
        | severian_mir::Rvalue::Unary { operand, .. }
        | severian_mir::Rvalue::Convert { operand, .. }
        | severian_mir::Rvalue::Await { task: operand } => {
            collect_operand_function(operand, definitions)
        }
        severian_mir::Rvalue::Binary { left, right, .. } => {
            collect_operand_function(left, definitions);
            collect_operand_function(right, definitions);
        }
        severian_mir::Rvalue::Aggregate { fields, .. } | severian_mir::Rvalue::Variant { fields, .. } => {
            for field in fields {
                collect_operand_function(field, definitions);
            }
        }
        severian_mir::Rvalue::BorrowShared(_)
        | severian_mir::Rvalue::BorrowExclusive(_)
        | severian_mir::Rvalue::AddressOf(_) => {}
    }
}

fn collect_operand_function(
    operand: &severian_mir::Operand,
    definitions: &mut BTreeSet<severian_universal::DefId>,
) {
    if let severian_mir::Operand::Function(definition) = operand {
        definitions.insert(*definition);
    }
}

fn with_core_prelude(
    ast: &severian_ast::Module,
    types: &severian_universal::TypeContext,
) -> Result<severian_ast::Module, CompileError> {
    if types.resolve_name("string").is_none() {
        return Ok(ast.clone());
    }
    // Package import resolution will eventually load this dependency from the
    // prelude's `import print from io`. Until then, bootstrap the same source
    // module explicitly instead of duplicating its foreign declaration in core.
    let mut io = SourceFile::virtual_source(
        "system/io/src/lib.sev",
        include_str!("../../../../../library/system/io/src/lib.sev"),
    );
    io.id = SourceId(u32::MAX);
    let tokens = severian_lexer::scan(&io).map_err(CompileError::Diagnostic)?;
    let mut module = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
    module.items.retain(|item| match item {
        severian_ast::Item::Function(function) if !function.decorators.is_empty() => {
            !ast.items.iter().any(|item| {
                matches!(item, severian_ast::Item::Function(existing)
                    if existing.name == function.name && !existing.decorators.is_empty())
            }) && function
                .parameters
                .iter()
                .all(|parameter| boundary_type_is_available(&parameter.annotation, types))
                && boundary_type_is_available(&function.result, types)
        }
        severian_ast::Item::Function(function) => {
            function.name == "print"
                && !ast.items.iter().any(|item| {
                    matches!(item, severian_ast::Item::Function(existing)
                        if existing.name == function.name && existing.decorators.is_empty())
                })
                && types.resolve_name("usize").is_some()
                && types.resolve_name("bool").is_some()
                && (!function.type_parameters.is_empty()
                    || (function
                        .parameters
                        .iter()
                        .all(|parameter| boundary_type_is_available(&parameter.annotation, types))
                        && boundary_type_is_available(&function.result, types)))
        }
        _ => false,
    });

    let mut size = SourceFile::virtual_source(
        "core/size/src/lib.sev",
        include_str!("../../../../../library/core/size/src/lib.sev"),
    );
    size.id = SourceId(u32::MAX - 1);
    let tokens = severian_lexer::scan(&size).map_err(CompileError::Diagnostic)?;
    let mut size = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
    size.items.retain(|item| match item {
        severian_ast::Item::Function(function) if !function.decorators.is_empty() => {
            function
                .parameters
                .iter()
                .all(|parameter| boundary_type_is_available(&parameter.annotation, types))
                && boundary_type_is_available(&function.result, types)
        }
        _ => true,
    });
    module.items.extend(size.items);

    let mut text = SourceFile::virtual_source(
        "core/text/src/lib.sev",
        include_str!("../../../../../library/core/text/src/lib.sev"),
    );
    text.id = SourceId(u32::MAX - 2);
    let tokens = severian_lexer::scan(&text).map_err(CompileError::Diagnostic)?;
    let mut text = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
    text.items.retain(|item| match item {
        severian_ast::Item::Function(function) if !function.decorators.is_empty() => {
            function
                .parameters
                .iter()
                .all(|parameter| boundary_type_is_available(&parameter.annotation, types))
                && boundary_type_is_available(&function.result, types)
        }
        _ => true,
    });
    module.items.extend(text.items);

    let mut prelude = SourceFile::virtual_source(
        "core/prelude.sev",
        include_str!("../../../../../library/core/prelude.sev"),
    );
    prelude.id = SourceId(u32::MAX - 3);
    let tokens = severian_lexer::scan(&prelude).map_err(CompileError::Diagnostic)?;
    let prelude = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
    module.items.extend(prelude.items);
    module.items.extend(ast.items.iter().cloned());
    Ok(module)
}

fn boundary_type_is_available(
    annotation: &severian_ast::TypeAnnotation,
    types: &severian_universal::TypeContext,
) -> bool {
    let Some((name, arguments)) = annotation.named_parts() else {
        return false;
    };
    (name == "Any" || types.resolve_name(name).is_some())
        && arguments
            .iter()
            .all(|argument| boundary_type_is_available(argument, types))
}

fn apply_external_calls_to_module(
    ast: &severian_ast::Module,
    external: &severian_xxi::ResolvedExternalModule,
    module: &mut severian_hir::Module,
    identity: Option<(&severian_semantic::ProgramIndex, severian_modules::ModuleId)>,
) -> Result<(), CompileError> {
    for declaration in &external.declarations {
        let ast_function = ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Function(function) => Some(function),
                _ => None,
            })
            .find(|function| {
                function.span.start == declaration.span_start
                    && function.span.end == declaration.span_end
            })
            .ok_or_else(|| external_metadata_error("XXI declaration has no source item"))?;
        let overload_ordinal = ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Function(function)
                    if function.name == ast_function.name
                        && function.span.start < ast_function.span.start =>
                {
                    Some(())
                }
                _ => None,
            })
            .count();
        let hir_function = if let Some((index, module_id)) = identity {
            let definition = index
                .function_definition(module_id, &ast_function.name, overload_ordinal)
                .ok_or_else(|| external_metadata_error("foreign definition is not indexed"))?;
            module
                .functions
                .iter_mut()
                .find(|function| function.definition == definition)
        } else {
            module
                .functions
                .iter_mut()
                .filter(|function| function.name == ast_function.name)
                .nth(overload_ordinal)
        }
        .ok_or_else(|| {
            external_metadata_error(&format!(
                "foreign definition `{}` has no typed HIR item",
                ast_function.name
            ))
        })?;
        let declaration = &declaration.function;
        hir_function.call_type = severian_hir::CallType::External(severian_hir::ExternalCall {
            interface: severian_hir::InterfaceId("xxi".into()),
            symbol: severian_hir::SymbolId(declaration.symbol.name.as_str().into()),
            provider: declaration
                .provider
                .as_ref()
                .map(|provider| severian_hir::ProviderId(provider.clone())),
            ffi: severian_hir::FfiId("ffi".into()),
            abi: severian_hir::AbiId(format!("{:?}", declaration.abi)),
        });
    }
    Ok(())
}

fn external_metadata_error(message: &str) -> CompileError {
    CompileError::Diagnostic(Diagnostic::new("E000701", message, None))
}

pub fn compile_source(source: &SourceFile, output: &Path) -> Result<Artifact, CompileError> {
    Compiler::new(TargetSpec::host())?.compile_source(source, output)
}

pub fn compile_file(source: &Path, output: &Path) -> Result<Artifact, CompileError> {
    Compiler::new(TargetSpec::host())?.compile_file(source, output)
}

pub fn check_file(source: &Path) -> Result<(), CompileError> {
    Compiler::new(TargetSpec::host())?.check_file(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_universal::TypeId;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_PACKAGE: AtomicUsize = AtomicUsize::new(0);

    fn temporary_package() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "severian-driver-package-semantic-{}-{}",
            std::process::id(),
            NEXT_PACKAGE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn self_hosted_bool_reaches_verified_mlir_on_the_standard_pipeline() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap();
        let source = repository.join("sev_compiler/universal/primitive/bool.sev");
        let compiler = Compiler::new(TargetSpec::host()).unwrap();
        let (hir, _, types) = compiler
            .check_file_to_hir(&source, CompileMode::Build)
            .unwrap();
        let boolean = types.resolve_name("bool").unwrap();

        assert_eq!(
            types.definition(boolean).unwrap().path,
            "universal.primitive.bool"
        );
        assert!(hir
            .modules
            .iter()
            .flat_map(|module| &module.classes)
            .all(|class| class.id != boolean));

        let mlir = compiler.emit_file(&source, EmitStage::Mlir).unwrap();
        assert!(mlir.contains("i1"));
        assert!(mlir.contains("arith.xori"));
    }

    #[test]
    fn compiler_term_generic_examples_reach_verified_mlir() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .to_path_buf();
        std::thread::Builder::new()
            .name("compiler-term-generics-mlir".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let compiler = Compiler::new(TargetSpec::host()).unwrap();
                for ordinal in 6..=25 {
                    let prefix = format!("{ordinal:02}-");
                    let directory = repository.join("docs/examples/01-types/04-generics");
                    let source = std::fs::read_dir(&directory)
                        .unwrap()
                        .map(|entry| entry.unwrap().path())
                        .find(|path| {
                            path.file_name()
                                .and_then(|name| name.to_str())
                                .is_some_and(|name| name.starts_with(&prefix))
                        })
                        .unwrap_or_else(|| panic!("missing generic example {prefix}"));
                    let mir = compiler
                        .check_file_to_mir(&source, CompileMode::Test)
                        .unwrap_or_else(|error| panic!("{}: {error}", source.display()));
                    let test = mir
                        .tests
                        .first()
                        .unwrap_or_else(|| panic!("{} has no test", source.display()));
                    let mlir = compiler
                        .compile_mir_to_mlir(&select_test(&mir, test.function))
                        .unwrap_or_else(|error| panic!("{}: {error}", source.display()));
                    assert!(mlir.contains("module {"), "{}: {mlir}", source.display());
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn ranked_tensor_jit_inputs_include_upload_byte_length() {
        let ty = severian_mlir::LoweredType::Tensor {
            element: severian_mlir::LoweredTensorElement::Float {
                format: severian_mlir::LoweredFloatFormat::Ieee(32),
            },
            shape: severian_mlir::LoweredTensorShape::Ranked(vec![
                severian_mlir::LoweredTensorDimension::Dynamic,
                severian_mlir::LoweredTensorDimension::Known(4),
            ]),
        };
        let source =
            render_tensor_jit_input(3, &ty, severian_fusion::ElementKind::IeeeFloat, false)
                .unwrap();
        assert!(source.contains("arg3_view.owner = arg3_allocated"));
        assert!(source.contains("arg3_view.byte_length = (32 + 7) / 8"));
        assert!(source.contains("axis < 2"));
    }

    #[test]
    fn sev_ranked_elementwise_source_reaches_direct_gpu_mlir_without_triton() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap();
        let source =
            repository.join("library/compute/tensor/examples/severian-gpu-mlir/src/main.sev");
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.devices.push(severian_target::Device {
            name: "test-amd-gpu".into(),
            kind: severian_target::DeviceKind::Gpu,
            architecture: "gfx1100".into(),
            features: severian_target::FeatureSet::from_names(["vendor.amd", "driver.rocm"]),
        });
        for capability in ["mlir.dialect.gpu", "mlir.dialect.memref"] {
            target.capabilities.insert(capability);
        }

        let mlir = std::thread::Builder::new()
            .name("severian-gpu-mlir-slice".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let compiler = Compiler::new(target).unwrap();
                let mir = compiler
                    .check_file_to_mir(&source, CompileMode::Build)
                    .unwrap();
                compiler.compile_mir_to_mlir(&mir).unwrap()
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(
            mlir.contains("severian.gpu.architecture = \"gfx1100\""),
            "{mlir}"
        );
        assert!(mlir.contains("gpu.launch blocks"));
        assert_eq!(mlir.matches("gpu.launch blocks").count(), 1, "{mlir}");
        assert!(mlir.contains("arith.addf"));
        assert!(mlir.contains("arith.select"));
        assert!(mlir.contains("memref.store"));
        assert!(!mlir.contains("__sev_gpu_launch_"));
        assert!(!mlir.contains("triton"));
        assert!(!mlir.contains("tt."));
    }

    #[test]
    fn imported_generic_calls_retain_definition_identity_through_codegen() {
        let root = temporary_package();
        std::fs::write(
            root.join("dependency.sev"),
            "def choose[T](value: T) -> T:\n    return value\ndef choose(value: string) -> string:\n    return value\n",
        )
        .unwrap();
        std::fs::write(
            root.join("root.sev"),
            "import \"dependency.sev\" as dependency\ndef main():\n    value: i32 = dependency.choose(42)\n",
        )
        .unwrap();
        Compiler::new(TargetSpec::host())
            .unwrap()
            .compile_file(&root.join("root.sev"), &root.join("program"))
            .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn stored_async_tasks_survive_cfg_verification_and_codegen() {
        let root = temporary_package();
        let source = root.join("async.sev");
        std::fs::write(
            &source,
            "def work(value: int) -> int:\n    return value * 2\n\ndef main():\n    first = async work(10) with self\n    second = async work(21) with self\n    print(await first + await second)\n",
        )
        .unwrap();
        Compiler::new(TargetSpec::host())
            .unwrap()
            .compile_file(&source, &root.join("program"))
            .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn self_owned_unit_tasks_are_awaited_before_function_return() {
        let root = temporary_package();
        let source = root.join("structured.sev");
        std::fs::write(
            &source,
            "def work():\n    pass\n\ndef main():\n    async work() with self and lock\n    async work() with self and lock\n",
        )
        .unwrap();
        let mlir = Compiler::new(TargetSpec::host())
            .unwrap()
            .emit_file(&source, EmitStage::Mlir)
            .unwrap();
        assert_eq!(mlir.matches("async.execute").count(), 2);
        assert_eq!(mlir.matches("async.await %v").count(), 2);
        assert_eq!(mlir.matches("func.call @__sev_task_lock").count(), 2);
        assert_eq!(mlir.matches("func.call @__sev_task_unlock").count(), 2);
        let last_await = mlir.rfind("async.await %v").unwrap();
        let function_return = mlir[last_await..].find("return").unwrap() + last_await;
        assert!(last_await < function_return);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn locked_async_class_updates_mutate_the_captured_storage() {
        let root = temporary_package();
        let source = root.join("locked-update.sev");
        std::fs::write(
            &source,
            "class Counter:\n    value: int\n    def increment():\n        value += 1\ndef main():\n    counter := Counter(0)\n    task = async counter.increment() with self and lock\n    await task\n    assert(counter.value == 1)\n",
        )
        .unwrap();
        let mlir = Compiler::new(TargetSpec::host())
            .unwrap()
            .emit_file(&source, EmitStage::Mlir)
            .unwrap();
        assert!(mlir.contains("async.execute"));
        assert!(mlir.contains("func.call @__sev_task_lock"));
        assert!(mlir.contains("%update_old_"));
        assert!(mlir.contains("llvm.store %update_result_"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bare_sources_receive_the_compiler_standard_package_set() {
        let root = temporary_package();
        let source = root.join("root.sev");
        std::fs::write(&source, "def main():\n    print(\"ready\")\n").unwrap();
        let compiler = Compiler::new(TargetSpec::host()).unwrap();
        let graph = compiler.standard_package_graph(&source).unwrap();
        let dependencies = &graph.packages[&graph.root].dependencies;
        for package in [
            "data",
            "environment",
            "file",
            "io",
            "json",
            "math",
            "model",
            "os",
            "parallel",
            "path",
            "process",
            "tensor",
            "yaml",
        ] {
            assert!(
                dependencies.contains_key(package),
                "missing standard package {package}"
            );
        }
        let model = dependencies["model"];
        assert!(graph.packages[&model]
            .dependencies
            .contains_key("higgs_audio_v2"));
        assert!(!dependencies.contains_key("higgs_audio_v2"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selecting_one_test_removes_other_test_functions() {
        let function = |id: u32| severian_mir::Function {
            id: severian_mir::FunctionId(id.into()),
            definition: severian_universal::DefId {
                package: 0,
                module: 0,
                declaration: severian_universal::DeclarationId::from_path(&format!("test.{id}")),
            },
            substitution: severian_universal::Substitution::default(),
            name: format!("test-{id}"),
            parameters: Vec::new(),
            result: TypeId(0),
            body: Some(severian_mir::CfgBody::default()),
            call_type: severian_mir::CallType::Severian,
        };
        let module = MirModule {
            functions: vec![function(0), function(1)],
            tests: vec![
                severian_mir::TestDeclaration {
                    name: "first".into(),
                    modes: Vec::new(),
                    function: severian_mir::FunctionId(0),
                    expectations: Vec::new(),
                },
                severian_mir::TestDeclaration {
                    name: "second".into(),
                    modes: Vec::new(),
                    function: severian_mir::FunctionId(1),
                    expectations: Vec::new(),
                },
            ],
            ..MirModule::default()
        };
        let selected = select_test(&module, severian_mir::FunctionId(0));
        assert_eq!(selected.functions.len(), 1);
        assert_eq!(selected.functions[0].id, severian_mir::FunctionId(0));
        assert!(selected.tests.is_empty());
    }

    #[test]
    fn qwen_voice_golden_reaches_ranked_structural_mlir_without_typed_symbols() {
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("docs/examples/08-numerics/16-qwen-voice-golden.sev");
        let (mlir, load_mlir) = std::thread::Builder::new()
            .name("qwen-voice-mlir-golden".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let compiler = Compiler::new(TargetSpec::host()).unwrap();
                let mir = compiler
                    .check_file_to_mir(&source, CompileMode::Test)
                    .unwrap();
                let compile_test = |name: &str| {
                    let selected = mir.tests.iter().find(|test| test.name == name).unwrap();
                    compiler
                        .compile_mir_to_mlir(&select_test(&mir, selected.function))
                        .unwrap()
                };
                (
                    compile_test(
                        "ranked Qwen attention MLP decoder and OmniVoice head shapes execute",
                    ),
                    compile_test(
                        "dependency load[T] enters ranked StorageView MLIR without a typed symbol",
                    ),
                )
            })
            .unwrap()
            .join()
            .unwrap();

        assert!(mlir.contains("linalg.generic"));
        assert!(mlir.contains("\"reduction\""));
        assert!(mlir.contains("tensor<1x1x2x4xf32>"));
        assert!(mlir
            .lines()
            .filter(|line| line.contains("func.func @__sev_artifact_"))
            .all(|line| !line.contains("tensor<*x")));
        assert!(!mlir.contains("matmul_rank"));
        assert!(!mlir.contains("matmul_f32"));
        assert!(load_mlir
            .contains("func.func private @__sev_safetensor_view(i64, !llvm.ptr) -> !llvm.ptr"));
        assert!(!load_mlir.contains("load_bf16"));
        assert!(load_mlir
            .lines()
            .filter(|line| line.contains("func.func @__sev_artifact_"))
            .all(|line| !line.contains("tensor<*x")));
    }

    #[test]
    fn native_provider_sources_are_resolved_from_the_package_manifest() {
        let root = temporary_package();
        let source_directory = root.join("src");
        std::fs::create_dir_all(&source_directory).unwrap();
        let source = source_directory.join("main.sev");
        std::fs::write(&source, "def main():\n    pass\n").unwrap();
        std::fs::write(root.join("native.c"), "").unwrap();
        std::fs::write(root.join("native.rs"), "").unwrap();
        std::fs::write(root.join("native.py"), "").unwrap();
        std::fs::write(
            root.join("package.toml"),
            "[package]\nname = \"native-providers\"\nversion = \"0.1.0\"\n\n[xxi.c]\nsources = [\"native.c\"]\n\n[xxi.rust]\nsources = [\"native.rs\"]\n\n[xxi.python]\nsources = [\"native.py\"]\n",
        )
        .unwrap();

        let providers = NativeProviderSources::discover(&source, None)
            .unwrap()
            .expect("native provider declarations");
        assert_eq!(providers.c, [root.join("native.c")]);
        assert_eq!(providers.rust, [root.join("native.rs")]);
        assert_eq!(providers.python, [root.join("native.py")]);

        std::fs::remove_dir_all(root).unwrap();
    }
}
