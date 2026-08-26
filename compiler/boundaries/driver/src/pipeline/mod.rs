use severian_backend::{Artifact, BackendError};
use severian_compile::{CompileContext, CompileHandler, CompilePlan, CompilerRegistry};
use severian_diagnostics::Diagnostic;
use severian_mir::{CfgStatement, Module as MirModule};
use severian_source::SourceFile;
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
        let plan =
            severian_compile::plan(mir, &self.context.types).map_err(CompileError::Compile)?;
        self.compile_plan_to_mlir(&plan)
    }

    fn compile_plan_to_mlir(&self, plan: &CompilePlan) -> Result<String, CompileError> {
        let target = crate::components::ensure_for_plan(plan, &self.target)
            .map_err(CompileError::Component)?;
        let artifacts = self
            .compile_handlers
            .compile(
                plan,
                &CompileContext {
                    types: &self.context.types,
                    target: &target,
                },
            )
            .map_err(CompileError::Compile)?;
        let resumed = plan.resumed_mir();
        let lir = severian_lowering::lower(&resumed, &self.context.types, &target)
            .map_err(CompileError::Lowering)?;
        let ordinary = severian_mlir::render(&lir).map_err(CompileError::Mlir)?;
        severian_mlir::compose(&ordinary, &artifacts, &target).map_err(CompileError::Mlir)
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
        let ast = with_core_prelude(ast, &self.context.types)?;
        let external = severian_xxi::resolve(
            &ast,
            &self.context.types,
            &severian_abi::AbiTarget::derive(&self.target),
        )
        .map_err(|error| {
            CompileError::Diagnostic(Diagnostic::new("E000701", error.to_string(), None))
        })?;
        let mut hir = severian_semantic::analyze_with_context(
            &ast,
            &self.context.types,
            severian_semantic::AnalysisContext {
                mode: match mode {
                    CompileMode::Build => severian_semantic::AnalysisMode::Build,
                    CompileMode::Test => severian_semantic::AnalysisMode::Test,
                },
                module_name,
            },
        )
        .map_err(CompileError::Diagnostic)?;
        apply_external_calls(&ast, &external, &mut hir)?;
        severian_ownership::validate(&hir).map_err(CompileError::Diagnostic)?;
        let mut mir = severian_mir::build(&hir).map_err(CompileError::MirVerify)?;
        severian_mir::run_required_pipeline(&mut mir, &self.context)
            .map_err(CompileError::MirPass)?;
        Ok(mir)
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
                let (hir, _) = self.check_file_to_hir(source, CompileMode::Build)?;
                Ok(format!("{hir:#?}\n"))
            }
            EmitStage::Mir => {
                let mir = self.check_file_to_mir(source, CompileMode::Build)?;
                Ok(format!("{mir:#?}\n"))
            }
            EmitStage::Lir => {
                let mir = self.check_file_to_mir(source, CompileMode::Build)?;
                let plan = severian_compile::plan(&mir, &self.context.types)
                    .map_err(CompileError::Compile)?;
                let lir = severian_lowering::lower(
                    &plan.resumed_mir(),
                    &self.context.types,
                    &self.target,
                )
                .map_err(CompileError::Lowering)?;
                Ok(format!("{lir:#?}\n"))
            }
            EmitStage::Mlir => {
                let mir = self.check_file_to_mir(source, CompileMode::Build)?;
                let plan = severian_compile::plan(&mir, &self.context.types)
                    .map_err(CompileError::Compile)?;
                let artifacts = self
                    .compile_handlers
                    .compile(
                        &plan,
                        &CompileContext {
                            types: &self.context.types,
                            target: &self.target,
                        },
                    )
                    .map_err(CompileError::Compile)?;
                let lir = severian_lowering::lower(
                    &plan.resumed_mir(),
                    &self.context.types,
                    &self.target,
                )
                .map_err(CompileError::Lowering)?;
                let ordinary = severian_mlir::render(&lir).map_err(CompileError::Mlir)?;
                let text = if artifacts.is_empty() {
                    // Emission is also a debugging boundary: return the
                    // generated form even when downstream MLIR verification
                    // is the behavior under investigation.
                    ordinary
                } else {
                    severian_mlir::compose(&ordinary, &artifacts, &self.target)
                        .map_err(CompileError::Mlir)?
                };
                Ok(format!("{}\n", text.trim_end()))
            }
        }
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
            severian_compile::plan(&mir, &self.context.types).map_err(CompileError::Compile)?;
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
                            modes: Vec::new(),
                            contracts: Vec::new(),
                            body,
                            compiler_cases: Vec::new(),
                            span: case.span,
                        }));
                    let result = self.check_ast_to_mir(&ast, CompileMode::Test, "compiler_case");
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
    ) -> Result<(severian_hir::Program, Vec<SourceFile>), CompileError> {
        let graph = self.resolve_modules(source)?;
        self.check_graph_to_hir(graph, mode)
    }

    fn check_graph_to_hir(
        &self,
        mut graph: severian_modules::ModuleGraph,
        mode: CompileMode,
    ) -> Result<(severian_hir::Program, Vec<SourceFile>), CompileError> {
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
        Ok((typed.hir, sources))
    }

    fn check_file_to_mir(
        &self,
        source: &Path,
        mode: CompileMode,
    ) -> Result<MirModule, CompileError> {
        let (hir, sources) = self.check_file_to_hir(source, mode)?;
        self.check_hir_to_mir(hir, sources)
    }

    fn check_graph_to_mir(
        &self,
        graph: severian_modules::ModuleGraph,
        mode: CompileMode,
    ) -> Result<MirModule, CompileError> {
        let (hir, sources) = self.check_graph_to_hir(graph, mode)?;
        self.check_hir_to_mir(hir, sources)
    }

    fn check_hir_to_mir(
        &self,
        hir: severian_hir::Program,
        sources: Vec<SourceFile>,
    ) -> Result<MirModule, CompileError> {
        let mut merged = severian_mir::build(&hir).map_err(CompileError::MirVerify)?;
        severian_mir::run_required_pipeline(&mut merged, &self.context)
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
        let plan =
            severian_compile::plan(mir, &self.context.types).map_err(CompileError::Compile)?;
        if linker_arguments.is_empty() && !plan.has_custom_regions() {
            let resumed = plan.resumed_mir();
            let lir = severian_lowering::lower(&resumed, &self.context.types, &self.target)
                .map_err(CompileError::Lowering)?;
            if severian_backend::supports_direct_lir(&lir) {
                return severian_backend::emit_executable(&lir, output)
                    .map_err(CompileError::Backend);
            }
        }
        let mlir = self.compile_plan_to_mlir(&plan)?;
        severian_backend::emit_mlir_executable_with_linker_arguments(
            &mlir,
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
        severian_modules::resolve_with_packages_and_max_errors(source, &packages, self.max_errors)
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
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("the driver crate is nested below the repository root");
        let standard = [
            ("abi", repository.join("library/interop/abi")),
            ("ai", repository.join("library/ai")),
            ("cli", repository.join("library/system/cli")),
            ("csv", repository.join("library/data/csv")),
            ("data_format", repository.join("library/data/format")),
            ("device", repository.join("library/system/device")),
            ("driver", repository.join("library/system/driver")),
            ("environment", repository.join("library/system/environment")),
            ("ffi", repository.join("library/interop/ffi")),
            ("file", repository.join("library/system/file")),
            ("io", repository.join("library/system/io")),
            ("json", repository.join("library/data/json")),
            ("math", repository.join("library/core/math")),
            ("os", repository.join("library/system/os")),
            ("parallel", repository.join("library/compute/parallel")),
            ("path", repository.join("library/system/path")),
            ("platform", repository.join("library/system/platform")),
            ("process", repository.join("library/system/process")),
            ("tensor", repository.join("library/tensor")),
            ("yaml", repository.join("library/data/yaml")),
        ];
        let mut next = packages
            .packages
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let mut standard_ids = BTreeMap::new();
        for (name, root) in standard {
            let library = if name == "tensor" {
                root.join("src/compiler.sev")
            } else {
                root.join("src/lib.sev")
            };
            if !library.is_file() {
                return Err(CompileError::Diagnostic(Diagnostic::new(
                    "C001001",
                    format!(
                        "compiler standard package `{name}` is missing {}",
                        library.display()
                    ),
                    None,
                )));
            }
            let id = severian_modules::PackageId(next);
            next = next.saturating_add(1);
            standard_ids.insert(name.to_owned(), id);
            packages.packages.insert(
                id,
                severian_modules::ResolvedPackage {
                    id,
                    root,
                    library,
                    dependencies: BTreeMap::new(),
                },
            );
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
        for statement in &mut block.statements {
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
    module
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
    let io = SourceFile::virtual_source(
        "system/io/src/lib.sev",
        include_str!("../../../../../library/system/io/src/lib.sev"),
    );
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
        _ => false,
    });

    let size = SourceFile::virtual_source(
        "core/size/src/lib.sev",
        include_str!("../../../../../library/core/size/src/lib.sev"),
    );
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

    let text = SourceFile::virtual_source(
        "core/text/src/lib.sev",
        include_str!("../../../../../library/core/text/src/lib.sev"),
    );
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

    let prelude = SourceFile::virtual_source(
        "core/prelude.sev",
        include_str!("../../../../../library/core/prelude.sev"),
    );
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
    types.resolve_name(name).is_some()
        && arguments
            .iter()
            .all(|argument| boundary_type_is_available(argument, types))
}

fn apply_external_calls(
    ast: &severian_ast::Module,
    external: &severian_xxi::ResolvedExternalModule,
    hir: &mut severian_hir::Program,
) -> Result<(), CompileError> {
    let Some(module) = hir.modules.first_mut() else {
        return Ok(());
    };
    apply_external_calls_to_module(ast, external, module, None)
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
            "environment",
            "file",
            "io",
            "json",
            "math",
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
