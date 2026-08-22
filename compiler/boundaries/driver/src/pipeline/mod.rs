use severian_backend::{Artifact, BackendError};
use severian_compile::{CompileContext, CompileHandler, CompilePlan, CompilerRegistry};
use severian_diagnostics::Diagnostic;
use severian_mir::{Module as MirModule, Operation as MirOperation};
use severian_source::SourceFile;
use severian_target::TargetSpec;
use severian_universal::{CompilerId, UniversalContext};
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

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
        }
    }
}

impl std::error::Error for CompileError {}

pub struct Compiler {
    context: UniversalContext,
    target: TargetSpec,
    compile_handlers: CompilerRegistry,
    packages: Option<severian_modules::PackageGraph>,
    coverage: bool,
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
        Self {
            context,
            target,
            compile_handlers: CompilerRegistry::new(),
            packages: None,
            coverage: false,
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
        let mir = self.check_source_to_mir(source)?;
        self.compile_mir(&mir, output)
    }

    fn check_source_to_mir(&self, source: &SourceFile) -> Result<MirModule, CompileError> {
        let tokens = severian_lexer::scan(source).map_err(CompileError::Diagnostic)?;
        let ast = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
        let mut mir =
            self.check_ast_to_mir(&ast, CompileMode::Build, &module_name(&source.path))?;
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
        self.compile_mir(&mir, output)
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
        let mir = self.check_file_to_mir(source, CompileMode::Test)?;
        let compiler_results = self.compiler_test_results(source)?;
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
                let selected = select_test(&mir, test.function);
                let artifact =
                    self.compile_mir(&selected, &output_directory.join(format!("test-{index}")))?;
                Ok(CompiledTest {
                    name: test.name.clone(),
                    modes: test.modes.clone(),
                    execution: TestExecution::Executable(artifact),
                    expectations: test.expectations.clone(),
                })
            })
            .collect()
    }

    fn compiler_test_results(&self, source: &Path) -> Result<Vec<Option<String>>, CompileError> {
        let graph = self.resolve_modules(source)?;
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
                    for statement in &case.body {
                        match statement {
                            severian_ast::Statement::Binding(binding) => {
                                ast.items.push(severian_ast::Item::Binding(binding.clone()));
                            }
                            severian_ast::Statement::Expression(expression) => {
                                ast.items
                                    .push(severian_ast::Item::Expression(expression.clone()));
                            }
                            _ => {
                                failure = Some(
                                    "compiler expectations currently accept declaration fragments only"
                                        .into(),
                                );
                                break;
                            }
                        }
                    }
                    if failure.is_some() {
                        break;
                    }
                    let result = self.check_ast_to_mir(&ast, CompileMode::Build, "compiler_case");
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

    fn check_file_to_mir(
        &self,
        source: &Path,
        mode: CompileMode,
    ) -> Result<MirModule, CompileError> {
        let mut graph = self.resolve_modules(source)?;
        let root_package = graph
            .modules
            .last()
            .map(|module| module.package)
            .expect("a resolved module graph contains its root");
        let mut sources = Vec::new();
        let mut external = Vec::new();
        for module in &mut graph.modules {
            let source = SourceFile::load(&module.path).map_err(|error| {
                CompileError::Diagnostic(Diagnostic::new(
                    "E000001",
                    format!("could not read {}: {error}", module.path.display()),
                    None,
                ))
            })?;
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
        .map_err(CompileError::Diagnostic)?;
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
        severian_ownership::validate(&typed.hir).map_err(CompileError::Diagnostic)?;
        let mut merged = severian_mir::build(&typed.hir).map_err(CompileError::MirVerify)?;
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

    fn compile_mir(&self, mir: &MirModule, output: &Path) -> Result<Artifact, CompileError> {
        let plan =
            severian_compile::plan(mir, &self.context.types).map_err(CompileError::Compile)?;
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

    fn resolve_modules(
        &self,
        source: &Path,
    ) -> Result<severian_modules::ModuleGraph, CompileError> {
        match &self.packages {
            Some(packages) => severian_modules::resolve_with_packages(source, packages),
            None => severian_modules::resolve(source),
        }
        .map_err(CompileError::Diagnostic)
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

fn attach_block_assertion_locations(block: &mut severian_mir::Block, source: &SourceFile) {
    for operation in &mut block.operations {
        match operation {
            MirOperation::Coverage { point } => {
                let before = source
                    .text
                    .get(..usize::try_from(point.span_start).unwrap_or(0))
                    .unwrap_or("");
                let line = u32::try_from(before.bytes().filter(|byte| *byte == b'\n').count() + 1)
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
            MirOperation::Assert { origin, .. } => {
                origin.location = assertion_location(origin, source);
            }
            MirOperation::If {
                then_block,
                else_block,
                ..
            } => {
                attach_block_assertion_locations(then_block, source);
                attach_block_assertion_locations(else_block, source);
            }
            MirOperation::Match { arms, .. } => {
                for arm in arms {
                    attach_block_assertion_locations(&mut arm.body, source);
                }
            }
            _ => {}
        }
    }
}

fn remove_coverage(block: &mut severian_mir::Block) {
    block
        .operations
        .retain(|operation| !matches!(operation, MirOperation::Coverage { .. }));
    for operation in &mut block.operations {
        match operation {
            MirOperation::If {
                then_block,
                else_block,
                ..
            } => {
                remove_coverage(then_block);
                remove_coverage(else_block);
            }
            MirOperation::Match { arms, .. } => {
                for arm in arms {
                    remove_coverage(&mut arm.body);
                }
            }
            _ => {}
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
    block: &severian_mir::Block,
    output: &mut BTreeSet<severian_mir::CoveragePoint>,
) {
    for operation in &block.operations {
        match operation {
            MirOperation::Coverage { point } => {
                output.insert(point.clone());
            }
            MirOperation::If {
                then_block,
                else_block,
                ..
            } => {
                collect_coverage_points(then_block, output);
                collect_coverage_points(else_block, output);
            }
            MirOperation::Match { arms, .. } => {
                for arm in arms {
                    collect_coverage_points(&arm.body, output);
                }
            }
            _ => {}
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
            function
                .parameters
                .iter()
                .all(|parameter| boundary_type_is_available(&parameter.annotation, types))
                && boundary_type_is_available(&function.result, types)
        }
        _ => true,
    });

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
        .ok_or_else(|| external_metadata_error("foreign definition has no typed HIR item"))?;
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
    use severian_mir::{Value, ValueId};
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
    fn selecting_one_test_removes_other_test_functions_and_keeps_dense_value_ids() {
        let function = |id: u32, value: u32| severian_mir::Function {
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
            body: Some(severian_mir::Block {
                operations: vec![MirOperation::Constant {
                    value: severian_universal::LiteralValue::Integer(value.to_string()),
                    result: ValueId(value),
                }],
            }),
            cfg: None,
            call_type: severian_mir::CallType::Severian,
        };
        let module = MirModule {
            values: vec![
                Value {
                    id: ValueId(0),
                    type_id: TypeId(0),
                },
                Value {
                    id: ValueId(1),
                    type_id: TypeId(0),
                },
            ],
            functions: vec![function(0, 0), function(1, 1)],
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
        assert_eq!(selected.values.len(), 2);
        assert_eq!(selected.values[0].id, ValueId(0));
        assert!(selected.tests.is_empty());
    }
}
