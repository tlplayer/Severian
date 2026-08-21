use severian_backend::{Artifact, BackendError};
use severian_compile::{CompileContext, CompileHandler, CompilePlan, CompilerRegistry};
use severian_diagnostics::Diagnostic;
use severian_hir::BindingId;
use severian_mir::{Module as MirModule, Operation as MirOperation, Value as MirValue, ValueId};
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
        let mir = self.check_source_to_mir(source)?;
        self.compile_mir(&mir, output)
    }

    fn check_source_to_mir(&self, source: &SourceFile) -> Result<MirModule, CompileError> {
        let tokens = severian_lexer::scan(source).map_err(CompileError::Diagnostic)?;
        let ast = severian_parser::parse(&tokens).map_err(CompileError::Diagnostic)?;
        let mut mir = self.check_ast_to_mir(&ast)?;
        attach_assertion_locations(&mut mir, source);
        Ok(mir)
    }

    fn check_ast_to_mir(&self, ast: &severian_ast::Module) -> Result<MirModule, CompileError> {
        let ast = with_core_prelude(ast, &self.context.types)?;
        let external = severian_xxi::resolve(
            &ast,
            &self.context.types,
            &severian_abi::AbiTarget::derive(&self.target),
        )
        .map_err(|error| {
            CompileError::Diagnostic(Diagnostic::new("E000701", error.to_string(), None))
        })?;
        let mut hir = severian_semantic::analyze(&ast, &self.context.types)
            .map_err(CompileError::Diagnostic)?;
        apply_external_calls(&ast, &external, &mut hir);
        severian_ownership::validate(&hir).map_err(CompileError::Diagnostic)?;
        Ok(severian_mir::build(&hir))
    }

    pub fn check_source(&self, source: &SourceFile) -> Result<(), CompileError> {
        self.check_source_to_mir(source).map(|_| ())
    }

    pub fn compile_file(&self, source: &Path, output: &Path) -> Result<Artifact, CompileError> {
        let mir = self.check_file_to_mir(source)?;
        self.compile_mir(&mir, output)
    }

    pub fn check_file(&self, source: &Path) -> Result<(), CompileError> {
        self.check_file_to_mir(source).map(|_| ())
    }

    pub fn compile_tests_file(
        &self,
        source: &Path,
        output_directory: &Path,
    ) -> Result<Vec<CompiledTest>, CompileError> {
        let mir = self.check_file_to_mir(source)?;
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
                let mut selected = mir.clone();
                selected.entry = Some(test.function);
                selected.tests.clear();
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
        let graph = severian_modules::resolve(source).map_err(CompileError::Diagnostic)?;
        let mut results = Vec::new();
        for module in &graph.modules {
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
                    let result = self.check_ast_to_mir(&ast);
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

    fn check_file_to_mir(&self, source: &Path) -> Result<MirModule, CompileError> {
        let graph = severian_modules::resolve(source).map_err(CompileError::Diagnostic)?;
        let modules = graph
            .modules
            .iter()
            .map(|module| {
                let source = SourceFile::load(&module.path).map_err(|error| {
                    CompileError::Diagnostic(Diagnostic::new(
                        "E000001",
                        format!("could not read {}: {error}", module.path.display()),
                        None,
                    ))
                })?;
                let mut mir = self.check_ast_to_mir(&module.ast)?;
                attach_assertion_locations(&mut mir, &source);
                Ok(mir)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(merge_modules(modules))
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

fn apply_external_calls(
    ast: &severian_ast::Module,
    external: &severian_xxi::ResolvedExternalModule,
    hir: &mut severian_hir::Program,
) {
    let Some(module) = hir.modules.first_mut() else {
        return;
    };
    let mut foreign = external.foreign.functions.iter();
    for (ast_function, hir_function) in ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Function(function) => Some(function),
            _ => None,
        })
        .zip(&mut module.functions)
    {
        if ast_function.decorators.is_empty() {
            continue;
        }
        let declaration = foreign
            .next()
            .expect("XXI returns every decorated function in source order");
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
}

fn merge_modules(modules: Vec<MirModule>) -> MirModule {
    let root = modules.len().saturating_sub(1);
    let mut merged = MirModule::default();
    for (index, module) in modules.into_iter().enumerate() {
        let value_offset = merged.values.len() as u32;
        let function_offset = merged.functions.len() as u32;
        let binding_offset = merged
            .bindings
            .iter()
            .map(|(binding, _)| binding.0)
            .max()
            .map_or(0, |binding| binding + 1);
        merged
            .values
            .extend(module.values.iter().map(|value| MirValue {
                id: ValueId(value.id.0 + value_offset),
                type_id: value.type_id,
            }));
        merged
            .bindings
            .extend(module.bindings.iter().map(|(binding, value)| {
                (
                    BindingId(binding.0 + binding_offset),
                    ValueId(value.0 + value_offset),
                )
            }));
        merged.globals.extend(
            module
                .globals
                .iter()
                .map(|value| ValueId(value.0 + value_offset)),
        );
        merged.initializer.operations.extend(
            module
                .initializer
                .operations
                .iter()
                .map(|operation| remap_operation(operation, value_offset, function_offset)),
        );
        merged.tests.extend(
            module
                .tests
                .iter()
                .map(|test| severian_mir::TestDeclaration {
                    name: test.name.clone(),
                    modes: test.modes.clone(),
                    function: severian_mir::FunctionId(test.function.0 + function_offset),
                    expectations: test.expectations.clone(),
                }),
        );
        for mut function in module.functions {
            function.id.0 += function_offset;
            function.parameters = function
                .parameters
                .into_iter()
                .map(|value| ValueId(value.0 + value_offset))
                .collect();
            if let Some(body) = &mut function.body {
                body.operations = body
                    .operations
                    .iter()
                    .map(|operation| remap_operation(operation, value_offset, function_offset))
                    .collect();
            }
            merged.functions.push(function);
        }
        if index == root {
            merged.entry = module
                .entry
                .map(|entry| severian_mir::FunctionId(entry.0 + function_offset));
        }
    }
    merged
}

fn remap_operation(operation: &MirOperation, offset: u32, function_offset: u32) -> MirOperation {
    let value = |value: ValueId| ValueId(value.0 + offset);
    match operation {
        MirOperation::Constant {
            value: literal,
            result,
        } => MirOperation::Constant {
            value: literal.clone(),
            result: value(*result),
        },
        MirOperation::Unary {
            operator,
            operand,
            result,
        } => MirOperation::Unary {
            operator: *operator,
            operand: value(*operand),
            result: value(*result),
        },
        MirOperation::Binary {
            operator,
            left,
            right,
            result,
        } => MirOperation::Binary {
            operator: *operator,
            left: value(*left),
            right: value(*right),
            result: value(*result),
        },
        MirOperation::Call {
            function,
            arguments,
            result,
        } => MirOperation::Call {
            function: severian_mir::FunctionId(function.0 + function_offset),
            arguments: arguments.iter().copied().map(value).collect(),
            result: value(*result),
        },
        MirOperation::Return { value: returned } => MirOperation::Return {
            value: returned.map(value),
        },
        MirOperation::Assert {
            condition,
            message,
            origin,
        } => MirOperation::Assert {
            condition: value(*condition),
            message: message.map(value),
            origin: origin.clone(),
        },
        MirOperation::If {
            condition,
            then_block,
            else_block,
        } => MirOperation::If {
            condition: value(*condition),
            then_block: severian_mir::Block {
                operations: then_block
                    .operations
                    .iter()
                    .map(|operation| remap_operation(operation, offset, function_offset))
                    .collect(),
            },
            else_block: severian_mir::Block {
                operations: else_block
                    .operations
                    .iter()
                    .map(|operation| remap_operation(operation, offset, function_offset))
                    .collect(),
            },
        },
        MirOperation::Match { subject, arms } => MirOperation::Match {
            subject: value(*subject),
            arms: arms
                .iter()
                .map(|arm| severian_mir::MatchArm {
                    type_id: arm.type_id,
                    body: severian_mir::Block {
                        operations: arm
                            .body
                            .operations
                            .iter()
                            .map(|operation| remap_operation(operation, offset, function_offset))
                            .collect(),
                    },
                })
                .collect(),
        },
        MirOperation::CompiledRegionCall {
            artifact,
            inputs,
            outputs,
        } => MirOperation::CompiledRegionCall {
            artifact: *artifact,
            inputs: inputs.iter().copied().map(value).collect(),
            outputs: outputs.iter().copied().map(value).collect(),
        },
    }
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
    use severian_mir::Value;
    use severian_universal::TypeId;

    #[test]
    fn module_merge_remaps_binding_and_value_identities() {
        let source_module = || MirModule {
            values: vec![Value {
                id: ValueId(0),
                type_id: TypeId(0),
            }],
            bindings: vec![(BindingId(0), ValueId(0))],
            ..MirModule::default()
        };
        let merged = merge_modules(vec![source_module(), source_module()]);
        assert_eq!(
            merged.bindings,
            vec![(BindingId(0), ValueId(0)), (BindingId(1), ValueId(1))]
        );
    }
}
