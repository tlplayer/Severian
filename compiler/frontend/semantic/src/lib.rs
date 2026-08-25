#![forbid(unsafe_code)]

mod package;
mod queries;

pub use package::{
    analyze_package, analyze_package_with_context, DefKind, Definition, ExportMap, FunctionDecl,
    ModuleScope, PackageAnalysisContext, ProgramIndex, Resolution, Scope, SignatureId, TraitDecl,
    TypedProgram, Visibility,
};
pub use queries::{QueryError, ScopeId, SemanticQueries};
pub use severian_universal::{DeclarationId, DefId};

use severian_ast::{
    BinaryOperator as AstBinaryOperator, Expression as AstExpression,
    ExpressionKind as AstExpressionKind, Literal as AstLiteral, Statement as AstStatement,
    TypeAnnotation, UnaryOperator as AstUnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_hir::{
    Binding, BindingId, Block, BoundaryType, CallType, ClassDeclaration as HirClassDeclaration,
    ClassFieldDeclaration as HirClassFieldDeclaration, Expression, ExpressionKind,
    FunctionDeclaration, FunctionId, FunctionParameter, HirId, Module, Program, Statement,
    TraitDeclaration as HirTraitDeclaration, TraitMethodDeclaration as HirTraitMethodDeclaration,
    TraitType as HirTraitType, TypeId,
};
use severian_universal::{
    BinaryOperator, LiteralValue, TypeConstraint, TypeContext, TypeError, UnaryOperator,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMode {
    Build,
    Test,
}

#[derive(Debug, Clone, Copy)]
pub struct AnalysisContext<'a> {
    pub mode: AnalysisMode,
    pub module_name: &'a str,
}

pub fn analyze(ast: &severian_ast::Module, types: &TypeContext) -> Result<Program, Diagnostic> {
    analyze_with_context(
        ast,
        types,
        AnalysisContext {
            mode: AnalysisMode::Build,
            module_name: "memory",
        },
    )
}

pub fn analyze_with_context(
    ast: &severian_ast::Module,
    types: &TypeContext,
    context: AnalysisContext<'_>,
) -> Result<Program, Diagnostic> {
    analyze_with_package_functions(ast, types, context, &[], &[], &[], &[], &[], &[], None)
}

#[derive(Debug, Clone)]
pub(crate) struct PackageFunction {
    pub lookup: String,
    pub id: FunctionId,
    pub definition: DefId,
    pub substitution: severian_universal::Substitution,
    pub type_parameters: Vec<severian_universal::GenericParamId>,
    pub parameter_names: Vec<String>,
    pub parameters: Vec<TypeId>,
    pub parameter_defaults: Vec<Option<AstExpression>>,
    pub parameter_unions: Vec<Option<Vec<TypeId>>>,
    pub result: TypeId,
    pub result_union: Option<Vec<TypeId>>,
    pub specificity: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct PackageClass {
    pub module: severian_modules::ModuleId,
    pub ty: TypeId,
    pub declaration: severian_ast::ClassDeclaration,
    pub lookups: BTreeMap<severian_modules::ModuleId, Vec<String>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PackageList {
    pub module: severian_modules::ModuleId,
    pub ty: TypeId,
    pub element: TypeId,
}

#[derive(Debug, Clone)]
pub(crate) struct PackageConstant {
    pub lookup: String,
    pub value: AstExpression,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn analyze_with_package_functions(
    ast: &severian_ast::Module,
    types: &TypeContext,
    context: AnalysisContext<'_>,
    visible_functions: &[PackageFunction],
    own_function_ids: &[FunctionId],
    test_function_ids: &[FunctionId],
    package_classes: &[PackageClass],
    package_lists: &[PackageList],
    package_constants: &[PackageConstant],
    source_module: Option<severian_modules::ModuleId>,
) -> Result<Program, Diagnostic> {
    let normalized_ast = normalize_extensions(ast)?;
    let ast = &normalized_ast;
    validate_trait_implementations(ast)?;
    validate_class_declarations(ast)?;
    let namespace_methods = collect_trait_namespace_methods(ast)?;
    let namespace_operators = collect_trait_namespace_operators(ast)?;
    let namespace_extension_operators = collect_extension_namespace_operators(ast)?;
    let namespace_hooks = collect_trait_namespace_hooks(ast)?;
    let mut analyzer = Analyzer {
        types,
        names: BTreeMap::new(),
        mutable_variables: BTreeSet::new(),
        value_substitutions: BTreeMap::new(),
        declarations: BTreeSet::new(),
        active_type_aliases: BTreeMap::new(),
        functions: BTreeMap::new(),
        source_functions: ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Function(function) => Some(function.clone()),
                _ => None,
            })
            .fold(BTreeMap::new(), |mut functions, function| {
                functions
                    .entry(function.name.clone())
                    .or_insert_with(Vec::new)
                    .push(function);
                functions
            }),
        mocks: BTreeMap::new(),
        mock_inline_stack: BTreeSet::new(),
        function_definitions: BTreeMap::new(),
        function_substitutions: BTreeMap::new(),
        function_specificity: BTreeMap::new(),
        parameter_effects: BTreeMap::new(),
        namespace_methods,
        namespace_operators,
        namespace_extension_operators,
        namespace_hooks,
        active_operator_namespaces: BTreeMap::new(),
        active_function_name: None,
        signatures: BTreeMap::new(),
        classes: ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Class(declaration) => {
                    Some((declaration.name.clone(), declaration.clone()))
                }
                _ => None,
            })
            .collect(),
        enums: BTreeMap::new(),
        enum_variants: BTreeMap::new(),
        enum_binding_variants: BTreeMap::new(),
        class_instances: BTreeMap::new(),
        class_instances_by_type: BTreeMap::new(),
        list_types: BTreeMap::new(),
        list_elements: BTreeMap::new(),
        pointer_types: BTreeMap::new(),
        pointer_elements: BTreeMap::new(),
        map_types: BTreeMap::new(),
        map_elements: BTreeMap::new(),
        platform_layout_types: BTreeMap::new(),
        set_type: None,
        set_element: None,
        channel_types: BTreeMap::new(),
        channel_elements: BTreeMap::new(),
        tuple_types: BTreeMap::new(),
        tuple_elements: BTreeMap::new(),
        function_types: BTreeMap::new(),
        union_types: BTreeMap::new(),
        fallible_types: BTreeMap::new(),
        optional_types: BTreeSet::new(),
        callable_bindings: BTreeMap::new(),
        binding_values: BTreeMap::new(),
        callable_substitutions: BTreeMap::new(),
        error_types: BTreeSet::new(),
        preserve_error_depth: 0,
        lowered_classes: Vec::new(),
        runtime_functions: Vec::new(),
        helper_bindings: Vec::new(),
        runtime_definitions: BTreeMap::new(),
        next_hir: 0,
        next_binding: 0,
        next_comprehension: 0,
        loop_depth: 0,
        unsafe_depth: 0,
        next_class_type: u32::MAX,
    };
    analyzer.install_package_types(package_classes, package_lists, source_module)?;
    analyzer.install_enums(ast)?;
    for constant in package_constants {
        let value = analyzer.expression(&constant.value, None)?;
        analyzer
            .value_substitutions
            .insert(constant.lookup.clone(), value);
    }
    for function in visible_functions {
        for members in function
            .parameter_unions
            .iter()
            .chain(std::iter::once(&function.result_union))
            .flatten()
        {
            analyzer.instantiate_union_type(members);
        }
        analyzer
            .functions
            .entry(function.lookup.clone())
            .or_default()
            .push(function.id);
        analyzer.signatures.insert(
            function.id,
            FunctionSignature {
                parameters: function
                    .parameters
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, type_id)| SignatureParameter {
                        name: function
                            .parameter_names
                            .get(index)
                            .cloned()
                            .unwrap_or_default(),
                        type_id,
                        default: function.parameter_defaults.get(index).cloned().flatten(),
                    })
                    .collect(),
                result: function.result,
            },
        );
        analyzer.parameter_effects.insert(
            function.id,
            vec![ParameterEffect::Shared; function.parameters.len()],
        );
        analyzer
            .function_definitions
            .insert(function.id, function.definition);
        analyzer
            .function_substitutions
            .insert(function.id, function.substitution.clone());
        analyzer
            .function_specificity
            .insert(function.id, function.specificity);
    }
    let mut module = Module::default();
    let mut expected_throw_functions = BTreeSet::new();
    for test in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Test(test) => Some(test),
        _ => None,
    }) {
        collect_expected_throw_functions(&test.body, &mut expected_throw_functions);
    }

    for declaration in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Trait(declaration) => Some(declaration),
        _ => None,
    }) {
        let mut methods = Vec::with_capacity(declaration.methods.len());
        for method in &declaration.methods {
            let parameters = method
                .parameters
                .iter()
                .map(|parameter| analyzer.resolve_trait_type(&parameter.annotation))
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let result = analyzer.resolve_trait_type(&method.result)?;
            methods.push(HirTraitMethodDeclaration {
                name: method.name.clone(),
                parameters,
                result,
            });
        }
        module.traits.push(HirTraitDeclaration {
            definition: synthetic_trait_definition(context.module_name, &declaration.name),
            name: declaration.name.clone(),
            methods,
        });
    }

    // Function identities and signatures are registered before executable
    // statements are analyzed. Bodies remain ordinary analyzed blocks.
    for (ordinal, ast_function) in ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Function(function) => Some(function),
            _ => None,
        })
        .enumerate()
    {
        let id = own_function_ids
            .get(ordinal)
            .copied()
            .unwrap_or(FunctionId(module.functions.len() as u128));
        let package_function = visible_functions.iter().find(|function| function.id == id);
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
        let definition = package_function.map_or_else(
            || synthetic_definition(context.module_name, ast_function, overload_ordinal),
            |function| function.definition,
        );
        let mut parameters = Vec::new();
        let mut parameter_types = Vec::new();
        for parameter in &ast_function.parameters {
            let type_id = analyzer.resolve_source_type(&parameter.annotation)?;
            let binding = analyzer.new_binding_id();
            parameter_types.push(type_id);
            parameters.push(FunctionParameter {
                binding,
                name: parameter.name.clone(),
                contract: universal_boundary(type_id),
            });
        }
        let mut result = analyzer.resolve_source_type(&ast_function.result)?;
        if expected_throw_functions.contains(&ast_function.name)
            && !analyzer.fallible_types.contains_key(&result)
            && !analyzer.is_error_type(result)
        {
            if let Some(error) = analyzer.inferred_thrown_error(ast_function)? {
                result = analyzer.instantiate_fallible_type(result, error);
            }
        }
        let compile_route = if types.definition(result).is_some() {
            types
                .compile_route(result)
                .map_err(|error| semantic_error(error.to_string(), ast_function.result.span))?
        } else {
            severian_universal::CompileRoute::Standard
        };
        analyzer.signatures.insert(
            id,
            FunctionSignature {
                parameters: ast_function
                    .parameters
                    .iter()
                    .zip(parameter_types.iter().copied())
                    .map(|(parameter, type_id)| SignatureParameter {
                        name: parameter.name.clone(),
                        type_id,
                        default: parameter.default.clone(),
                    })
                    .collect(),
                result,
            },
        );
        analyzer.parameter_effects.insert(
            id,
            vec![ParameterEffect::Shared; ast_function.parameters.len()],
        );
        if own_function_ids.is_empty() {
            analyzer
                .functions
                .entry(ast_function.name.clone())
                .or_default()
                .push(id);
            analyzer.function_definitions.insert(id, definition);
            analyzer
                .function_substitutions
                .insert(id, severian_universal::Substitution::default());
            analyzer.function_specificity.insert(id, 0);
        }
        module.functions.push(FunctionDeclaration {
            id,
            definition,
            substitution: package_function
                .map(|function| function.substitution.clone())
                .unwrap_or_default(),
            name: ast_function.name.clone(),
            type_parameters: package_function
                .map(|function| function.type_parameters.clone())
                .unwrap_or_default(),
            parameters,
            result: universal_boundary(result),
            compile_route,
            call_type: CallType::Severian,
            body: None,
        });
    }
    let source_function_count = module.functions.len();
    if context.mode == AnalysisMode::Test {
        for (index, test) in ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Test(test) => Some(test),
                _ => None,
            })
            .enumerate()
        {
            let id = test_function_ids
                .get(index)
                .copied()
                .unwrap_or(FunctionId(module.functions.len() as u128));
            let unit = types.resolve_name("unit").expect("bootstrap defines unit");
            let mode_suffix = if test.modes.is_empty() {
                String::new()
            } else {
                format!("_with_{}", test.modes.join("-"))
            };
            let name_suffix = test
                .name
                .as_deref()
                .map(internal_name_part)
                .filter(|name| !name.is_empty())
                .map(|name| format!("_{name}"))
                .unwrap_or_default();
            module.functions.push(FunctionDeclaration {
                id,
                definition: synthetic_test_definition(context.module_name, index),
                substitution: severian_universal::Substitution::default(),
                name: format!(
                    "__sev_{}_test{mode_suffix}{name_suffix}_{index}",
                    context.module_name
                ),
                parameters: Vec::new(),
                type_parameters: Vec::new(),
                result: universal_boundary(unit),
                compile_route: severian_universal::CompileRoute::Standard,
                call_type: CallType::Severian,
                body: None,
            });
            let mut modes = test
                .modes
                .iter()
                .map(|mode| test_mode(mode, test.span))
                .collect::<Result<Vec<_>, _>>()?;
            let panic_binding = integration_panic_binding(&test.body);
            if modes.is_empty()
                && test.body.iter().any(|statement| {
                    integration_expectation(statement, panic_binding.as_deref()).is_some()
                })
            {
                modes.push(severian_hir::TestMode::Integration);
            }
            let expectations = if test.contracts.is_empty() {
                Vec::new()
            } else if modes.contains(&severian_hir::TestMode::Profile) {
                test.contracts
                    .iter()
                    .map(profile_duration_expectation)
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                return Err(Diagnostic::new(
                    "E000217",
                    "test timing contracts require the `profile` test mode",
                    Some(test.span),
                ));
            };
            module.tests.push(severian_hir::TestDeclaration {
                name: test
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("test {}", index + 1)),
                modes,
                function: id,
                expectations,
            });
        }
    }

    for item in &ast.items {
        match item {
            severian_ast::Item::Binding(binding) => {
                let id = analyzer.binding(binding, &mut module.bindings)?;
                module.initializer.statements.push(Statement::Binding(id));
            }
            severian_ast::Item::Expression(expression) => {
                module.initializer.statements.push(Statement::Expression(
                    analyzer.expression(expression, None)?,
                ));
            }
            _ => {}
        }
    }

    let globals = analyzer.names.clone();
    let ast_functions = ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Function(function) => Some(function),
        _ => None,
    });
    for (ast_function, function) in ast_functions.zip(module.functions.iter_mut()) {
        let Some(ast_body) = &ast_function.body else {
            continue;
        };
        if function
            .parameters
            .iter()
            .any(|parameter| analyzer.function_types.contains_key(&parameter.contract.ty))
        {
            // Higher-order source functions are specialized at their call sites.
            // Their unspecialized body has no concrete callable implementation.
            continue;
        }
        analyzer.names = globals.clone();
        analyzer.active_function_name = Some(ast_function.name.clone());
        analyzer.declarations.clear();
        analyzer.active_type_aliases.clear();
        for (index, name) in ast_function.type_parameters.iter().enumerate() {
            if let Some(ty) = function
                .substitution
                .get(severian_universal::GenericParamId(index as u32))
            {
                analyzer.active_type_aliases.insert(name.clone(), ty);
            }
        }
        analyzer.mocks.clear();
        analyzer.callable_substitutions.clear();
        analyzer.active_operator_namespaces = operator_namespaces(&ast_function.decorators);
        for parameter in &function.parameters {
            let type_id = parameter.contract.ty;
            if !analyzer.declarations.insert(parameter.name.clone()) {
                return Err(Diagnostic::new(
                    "E000203",
                    format!("parameter `{}` is declared more than once", parameter.name),
                    Some(ast_function.span),
                ));
            }
            let variable = severian_hir::VariableId(parameter.binding.0);
            analyzer.mutable_variables.insert(variable);
            analyzer.names.insert(
                parameter.name.clone(),
                (parameter.binding, variable, type_id),
            );
        }
        let result_type = function.result.ty;
        let (mut body, hooks) =
            analyzer.lower_function_hooks(ast_function, &mut module.bindings, result_type)?;
        for contract in ast_function
            .contracts
            .iter()
            .filter(|contract| !contract.deferred)
        {
            body.statements.push(analyzer.contract_assertion(contract)?);
        }
        body.statements.extend(
            analyzer
                .block(ast_body, &mut module.bindings, result_type)?
                .statements,
        );
        let deferred = ast_function
            .contracts
            .iter()
            .filter(|contract| contract.deferred)
            .map(|contract| analyzer.contract_assertion(contract))
            .collect::<Result<Vec<_>, _>>()?;
        if !deferred.is_empty() {
            insert_before_returns(&mut body, &deferred);
            if block_flow(ast_body) == ControlFlow::FallsThrough {
                body.statements.extend(deferred);
            }
        }
        if !hooks.is_empty() {
            insert_hook_exits(&mut body, &hooks);
            if block_flow(ast_body) == ControlFlow::FallsThrough {
                for hook in hooks.iter().rev() {
                    if let Some((field, duration)) = &hook.duration {
                        body.statements.push(Statement::FieldSet {
                            binding: hook.context,
                            field: *field,
                            value: duration.clone(),
                        });
                    }
                    body.statements
                        .extend(hook.without_phase.statements.iter().cloned());
                }
            }
        }
        let allows_fallthrough = result_type
            == types.resolve_name("unit").expect("bootstrap defines unit")
            || types
                .definition(result_type)
                .is_some_and(|definition| definition.name == "None");
        let falls_through = block_flow(ast_body) == ControlFlow::FallsThrough;
        if !allows_fallthrough && falls_through {
            return Err(Diagnostic::new(
                "E000209",
                "not every path in this function returns its declared result",
                Some(ast_function.span),
            ));
        }
        if allows_fallthrough
            && falls_through
            && types
                .definition(result_type)
                .is_some_and(|definition| definition.name == "None")
        {
            body.statements.push(Statement::Return(Some(
                analyzer.default_expression(result_type, ast_function.span)?,
            )));
        }
        if let Some(fallible) = analyzer.fallible_types.get(&result_type).copied() {
            let catch_binding = analyzer.new_binding_id();
            let catch_variable = severian_hir::VariableId(catch_binding.0);
            let catch_value = analyzer.default_expression(fallible.error, ast_function.span)?;
            module.bindings.push(Binding {
                id: catch_binding,
                variable: catch_variable,
                type_id: fallible.error,
                value: catch_value,
                mutable: false,
                preserve_error: false,
                span: ast_function.span,
            });
            let error = Expression {
                id: analyzer.next_id(),
                type_id: fallible.error,
                kind: ExpressionKind::Binding(catch_binding),
                span: ast_function.span,
            };
            let core_error = analyzer
                .types
                .resolve_name("Error")
                .expect("bootstrap defines Error");
            let error = if fallible.error == core_error {
                let string = analyzer
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                let frame = Expression {
                    id: analyzer.next_id(),
                    type_id: string,
                    kind: ExpressionKind::Literal(LiteralValue::String(ast_function.name.clone())),
                    span: ast_function.span,
                };
                analyzer.runtime_call(
                    "__sev_error_propagate",
                    &[fallible.error, string],
                    fallible.error,
                    vec![error, frame],
                    ast_function.span,
                )
            } else {
                error
            };
            let propagated = analyzer.fallible_error_expression(
                result_type,
                fallible,
                error,
                ast_function.span,
            )?;
            body = Block {
                statements: vec![Statement::Try {
                    body,
                    catch_binding,
                    catch_body: Block {
                        statements: vec![Statement::Return(Some(propagated))],
                    },
                    span: ast_function.span,
                }],
            };
        }
        let effects = function
            .parameters
            .iter()
            .map(|parameter| analyzer.inferred_parameter_effect(&body, parameter.binding))
            .collect();
        analyzer.parameter_effects.insert(function.id, effects);
        function.body = Some(body);
        if function.name == "main" {
            let arguments_type = types.resolve_name("args").expect("bootstrap defines args");
            let valid_parameters = match function.parameters.as_slice() {
                [] => true,
                [parameter] => parameter.contract.ty == arguments_type,
                _ => false,
            };
            if !valid_parameters {
                return Err(Diagnostic::new(
                    "E000209",
                    "entry must be `main()` or `main(args: args)`",
                    Some(ast_function.span),
                ));
            }
            if module.entry.replace(function.id).is_some() {
                return Err(Diagnostic::new(
                    "E000208",
                    "module defines more than one `main` function",
                    Some(ast_function.span),
                ));
            }
        }
    }
    if context.mode == AnalysisMode::Test {
        for (offset, test) in ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Test(test) => Some(test),
                _ => None,
            })
            .enumerate()
        {
            analyzer.names = globals.clone();
            analyzer.active_function_name = Some(
                test.name
                    .clone()
                    .unwrap_or_else(|| format!("test {}", offset + 1)),
            );
            analyzer.declarations.clear();
            analyzer.mocks.clear();
            analyzer.callable_substitutions.clear();
            analyzer.active_operator_namespaces.clear();
            let unit = types.resolve_name("unit").expect("bootstrap defines unit");
            let mut body = Block::default();
            if module.tests[offset]
                .modes
                .contains(&severian_hir::TestMode::Compiler)
            {
                if test.compiler_cases.is_empty() {
                    return Err(Diagnostic::new(
                        "E000217",
                        "a compiler test requires at least one `accept:` or `reject:` case",
                        Some(test.span),
                    ));
                }
                if test
                    .compiler_cases
                    .iter()
                    .any(|case| case.diagnostic_name.is_some())
                {
                    return Err(Diagnostic::new(
                        "E000217",
                        "named compiler diagnostics are not implemented; use `reject:` without a binding",
                        Some(test.span),
                    ));
                }
                module.functions[source_function_count + offset].body = Some(body);
                continue;
            }
            if module.tests[offset].modes.iter().any(|mode| {
                matches!(
                    mode,
                    severian_hir::TestMode::Model
                        | severian_hir::TestMode::Differential
                )
            }) {
                module.functions[source_function_count + offset].body = Some(body);
                continue;
            }
            let panic_binding = integration_panic_binding(&test.body);
            let case_values = if test.cases.is_empty() {
                vec![Vec::new()]
            } else {
                test.cases.clone()
            };
            for values in case_values {
                if values.len() != test.parameters.len() {
                    return Err(Diagnostic::new(
                        "E000217",
                        format!(
                            "parameterized test expects {} value(s) per case, received {}",
                            test.parameters.len(),
                            values.len()
                        ),
                        Some(test.span),
                    ));
                }
                let previous_substitutions = analyzer.value_substitutions.clone();
                for (parameter, value) in test.parameters.iter().zip(&values) {
                    let value = analyzer.expression(value, None)?;
                    analyzer.value_substitutions.insert(parameter.clone(), value);
                }
                for statement in &test.body {
                    if module.tests[offset]
                        .modes
                        .contains(&severian_hir::TestMode::Profile)
                    {
                        if let Some(expectation) = profile_statement_expectation(statement)? {
                            module.tests[offset].expectations.push(expectation);
                            continue;
                        }
                    }
                    if module.tests[offset]
                        .modes
                        .contains(&severian_hir::TestMode::Integration)
                    {
                        if let Some(expectation) =
                            integration_expectation(statement, panic_binding.as_deref())
                        {
                            module.tests[offset].expectations.push(expectation);
                            continue;
                        }
                    }
                    let mut statement = statement.clone();
                    let preludes = analyzer.lower_statement_comprehensions(
                        &mut statement,
                        &mut module.bindings,
                        unit,
                    )?;
                    body.statements.extend(preludes);
                    body.statements.push(analyzer.statement(
                        &statement,
                        &mut module.bindings,
                        unit,
                    )?);
                }
                analyzer.value_substitutions = previous_substitutions;
            }
            module.functions[source_function_count + offset].body = Some(body);
        }
    }
    module.bindings.append(&mut analyzer.helper_bindings);
    module.functions.extend(analyzer.runtime_functions.clone());
    module.classes = analyzer.lowered_classes.clone();
    Ok(Program {
        modules: vec![module],
    })
}

pub(crate) fn normalize_extensions(
    ast: &severian_ast::Module,
) -> Result<severian_ast::Module, Diagnostic> {
    let mut normalized = ast.clone();
    let extensions = ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Extension(extension) => Some(extension.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    for extension in &extensions {
        let Some((target, _)) = extension.target.named_parts() else {
            return Err(Diagnostic::new(
                "E000204",
                "an extension target must be a named type",
                Some(extension.target.span),
            ));
        };
        let source_class = normalized.items.iter().find_map(|item| match item {
            severian_ast::Item::Class(class) if class.name == target => Some(class),
            _ => None,
        });
        if source_class.is_none() && target != "set" {
            return Err(Diagnostic::new(
                "E000204",
                format!("cannot extend unknown type `{target}`"),
                Some(extension.target.span),
            ));
        }
        for method in &extension.methods {
            let directly_defined = source_class.is_some_and(|class| {
                class.fields.iter().any(|known| known.name == method.name)
                    || class.methods.iter().any(|known| known.name == method.name)
                    || class
                        .constructors
                        .iter()
                        .any(|known| known.name == method.name)
            }) || (target == "set"
                && matches!(
                    method.name.as_str(),
                    "add"
                        | "clear"
                        | "contains"
                        | "intersection"
                        | "symmetric_difference"
                        | "union"
                ));
            if directly_defined {
                return Err(Diagnostic::new(
                    "E000203",
                    format!(
                        "extension cannot replace behavior `{target}.{}` defined directly on `{target}`",
                        method.name
                    ),
                    Some(method.span),
                )
                .with_help("rename the extension member; `extend` never participates in overriding"));
            }
        }
        for operator in &extension.operators {
            let directly_defined = source_class.is_some_and(|class| {
                class
                    .operators
                    .iter()
                    .any(|known| known.operator == operator.operator)
            }) || (target == "set"
                && matches!(
                    operator.operator,
                    severian_ast::OperatorSyntax::Equal
                        | severian_ast::OperatorSyntax::NotEqual
                        | severian_ast::OperatorSyntax::Contains
                ));
            if directly_defined {
                return Err(Diagnostic::new(
                    "E000203",
                    format!(
                        "extension cannot replace operator `{}.{}` defined directly on `{target}`",
                        target,
                        ast_operator_spelling(operator.operator)
                    ),
                    Some(operator.span),
                )
                .with_help("remove the extension operator; `extend` never participates in overriding"));
            }
        }

        // An undecorated extension is globally visible and can use the normal
        // class-member path. Decorated extensions remain separate so their
        // behavior is only visible while that namespace is active.
        if extension.decorators.is_empty() {
            let Some(class) = normalized.items.iter_mut().find_map(|item| match item {
                severian_ast::Item::Class(class) if class.name == target => Some(class),
                _ => None,
            }) else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("global extensions of built-in type `{target}` are not yet supported"),
                    Some(extension.span),
                ));
            };
            class.methods.extend(extension.methods.clone());
            class.operators.extend(extension.operators.clone());
        }
    }
    normalized.items.retain(|item| {
        !matches!(item, severian_ast::Item::Extension(extension) if extension.decorators.is_empty())
    });
    Ok(normalized)
}

struct Analyzer<'a> {
    types: &'a TypeContext,
    names: BTreeMap<String, (BindingId, severian_hir::VariableId, TypeId)>,
    mutable_variables: BTreeSet<severian_hir::VariableId>,
    value_substitutions: BTreeMap<String, Expression>,
    /// Names declared in the current lexical scope. `names` also contains
    /// readable parent bindings, which may be shadowed by this set.
    declarations: BTreeSet<String>,
    active_type_aliases: BTreeMap<String, TypeId>,
    next_hir: u32,
    next_binding: u32,
    next_comprehension: u32,
    loop_depth: usize,
    unsafe_depth: usize,
    functions: BTreeMap<String, Vec<FunctionId>>,
    source_functions: BTreeMap<String, Vec<severian_ast::FunctionDeclaration>>,
    mocks: BTreeMap<String, ActiveMock>,
    mock_inline_stack: BTreeSet<String>,
    function_definitions: BTreeMap<FunctionId, DefId>,
    function_substitutions: BTreeMap<FunctionId, severian_universal::Substitution>,
    function_specificity: BTreeMap<FunctionId, u8>,
    parameter_effects: BTreeMap<FunctionId, Vec<ParameterEffect>>,
    namespace_methods: BTreeMap<String, NamespaceTraitMethod>,
    namespace_operators: BTreeMap<String, NamespaceTraitOperator>,
    namespace_extension_operators: BTreeMap<String, NamespaceExtensionOperator>,
    namespace_hooks: BTreeMap<String, NamespaceTraitHook>,
    active_operator_namespaces: BTreeMap<String, Vec<String>>,
    active_function_name: Option<String>,
    signatures: BTreeMap<FunctionId, FunctionSignature>,
    classes: BTreeMap<String, severian_ast::ClassDeclaration>,
    enums: BTreeMap<String, EnumInstance>,
    enum_variants: BTreeMap<String, (String, usize)>,
    enum_binding_variants: BTreeMap<severian_hir::VariableId, (String, String)>,
    class_instances: BTreeMap<(String, Vec<TypeId>), ClassInstance>,
    class_instances_by_type: BTreeMap<TypeId, ClassInstance>,
    list_types: BTreeMap<TypeId, TypeId>,
    list_elements: BTreeMap<TypeId, TypeId>,
    pointer_types: BTreeMap<TypeId, TypeId>,
    pointer_elements: BTreeMap<TypeId, TypeId>,
    map_types: BTreeMap<(TypeId, TypeId), TypeId>,
    map_elements: BTreeMap<TypeId, (TypeId, TypeId)>,
    platform_layout_types: BTreeMap<TypeId, TypeId>,
    set_type: Option<TypeId>,
    set_element: Option<TypeId>,
    channel_types: BTreeMap<TypeId, TypeId>,
    channel_elements: BTreeMap<TypeId, TypeId>,
    tuple_types: BTreeMap<Vec<TypeId>, TypeId>,
    tuple_elements: BTreeMap<TypeId, Vec<TypeId>>,
    function_types: BTreeMap<TypeId, FunctionType>,
    union_types: BTreeMap<TypeId, Vec<TypeId>>,
    fallible_types: BTreeMap<TypeId, FallibleType>,
    optional_types: BTreeSet<TypeId>,
    callable_bindings: BTreeMap<severian_hir::VariableId, CallableValue>,
    binding_values: BTreeMap<BindingId, Expression>,
    callable_substitutions: BTreeMap<String, ResolvedCallable>,
    error_types: BTreeSet<TypeId>,
    preserve_error_depth: usize,
    lowered_classes: Vec<HirClassDeclaration>,
    runtime_functions: Vec<FunctionDeclaration>,
    helper_bindings: Vec<Binding>,
    runtime_definitions: BTreeMap<String, DefId>,
    next_class_type: u32,
}

#[derive(Debug, Clone)]
struct NamespaceTraitMethod {
    trait_name: String,
    declaration: severian_ast::FunctionDeclaration,
    implementations: Vec<(String, severian_ast::FunctionDeclaration)>,
}

#[derive(Debug, Clone)]
struct NamespaceTraitOperator {
    trait_name: String,
    declaration: severian_ast::OperatorDeclaration,
    implementations: Vec<(String, severian_ast::OperatorImplementation)>,
}

#[derive(Debug, Clone)]
struct NamespaceExtensionOperator {
    namespace: String,
    target: TypeAnnotation,
    implementation: severian_ast::OperatorImplementation,
}

#[derive(Debug, Clone)]
struct NamespaceTraitHook {
    trait_name: String,
    members: Vec<NamespaceTraitHookMember>,
}

#[derive(Debug, Clone)]
struct NamespaceTraitHookMember {
    method_name: String,
    selectors: Vec<String>,
    implementations: Vec<(String, severian_ast::FunctionDeclaration)>,
}

#[derive(Debug, Clone)]
struct LoweredHook {
    context: BindingId,
    result_field: Option<u32>,
    error_field: Option<u32>,
    duration: Option<(u32, Expression)>,
    without_phase: Block,
}

#[derive(Debug, Clone)]
struct ClassInstance {
    ty: TypeId,
    name: String,
    fields: Vec<HirClassFieldDeclaration>,
    source_fields: Vec<severian_ast::PropertyDeclaration>,
    constructors: Vec<severian_ast::FunctionDeclaration>,
    methods: Vec<severian_ast::FunctionDeclaration>,
}

#[derive(Debug, Clone)]
struct EnumInstance {
    ty: TypeId,
    name: String,
    fields: Vec<HirClassFieldDeclaration>,
    variants: Vec<severian_ast::EnumVariant>,
}

fn enum_payload_index(
    variants: &[severian_ast::EnumVariant],
    variant: usize,
    payload: usize,
) -> usize {
    1 + variants
        .iter()
        .take(variant)
        .map(|variant| variant.fields.len())
        .sum::<usize>()
        + payload
}

#[derive(Debug, Clone, Copy)]
struct FallibleType {
    success: TypeId,
    error: TypeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FunctionType {
    parameters: Vec<TypeId>,
    result: TypeId,
}

#[derive(Debug, Clone)]
enum CallableValue {
    Direct(FunctionId),
    Lambda {
        parameters: Vec<String>,
        body: AstExpression,
        closure: BindingId,
        closure_type: TypeId,
        captures: Vec<(String, TypeId)>,
    },
}

#[derive(Debug, Clone)]
struct ResolvedCallable {
    value: CallableValue,
    signature: FunctionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ParameterEffect {
    Shared,
    Exclusive,
    Move,
}

struct PendingLambda {
    parameters: Vec<String>,
    body: AstExpression,
    closure_type: TypeId,
    captures: Vec<(String, TypeId)>,
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    parameters: Vec<SignatureParameter>,
    result: TypeId,
}

#[derive(Debug, Clone)]
struct SignatureParameter {
    name: String,
    type_id: TypeId,
    default: Option<AstExpression>,
}

#[derive(Debug, Clone)]
struct ActiveMock {
    cases: Vec<severian_ast::MockCase>,
    fallback: AstExpression,
}

enum Prepared {
    Literal(severian_universal::LiteralValue, severian_source::Span),
    Resolved(Expression),
}

impl Prepared {
    fn constraint(&self) -> TypeConstraint {
        match self {
            Self::Literal(value, _) => TypeConstraint::Literal(value.kind()),
            Self::Resolved(expression) => TypeConstraint::Known(expression.type_id),
        }
    }
}

impl Analyzer<'_> {
    fn inferred_thrown_error(
        &self,
        function: &severian_ast::FunctionDeclaration,
    ) -> Result<Option<TypeId>, Diagnostic> {
        let Some(body) = function.body.as_deref() else {
            return Ok(None);
        };
        let mut names = BTreeSet::new();
        collect_thrown_error_names(body, &mut names);
        let mut resolved = names
            .iter()
            .filter_map(|name| {
                self.class_instances
                    .get(&(name.clone(), Vec::new()))
                    .map(|instance| instance.ty)
                    .or_else(|| self.types.resolve_name(name))
            })
            .filter(|ty| self.is_error_type(*ty))
            .collect::<BTreeSet<_>>();
        if resolved.len() > 1 {
            return Err(Diagnostic::new(
                "E000204",
                "implicit thrown-error inference requires one error type",
                Some(function.span),
            )
            .with_help("declare an explicit fallible result union for multiple error types"));
        }
        Ok(resolved.pop_first())
    }

    fn call_thrown_error(&self, expression: &AstExpression) -> Option<TypeId> {
        let AstExpressionKind::Call { callee, arguments } = &expression.kind else {
            return None;
        };
        let name = callable_path(callee)?;
        self.functions.get(&name)?.iter().find_map(|function| {
            let signature = self.signatures.get(function)?;
            if signature.parameters.len() != arguments.len() {
                return None;
            }
            self.fallible_types
                .get(&signature.result)
                .map(|fallible| fallible.error)
        })
    }

    fn install_enums(&mut self, ast: &severian_ast::Module) -> Result<(), Diagnostic> {
        let declarations = ast
            .items
            .iter()
            .filter_map(|item| match item {
                severian_ast::Item::Enum(declaration) => Some(declaration),
                _ => None,
            })
            .cloned()
            .collect::<Vec<_>>();
        for declaration in &declarations {
            let variants = declaration
                .variants
                .iter()
                .map(|variant| variant.name.as_str())
                .collect::<BTreeSet<_>>();
            let mut accepted_values: Vec<(&AstLiteral, &str)> = Vec::new();
            for variant in &declaration.variants {
                if !variant.accepted_values.is_empty() && !variant.fields.is_empty() {
                    return Err(Diagnostic::new(
                        "E000213",
                        format!(
                            "enum variant `{}.{}` cannot combine accepted values with payload fields",
                            declaration.name, variant.name
                        ),
                        Some(variant.span),
                    ));
                }
                for value in &variant.accepted_values {
                    if let Some((_, previous)) = accepted_values
                        .iter()
                        .find(|(accepted, _)| *accepted == value)
                    {
                        return Err(Diagnostic::new(
                            "E000213",
                            format!(
                                "enum accepted value is shared by `{}.{previous}` and `{}.{}`",
                                declaration.name, declaration.name, variant.name
                            ),
                            Some(variant.span),
                        )
                        .with_help("each accepted value must identify exactly one enum variant"));
                    }
                    accepted_values.push((value, variant.name.as_str()));
                }
                for transition in &variant.transitions {
                    if !variants.contains(transition.as_str()) {
                        return Err(Diagnostic::new(
                            "E000213",
                            format!(
                                "enum variant `{}.{}` transitions to unknown variant `{transition}`",
                                declaration.name, variant.name
                            ),
                            Some(variant.span),
                        ));
                    }
                }
            }
        }
        for declaration in &declarations {
            let ty = self
                .class_instances
                .get(&(declaration.name.clone(), Vec::new()))
                .map(|instance| instance.ty)
                .unwrap_or_else(|| {
                    let ty = TypeId(self.next_class_type);
                    self.next_class_type = self.next_class_type.saturating_sub(1);
                    ty
                });
            let placeholder = ClassInstance {
                ty,
                name: declaration.name.clone(),
                fields: Vec::new(),
                source_fields: Vec::new(),
                constructors: Vec::new(),
                methods: Vec::new(),
            };
            self.class_instances
                .insert((declaration.name.clone(), Vec::new()), placeholder.clone());
            self.class_instances_by_type.insert(ty, placeholder);
            self.enums.insert(
                declaration.name.clone(),
                EnumInstance {
                    ty,
                    name: declaration.name.clone(),
                    fields: Vec::new(),
                    variants: declaration.variants.clone(),
                },
            );
        }
        let integer = self.tag_type();
        for declaration in declarations {
            let ty = self.enums[&declaration.name].ty;
            let mut fields = vec![HirClassFieldDeclaration {
                name: "__tag".into(),
                ty: integer,
            }];
            for (ordinal, variant) in declaration.variants.iter().enumerate() {
                self.enum_variants
                    .insert(variant.name.clone(), (declaration.name.clone(), ordinal));
                self.enum_variants.insert(
                    format!("{}.{}", declaration.name, variant.name),
                    (declaration.name.clone(), ordinal),
                );
                for field in &variant.fields {
                    let field_type = self.resolve_source_type(&field.annotation)?;
                    fields.push(HirClassFieldDeclaration {
                        name: format!("__variant_{ordinal}_{}", field.name),
                        ty: field_type,
                    });
                }
            }
            let instance = self.enums.get_mut(&declaration.name).unwrap();
            instance.fields.clone_from(&fields);
            self.class_instances.insert(
                (declaration.name.clone(), Vec::new()),
                ClassInstance {
                    ty,
                    name: declaration.name.clone(),
                    fields: fields.clone(),
                    source_fields: Vec::new(),
                    constructors: Vec::new(),
                    methods: Vec::new(),
                },
            );
            self.class_instances_by_type.insert(
                ty,
                ClassInstance {
                    ty,
                    name: declaration.name.clone(),
                    fields: fields.clone(),
                    source_fields: Vec::new(),
                    constructors: Vec::new(),
                    methods: Vec::new(),
                },
            );
            self.lowered_classes.retain(|class| class.id != ty);
            self.lowered_classes.push(HirClassDeclaration {
                id: ty,
                name: declaration.name,
                fields,
            });
        }
        Ok(())
    }

    fn enum_variant_expression(&self, expression: &AstExpression) -> Option<(String, String)> {
        let path = match &expression.kind {
            AstExpressionKind::Call { callee, .. } => callable_path(callee),
            _ => callable_path(expression),
        };
        if let Some(path) = path {
            if let Some((enum_name, ordinal)) = self.enum_variants.get(&path) {
                let variant = self.enums[enum_name].variants[*ordinal].name.clone();
                return Some((enum_name.clone(), variant));
            }
        }
        if let AstExpressionKind::Name(name) = &expression.kind {
            if let Some((_, variable, _)) = self.names.get(name) {
                return self.enum_binding_variants.get(variable).cloned();
            }
        }
        None
    }

    fn validate_enum_transition(
        &self,
        enum_name: &str,
        from: &str,
        to: &str,
        span: severian_source::Span,
    ) -> Result<(), Diagnostic> {
        let instance = &self.enums[enum_name];
        if !instance
            .variants
            .iter()
            .any(|variant| !variant.transitions.is_empty())
        {
            return Ok(());
        }
        let variant = instance
            .variants
            .iter()
            .find(|variant| variant.name == from)
            .expect("known enum state names a declared variant");
        if variant.transitions.iter().any(|allowed| allowed == to) {
            return Ok(());
        }
        let allowed = if variant.transitions.is_empty() {
            format!("`{enum_name}.{from}` is a terminal state")
        } else {
            format!(
                "allowed next state(s): {}",
                variant
                    .transitions
                    .iter()
                    .map(|target| format!("`{enum_name}.{target}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        Err(Diagnostic::new(
            "E000213",
            format!("invalid enum transition `{enum_name}.{from} -> {enum_name}.{to}`"),
            Some(span),
        )
        .with_help(allowed))
    }

    fn install_package_types(
        &mut self,
        classes: &[PackageClass],
        lists: &[PackageList],
        source_module: Option<severian_modules::ModuleId>,
    ) -> Result<(), Diagnostic> {
        let storage = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        for list in lists {
            self.list_types.insert(list.element, list.ty);
            self.list_elements.insert(list.ty, list.element);
            if source_module == Some(list.module) {
                self.lowered_classes.push(HirClassDeclaration {
                    id: list.ty,
                    name: format!("list[type#{}]", list.element.0),
                    fields: vec![HirClassFieldDeclaration {
                        name: "storage".into(),
                        ty: storage,
                    }],
                });
            }
            self.next_class_type = self.next_class_type.min(list.ty.0.saturating_sub(1));
        }
        // Install every visible class name before resolving fields so package
        // annotations may refer forward and through an imported namespace.
        let mut resolved_visible_instances = self.class_instances.clone();
        for package_class in classes {
            if source_module
                .is_some_and(|module| !package_class.lookups.contains_key(&module))
            {
                continue;
            }
            if !package_class.declaration.type_parameters.is_empty() {
                if let Some(source_module) = source_module {
                    for lookup in package_class
                        .lookups
                        .get(&source_module)
                        .into_iter()
                        .flatten()
                    {
                        self.classes
                            .insert(lookup.clone(), package_class.declaration.clone());
                    }
                }
                continue;
            }
            let is_error = package_class
                .declaration
                .traits
                .iter()
                .any(|implemented| implemented.simple_name() == Some("Error"))
                || package_class.declaration.name.ends_with("Error");
            let placeholder = ClassInstance {
                ty: package_class.ty,
                name: package_class.declaration.name.clone(),
                fields: Vec::new(),
                source_fields: package_class.declaration.fields.clone(),
                constructors: package_class.declaration.constructors.clone(),
                methods: package_class.declaration.methods.clone(),
            };
            self.class_instances_by_type
                .insert(package_class.ty, placeholder.clone());
            if let Some(source_module) = source_module {
                for lookup in package_class
                    .lookups
                    .get(&source_module)
                    .into_iter()
                    .flatten()
                {
                    self.class_instances
                        .insert((lookup.clone(), Vec::new()), placeholder.clone());
                }
            }
            if is_error {
                self.error_types.insert(package_class.ty);
            }
        }
        for package_class in classes {
            if source_module
                .is_some_and(|module| !package_class.lookups.contains_key(&module))
            {
                continue;
            }
            if !package_class.declaration.type_parameters.is_empty() {
                continue;
            }
            self.class_instances = resolved_visible_instances.clone();
            for definition_class in classes {
                let Some(instance) = self
                    .class_instances_by_type
                    .get(&definition_class.ty)
                    .cloned()
                else {
                    continue;
                };
                for lookup in definition_class
                    .lookups
                    .get(&package_class.module)
                    .into_iter()
                    .flatten()
                {
                    self.class_instances
                        .insert((lookup.clone(), Vec::new()), instance.clone());
                }
            }
            let is_error = package_class
                .declaration
                .traits
                .iter()
                .any(|implemented| implemented.simple_name() == Some("Error"))
                || package_class.declaration.name.ends_with("Error");
            let mut fields = package_class
                .declaration
                .fields
                .iter()
                .map(|field| {
                    Ok(HirClassFieldDeclaration {
                        name: field.name.clone(),
                        ty: self.resolve_source_type(&field.annotation)?,
                    })
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            if is_error && fields.is_empty() {
                fields.push(HirClassFieldDeclaration {
                    name: "__error".into(),
                    ty: self
                        .types
                        .resolve_name("Error")
                        .expect("bootstrap defines Error"),
                });
            }
            let instance = ClassInstance {
                ty: package_class.ty,
                name: package_class.declaration.name.clone(),
                fields: fields.clone(),
                source_fields: package_class.declaration.fields.clone(),
                constructors: package_class.declaration.constructors.clone(),
                methods: package_class.declaration.methods.clone(),
            };
            self.class_instances = resolved_visible_instances;
            self.class_instances_by_type
                .insert(package_class.ty, instance.clone());
            if let Some(source_module) = source_module {
                for lookup in package_class
                    .lookups
                    .get(&source_module)
                    .into_iter()
                    .flatten()
                {
                    self.class_instances
                        .insert((lookup.clone(), Vec::new()), instance.clone());
                }
            }
            resolved_visible_instances = self.class_instances.clone();
            if source_module == Some(package_class.module) {
                self.lowered_classes.push(HirClassDeclaration {
                    id: package_class.ty,
                    name: package_class.declaration.name.clone(),
                    fields,
                });
            }
            self.next_class_type = self
                .next_class_type
                .min(package_class.ty.0.saturating_sub(1));
        }
        Ok(())
    }

    fn resolve_source_type(&mut self, annotation: &TypeAnnotation) -> Result<TypeId, Diagnostic> {
        if let Some((
            "borrowed" | "owned" | "transferred" | "out" | "inout" | "nullable",
            [inner],
        )) = annotation.named_parts()
        {
            return self.resolve_source_type(inner);
        }
        if let severian_ast::TypeAnnotationKind::Function { parameters, result } = &annotation.kind
        {
            let parameters = parameters
                .iter()
                .map(|parameter| self.resolve_source_type(parameter))
                .collect::<Result<Vec<_>, _>>()?;
            let result = self.resolve_source_type(result)?;
            return Ok(self.instantiate_function_type(&parameters, result));
        }
        if let severian_ast::TypeAnnotationKind::Union(members) = &annotation.kind {
            let optional = members
                .iter()
                .any(|member| matches!(member.simple_name(), Some("None" | "absent")));
            let mut success = Vec::new();
            let mut errors = Vec::new();
            for member in members {
                if matches!(member.simple_name(), Some("None" | "absent")) {
                    continue;
                }
                let ty = self.resolve_source_type(member)?;
                if self.is_error_type(ty) {
                    errors.push(ty);
                } else {
                    success.push(ty);
                }
            }
            success.sort();
            success.dedup();
            errors.sort();
            errors.dedup();
            if let ([success], [error]) = (success.as_slice(), errors.as_slice()) {
                return Ok(self.instantiate_fallible_type(*success, *error));
            }
            if errors.is_empty() {
                return match success.as_slice() {
                    [success] => {
                        if optional {
                            self.optional_types.insert(*success);
                        }
                        Ok(*success)
                    }
                    [_, _, ..] => Ok(self.instantiate_union_type(&success)),
                    [] => Err(Diagnostic::new(
                        "E000204",
                        "a union must contain at least one concrete type",
                        Some(annotation.span),
                    )),
                };
            }
            return Err(Diagnostic::new(
                "E000204",
                "fallible unions currently require one success type and one error type",
                Some(annotation.span),
            ));
        }
        if let Some(("list", [element])) = annotation.named_parts() {
            let element = self.resolve_source_type(element)?;
            if let Some(list) = self.list_types.get(&element) {
                return Ok(*list);
            }
            return Ok(self.instantiate_list_type(element));
        }
        if let Some(("set", [element])) = annotation.named_parts() {
            let element = self.resolve_source_type(element)?;
            return self.ensure_set_type(element, annotation.span);
        }
        if let Some(("pointer", [element])) = annotation.named_parts() {
            let element = self.resolve_source_type(element)?;
            return Ok(self.instantiate_pointer_type(element));
        }
        if let Some(("tuple", elements)) = annotation.named_parts() {
            let elements = elements
                .iter()
                .map(|element| self.resolve_source_type(element))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(self.instantiate_tuple_type(&elements));
        }
        if let Some(("map", [key, value])) = annotation.named_parts() {
            let key = self.resolve_source_type(key)?;
            let value = self.resolve_source_type(value)?;
            return Ok(self.instantiate_map_type(key, value));
        }
        if let Some(name) = annotation.simple_name() {
            if let Some(ty) = self.active_type_aliases.get(name) {
                return Ok(*ty);
            }
            if let Some(instance) = self.class_instances.get(&(name.to_owned(), Vec::new())) {
                return Ok(instance.ty);
            }
        }
        resolve_type_annotation(self.types, annotation)
    }

    fn is_error_type(&self, ty: TypeId) -> bool {
        self.types.resolve_name("Error") == Some(ty) || self.error_types.contains(&ty)
    }

    fn resolve_trait_type(
        &mut self,
        annotation: &TypeAnnotation,
    ) -> Result<HirTraitType, Diagnostic> {
        if annotation.simple_name() == Some("Self") {
            Ok(HirTraitType::SelfType)
        } else {
            self.resolve_source_type(annotation)
                .map(HirTraitType::Concrete)
        }
    }

    fn new_binding_id(&mut self) -> BindingId {
        let id = BindingId(self.next_binding);
        self.next_binding += 1;
        id
    }

    fn binding(
        &mut self,
        ast_binding: &severian_ast::Binding,
        bindings: &mut Vec<Binding>,
    ) -> Result<BindingId, Diagnostic> {
        let inferred_update = !ast_binding.update
            && !ast_binding.mutable
            && ast_binding.annotation.is_none()
            && self.declarations.contains(&ast_binding.name);
        let is_update = ast_binding.update || inferred_update;
        let update_type = if is_update {
            Some(
                self.names
                    .get(&ast_binding.name)
                    .map(|(_, _, type_id)| *type_id)
                    .ok_or_else(|| {
                        Diagnostic::new(
                            "E000201",
                            format!("cannot update unknown binding `{}`", ast_binding.name),
                            Some(ast_binding.span),
                        )
                    })?,
            )
        } else {
            None
        };
        if is_update {
            let variable = self.names[&ast_binding.name].1;
            if !self.mutable_variables.contains(&variable) {
                return Err(Diagnostic::new(
                    "E000203",
                    format!(
                        "cannot assign to immutable binding `{}`; declare it with `:=` to allow reassignment",
                        ast_binding.name
                    ),
                    Some(ast_binding.span),
                ));
            }
        }
        if !is_update && !self.declarations.insert(ast_binding.name.clone()) {
            return Err(Diagnostic::new(
                "E000203",
                format!("binding `{}` is already defined", ast_binding.name),
                Some(ast_binding.span),
            ));
        }
        let next_enum_variant = self.enum_variant_expression(&ast_binding.value);
        if is_update {
            let variable = self.names[&ast_binding.name].1;
            if let (Some((enum_name, from)), Some((next_enum, to))) = (
                self.enum_binding_variants.get(&variable).cloned(),
                next_enum_variant.as_ref(),
            ) {
                if enum_name == *next_enum {
                    self.validate_enum_transition(&enum_name, &from, to, ast_binding.value.span)?;
                }
            }
        }
        let expected = ast_binding
            .annotation
            .as_ref()
            .map(|annotation| self.resolve_source_type(annotation))
            .transpose()?
            .or(update_type);
        let (value, pending_lambda) =
            if let AstExpressionKind::Lambda { parameters, body } = &ast_binding.value.kind {
                if is_update {
                    return Err(Diagnostic::new(
                        "E000205",
                    "lambda bindings cannot be reassigned",
                    Some(ast_binding.span),
                ));
            }
            self.lambda_binding_value(parameters, body, ast_binding.value.span)?
        } else {
            if ast_binding.preserve_error {
                self.preserve_error_depth += 1;
            }
            let value = self.expression(&ast_binding.value, expected);
            if ast_binding.preserve_error {
                self.preserve_error_depth -= 1;
                }
                (value?, None)
            };
        let type_id = pending_lambda.as_ref().map_or_else(
            || expected.unwrap_or(value.type_id),
            |lambda| lambda.closure_type,
        );
        if pending_lambda.is_none() && !self.types.assignable(value.type_id, type_id) {
            return Err(Diagnostic::new(
                "E000205",
                "binding value is not assignable to its declared type",
                Some(ast_binding.value.span),
            ));
        }
        let id = self.new_binding_id();
        let variable = if is_update {
            self.names[&ast_binding.name].1
        } else {
            severian_hir::VariableId(id.0)
        };
        if !is_update && ast_binding.mutable {
            self.mutable_variables.insert(variable);
        }
        let mutable = self.mutable_variables.contains(&variable);
        if let Some(variant) = next_enum_variant {
            self.enum_binding_variants.insert(variable, variant);
        } else if is_update {
            self.enum_binding_variants.remove(&variable);
        }
        self.names
            .insert(ast_binding.name.clone(), (id, variable, type_id));
        if let Some(lambda) = pending_lambda {
            self.callable_bindings.insert(
                variable,
                CallableValue::Lambda {
                    parameters: lambda.parameters,
                    body: lambda.body,
                    closure: id,
                    closure_type: lambda.closure_type,
                    captures: lambda.captures,
                },
            );
        }
        self.binding_values.insert(id, value.clone());
        bindings.push(Binding {
            id,
            variable,
            type_id,
            value,
            mutable,
            preserve_error: ast_binding.preserve_error,
            span: ast_binding.span,
        });
        Ok(id)
    }

    fn lower_destructure_value(
        &mut self,
        names: &[String],
        value: Expression,
        mutable: bool,
        span: severian_source::Span,
        bindings: &mut Vec<Binding>,
    ) -> Result<Statement, Diagnostic> {
        let elements = self
            .tuple_elements
            .get(&value.type_id)
            .cloned()
            .ok_or_else(|| {
                Diagnostic::new(
                    "E000205",
                    "a destructuring binding requires a tuple value",
                    Some(span),
                )
            })?;
        if elements.len() != names.len() {
            return Err(Diagnostic::new(
                "E000205",
                format!(
                    "tuple has {} element(s), but the binding pattern has {} name(s)",
                    elements.len(),
                    names.len()
                ),
                Some(span),
            ));
        }

        let tuple_type = value.type_id;
        let temporary = self.new_binding_id();
        bindings.push(Binding {
            id: temporary,
            variable: severian_hir::VariableId(temporary.0),
            type_id: value.type_id,
            value,
            mutable: false,
            preserve_error: false,
            span,
        });
        let mut statements = vec![Statement::Binding(temporary)];
        for (index, (name, element)) in names.iter().zip(elements).enumerate() {
            let existing = self.names.get(name).copied();
            let (variable, is_update) = if let Some((_, variable, known_type)) = existing {
                if !self.mutable_variables.contains(&variable) {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("cannot assign to immutable binding `{name}`"),
                        Some(span),
                    ));
                }
                if !self.types.assignable(element, known_type) {
                    return Err(Diagnostic::new(
                        "E000205",
                        format!("tuple element is not assignable to binding `{name}`"),
                        Some(span),
                    ));
                }
                (variable, true)
            } else {
                if !self.declarations.insert(name.clone()) {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("binding `{name}` is already defined"),
                        Some(span),
                    ));
                }
                let variable = severian_hir::VariableId(self.next_binding);
                if mutable {
                    self.mutable_variables.insert(variable);
                }
                (variable, false)
            };
            let id = self.new_binding_id();
            let field = Expression {
                id: self.next_id(),
                type_id: element,
                kind: ExpressionKind::Field {
                    object: Box::new(Expression {
                        id: self.next_id(),
                        type_id: tuple_type,
                        kind: ExpressionKind::Binding(temporary),
                        span,
                    }),
                    index: index as u32,
                },
                span,
            };
            bindings.push(Binding {
                id,
                variable,
                type_id: element,
                value: field,
                mutable: mutable || is_update,
                preserve_error: false,
                span,
            });
            self.names.insert(name.clone(), (id, variable, element));
            statements.push(Statement::Binding(id));
        }
        Ok(Statement::Sequence(Block { statements }))
    }

    fn lambda_binding_value(
        &mut self,
        parameters: &[String],
        body: &AstExpression,
        span: severian_source::Span,
    ) -> Result<(Expression, Option<PendingLambda>), Diagnostic> {
        let mut mentioned = BTreeSet::new();
        collect_expression_names(body, &mut mentioned);
        for parameter in parameters {
            mentioned.remove(parameter);
        }
        let captures = mentioned
            .into_iter()
            .filter_map(|name| {
                self.names
                    .get(&name)
                    .map(|(binding, _, ty)| (name, *binding, *ty))
            })
            .collect::<Vec<_>>();
        let capture_types = captures
            .iter()
            .map(|(name, _, ty)| (name.clone(), *ty))
            .collect::<Vec<_>>();
        let closure_type = self.instantiate_lambda_type(&capture_types);
        let fields = captures
            .iter()
            .map(|(_, binding, ty)| Expression {
                id: self.next_id(),
                type_id: *ty,
                kind: ExpressionKind::Binding(*binding),
                span,
            })
            .collect();
        Ok((
            Expression {
                id: self.next_id(),
                type_id: closure_type,
                kind: ExpressionKind::Aggregate {
                    class: closure_type,
                    fields,
                },
                span,
            },
            Some(PendingLambda {
                parameters: parameters.to_vec(),
                body: body.clone(),
                closure_type,
                captures: capture_types,
            }),
        ))
    }

    fn statement(
        &mut self,
        statement: &AstStatement,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        match statement {
            AstStatement::Try {
                body,
                catch_binding,
                catch_annotation,
                catch_body,
                span,
            } => {
                let error_type = catch_annotation
                    .as_ref()
                    .map(|annotation| self.resolve_source_type(annotation))
                    .transpose()?
                    .unwrap_or_else(|| {
                        self.types
                            .resolve_name("Error")
                            .expect("bootstrap defines Error")
                    });
                if !self.is_error_type(error_type) {
                    return Err(Diagnostic::new(
                        "E000215",
                        "a catch binding must have an error type",
                        Some(*span),
                    ));
                }

                let outer_names = self.names.clone();
                let outer_declarations = self.declarations.clone();
                let body = self.block(body, bindings, result_type)?;
                self.names.clone_from(&outer_names);
                self.declarations.clone_from(&outer_declarations);

                let id = self.new_binding_id();
                let variable = severian_hir::VariableId(id.0);
                if !self.declarations.insert(catch_binding.clone()) {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("binding `{catch_binding}` is declared more than once"),
                        Some(*span),
                    ));
                }
                self.names
                    .insert(catch_binding.clone(), (id, variable, error_type));
                let default = self.default_expression(error_type, *span)?;
                bindings.push(Binding {
                    id,
                    variable,
                    type_id: error_type,
                    value: default,
                    mutable: false,
                    preserve_error: false,
                    span: *span,
                });
                let catch_body = self.block(catch_body, bindings, result_type)?;
                self.names = outer_names;
                self.declarations = outer_declarations;
                Ok(Statement::Try {
                    body,
                    catch_binding: id,
                    catch_body,
                    span: *span,
                })
            }
            AstStatement::FallibleElse {
                value,
                error_binding,
                body,
                span,
            } => {
                self.preserve_error_depth += 1;
                let result = self.expression(value, None);
                self.preserve_error_depth -= 1;
                let result = result?;
                let Some(fallible) = self.fallible_types.get(&result.type_id).copied() else {
                    return Err(Diagnostic::new(
                        "E000204",
                        "`else error:` requires a fallible result",
                        Some(value.span),
                    ));
                };
                let result_binding = self.new_binding_id();
                let result_variable = severian_hir::VariableId(result_binding.0);
                bindings.push(Binding {
                    id: result_binding,
                    variable: result_variable,
                    type_id: result.type_id,
                    value: result,
                    mutable: false,
                    preserve_error: true,
                    span: *span,
                });
                let result_value = Expression {
                    id: self.next_id(),
                    type_id: fallible_type_id(fallible.success, fallible.error),
                    kind: ExpressionKind::Binding(result_binding),
                    span: *span,
                };
                let boolean = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");
                let ok = Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Field {
                        object: Box::new(result_value.clone()),
                        index: 0,
                    },
                    span: *span,
                };
                let failed = Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Unary {
                        operator: UnaryOperator::Not,
                        operand: Box::new(ok),
                    },
                    span: *span,
                };
                let error = Expression {
                    id: self.next_id(),
                    type_id: fallible.error,
                    kind: ExpressionKind::Field {
                        object: Box::new(result_value),
                        index: 2,
                    },
                    span: *span,
                };
                let id = self.new_binding_id();
                let variable = severian_hir::VariableId(id.0);
                bindings.push(Binding {
                    id,
                    variable,
                    type_id: fallible.error,
                    value: error,
                    mutable: false,
                    preserve_error: false,
                    span: *span,
                });
                let outer_names = self.names.clone();
                let outer_declarations = self.declarations.clone();
                if !self.declarations.insert(error_binding.clone()) {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("binding `{error_binding}` is declared more than once"),
                        Some(*span),
                    ));
                }
                self.names
                    .insert(error_binding.clone(), (id, variable, fallible.error));
                let handler = self.block(body, bindings, result_type)?;
                self.names = outer_names;
                self.declarations = outer_declarations;
                Ok(Statement::Sequence(Block {
                    statements: vec![
                        Statement::Binding(result_binding),
                        Statement::Binding(id),
                        Statement::If {
                            condition: failed,
                            then_block: handler,
                            else_block: Block::default(),
                        },
                    ],
                }))
            }
            AstStatement::Defer { .. } => Ok(Statement::Sequence(Block::default())),
            AstStatement::Unsafe { body, .. } => {
                self.unsafe_depth += 1;
                let lowered = self.block(body, bindings, result_type);
                self.unsafe_depth -= 1;
                Ok(Statement::Sequence(lowered?))
            }
            AstStatement::While {
                condition,
                initializer,
                guards,
                body,
                span,
            } => {
                let mut sequence = Block::default();
                if let Some(initializer) = initializer {
                    sequence
                        .statements
                        .push(Statement::Binding(self.binding(initializer, bindings)?));
                }
                let condition = self.condition_expression(condition)?;
                let mut guarded_prefix = Block::default();
                let mut guarded_outer_names = None;
                let mut guarded_outer_declarations = None;
                if !guards.is_empty() {
                    let Some(AstStatement::Destructure {
                        names,
                        value,
                        mutable: _,
                        span: binding_span,
                    }) = body.first()
                    else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "a guarded loop must begin by destructuring the value used by its guards",
                            Some(*span),
                        ));
                    };
                    let resolved = self.expression(value, None)?;
                    let elements = self
                        .tuple_elements
                        .get(&resolved.type_id)
                        .cloned()
                        .ok_or_else(|| {
                            Diagnostic::new(
                                "E000205",
                                "a destructuring binding requires a tuple value",
                                Some(value.span),
                            )
                        })?;
                    if elements.len() != names.len() {
                        return Err(Diagnostic::new(
                            "E000205",
                            format!(
                                "tuple has {} element(s), but the binding pattern has {} name(s)",
                                elements.len(),
                                names.len()
                            ),
                            Some(*binding_span),
                        ));
                    }
                    for (name, element) in names.iter().zip(elements) {
                        if !self.declarations.insert(name.clone()) {
                            return Err(Diagnostic::new(
                                "E000203",
                                format!("binding `{name}` is already defined"),
                                Some(*binding_span),
                            ));
                        }
                        let id = self.new_binding_id();
                        let variable = severian_hir::VariableId(id.0);
                        self.mutable_variables.insert(variable);
                        self.names.insert(name.clone(), (id, variable, element));
                        bindings.push(Binding {
                            id,
                            variable,
                            type_id: element,
                            value: self.default_expression(element, *binding_span)?,
                            mutable: true,
                            preserve_error: false,
                            span: *binding_span,
                        });
                        sequence.statements.push(Statement::Binding(id));
                    }
                    guarded_outer_names = Some(self.names.clone());
                    guarded_outer_declarations = Some(self.declarations.clone());
                    guarded_prefix.statements.push(self.lower_destructure_value(
                        names,
                        resolved,
                        true,
                        *binding_span,
                        bindings,
                    )?);
                    for guard in guards {
                        let satisfied = self.condition_expression(&guard.condition)?;
                        let failed = Expression {
                            id: self.next_id(),
                            type_id: satisfied.type_id,
                            kind: ExpressionKind::Unary {
                                operator: UnaryOperator::Not,
                                operand: Box::new(satisfied),
                            },
                            span: guard.span,
                        };
                        let action = match guard.action {
                            severian_ast::LoopGuardAction::Continue => {
                                Statement::Continue { span: guard.span }
                            }
                            severian_ast::LoopGuardAction::Break => {
                                Statement::Break { span: guard.span }
                            }
                        };
                        guarded_prefix.statements.push(Statement::If {
                            condition: failed,
                            then_block: Block {
                                statements: vec![action],
                            },
                            else_block: Block::default(),
                        });
                    }
                }
                let outer_names = guarded_outer_names.unwrap_or_else(|| self.names.clone());
                let outer_declarations =
                    guarded_outer_declarations.unwrap_or_else(|| self.declarations.clone());
                self.loop_depth += 1;
                let lowered = self.block(
                    if guards.is_empty() { body } else { &body[1..] },
                    bindings,
                    result_type,
                );
                self.loop_depth -= 1;
                let mut body = lowered?;
                guarded_prefix.statements.append(&mut body.statements);
                let body = guarded_prefix;
                self.names = outer_names;
                self.declarations = outer_declarations;
                sequence.statements.push(Statement::While {
                    condition,
                    body,
                    span: *span,
                });
                Ok(Statement::Sequence(sequence))
            }
            AstStatement::For {
                binding,
                second_binding,
                iterable,
                initializer,
                body,
                span,
            } => {
                let initializer_statement = initializer
                    .as_ref()
                    .map(|initializer| self.binding(initializer, bindings))
                    .transpose()?
                    .map(Statement::Binding);
                if let Some(values) = static_range_values(iterable) {
                    if second_binding.is_some() {
                        return Err(Diagnostic::new(
                            "E000211",
                            "range iteration provides one loop value",
                            Some(*span),
                        ));
                    }
                    let mut sequence = Block::default();
                    if let Some(initializer) = initializer_statement.clone() {
                        sequence.statements.push(initializer);
                    }
                    for value in values {
                        let mut environment = BTreeMap::from([(binding.clone(), value)]);
                        let (specialized, control) = specialize_loop_body(body, &mut environment);
                        let value = AstExpression {
                            kind: AstExpressionKind::Literal(AstLiteral::Integer(
                                value.to_string(),
                            )),
                            span: iterable.span,
                        };
                        let loop_binding = severian_ast::Binding {
                            name: binding.clone(),
                            annotation: None,
                            value,
                            mutable: !self.declarations.contains(binding),
                            update: self.declarations.contains(binding),
                            preserve_error: false,
                            span: iterable.span,
                        };
                        sequence
                            .statements
                            .push(Statement::Binding(self.binding(&loop_binding, bindings)?));
                        let lowered = self.block(&specialized, bindings, result_type)?;
                        sequence.statements.extend(lowered.statements);
                        if control == StaticLoopControl::Break {
                            break;
                        }
                    }
                    return Ok(Statement::Sequence(sequence));
                }

                let iterable_value = self.expression(iterable, None)?;
                let item_bindings = if let Some(element_type) =
                    self.list_elements.get(&iterable_value.type_id).copied()
                {
                    if second_binding.is_some() {
                        return Err(Diagnostic::new(
                            "E000211",
                            "list iteration provides one loop value",
                            Some(*span),
                        ));
                    }
                    vec![(binding.clone(), element_type, 0)]
                } else if self.set_type == Some(iterable_value.type_id) {
                    if second_binding.is_some() {
                        return Err(Diagnostic::new(
                            "E000211",
                            "set iteration provides one loop value",
                            Some(*span),
                        ));
                    }
                    let element = self.set_element.ok_or_else(|| {
                        Diagnostic::new("E000211", "set element type is unknown", Some(*span))
                    })?;
                    vec![(binding.clone(), element, 0)]
                } else if let Some((key, value)) =
                    self.map_elements.get(&iterable_value.type_id).copied()
                {
                    let Some(value_binding) = second_binding else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "map iteration requires `for key, value in map`",
                            Some(*span),
                        ));
                    };
                    vec![(binding.clone(), key, 0), (value_binding.clone(), value, 1)]
                } else {
                    return Err(Diagnostic::new(
                        "E000211",
                        "for iteration is implemented for lists, maps, and literal ranges",
                        Some(*span),
                    ));
                };
                let iterable_type = iterable_value.type_id;
                let usize_type = self
                    .types
                    .resolve_name("usize")
                    .expect("bootstrap defines usize");
                let bool_type = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");

                let iterable_id = self.new_binding_id();
                bindings.push(Binding {
                    id: iterable_id,
                    variable: severian_hir::VariableId(iterable_id.0),
                    type_id: iterable_type,
                    value: iterable_value,
                    mutable: false,
                    preserve_error: false,
                    span: iterable.span,
                });
                let iterable_reference = Expression {
                    id: self.next_id(),
                    type_id: iterable_type,
                    kind: ExpressionKind::Binding(iterable_id),
                    span: iterable.span,
                };

                let index_id = self.new_binding_id();
                let index_variable = severian_hir::VariableId(index_id.0);
                bindings.push(Binding {
                    id: index_id,
                    variable: index_variable,
                    type_id: usize_type,
                    value: self.integer_expression("0", usize_type, iterable.span),
                    mutable: true,
                    preserve_error: false,
                    span: iterable.span,
                });
                let index_reference = Expression {
                    id: self.next_id(),
                    type_id: usize_type,
                    kind: ExpressionKind::Binding(index_id),
                    span: iterable.span,
                };

                let storage =
                    self.collection_storage_expression(iterable_reference.clone(), 0, *span);
                let storage_type = storage.type_id;
                let length = self.runtime_call(
                    "__sev_list_len",
                    &[storage_type],
                    usize_type,
                    vec![storage],
                    *span,
                );
                let condition = Expression {
                    id: self.next_id(),
                    type_id: bool_type,
                    kind: ExpressionKind::Binary {
                        operator: BinaryOperator::Less,
                        left: Box::new(index_reference.clone()),
                        right: Box::new(length),
                    },
                    span: *span,
                };

                let outer_names = self.names.clone();
                let outer_declarations = self.declarations.clone();
                let mut loop_binding_ids = Vec::with_capacity(item_bindings.len());
                for (name, element_type, storage_index) in item_bindings {
                    let body_iterable_reference = Expression {
                        id: self.next_id(),
                        type_id: iterable_type,
                        kind: ExpressionKind::Binding(iterable_id),
                        span: iterable.span,
                    };
                    let body_index_reference = Expression {
                        id: self.next_id(),
                        type_id: usize_type,
                        kind: ExpressionKind::Binding(index_id),
                        span: iterable.span,
                    };
                    let storage = self.collection_storage_expression(
                        body_iterable_reference,
                        storage_index,
                        *span,
                    );
                    let suffix = self.list_runtime_suffix(element_type, *span)?;
                    let item = self.runtime_call(
                        &format!("__sev_list_get_{suffix}"),
                        &[storage_type, usize_type],
                        element_type,
                        vec![storage, body_index_reference],
                        *span,
                    );
                    let loop_binding_id = self.new_binding_id();
                    let loop_variable = severian_hir::VariableId(loop_binding_id.0);
                    self.mutable_variables.insert(loop_variable);
                    bindings.push(Binding {
                        id: loop_binding_id,
                        variable: loop_variable,
                        type_id: element_type,
                        value: item,
                        mutable: true,
                        preserve_error: false,
                        span: *span,
                    });
                    self.names
                        .insert(name.clone(), (loop_binding_id, loop_variable, element_type));
                    self.declarations.insert(name);
                    loop_binding_ids.push(loop_binding_id);
                }
                self.loop_depth += 1;
                let lowered = self.block(body, bindings, result_type);
                self.loop_depth -= 1;
                let mut loop_body = lowered?;
                self.names = outer_names;
                self.declarations = outer_declarations;
                for loop_binding_id in loop_binding_ids.into_iter().rev() {
                    loop_body
                        .statements
                        .insert(0, Statement::Binding(loop_binding_id));
                }

                let next_index_id = self.new_binding_id();
                let one = self.integer_expression("1", usize_type, *span);
                let increment_index_reference = Expression {
                    id: self.next_id(),
                    type_id: usize_type,
                    kind: ExpressionKind::Binding(index_id),
                    span: iterable.span,
                };
                let next_index = Expression {
                    id: self.next_id(),
                    type_id: usize_type,
                    kind: ExpressionKind::Binary {
                        operator: BinaryOperator::Add,
                        left: Box::new(increment_index_reference),
                        right: Box::new(one),
                    },
                    span: *span,
                };
                bindings.push(Binding {
                    id: next_index_id,
                    variable: index_variable,
                    type_id: usize_type,
                    value: next_index,
                    mutable: true,
                    preserve_error: false,
                    span: *span,
                });
                let increment = Statement::Binding(next_index_id);
                increment_before_continue(&mut loop_body, &increment);
                loop_body.statements.push(increment);

                let mut statements = vec![
                        Statement::Binding(iterable_id),
                        Statement::Binding(index_id),
                        Statement::While {
                            condition,
                            body: loop_body,
                            span: *span,
                        },
                    ];
                if let Some(initializer) = initializer_statement {
                    statements.insert(0, initializer);
                }
                Ok(Statement::Sequence(Block { statements }))
            }
            AstStatement::Break { span } => {
                if self.loop_depth == 0 {
                    return Err(Diagnostic::new(
                        "E000211",
                        "`break` is only valid inside a loop",
                        Some(*span),
                    ));
                }
                Ok(Statement::Break { span: *span })
            }
            AstStatement::Continue { span } => {
                if self.loop_depth == 0 {
                    return Err(Diagnostic::new(
                        "E000211",
                        "`continue` is only valid inside a loop",
                        Some(*span),
                    ));
                }
                Ok(Statement::Continue { span: *span })
            }
            AstStatement::Binding(binding) => {
                Ok(Statement::Binding(self.binding(binding, bindings)?))
            }
            AstStatement::Destructure {
                names,
                value,
                mutable,
                span,
            } => {
                let value = self.expression(value, None)?;
                self.lower_destructure_value(names, value, *mutable, *span, bindings)
            }
            AstStatement::FieldAssignment {
                object,
                field,
                value,
                span,
            } => {
                if field.starts_with('_') {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("field `{field}` cannot be written outside its class"),
                        Some(*span),
                    )
                    .with_help("use a public method to update protected or private state"));
                }
                let AstExpressionKind::Name(object_name) = &object.kind else {
                    return Err(Diagnostic::new(
                        "E000211",
                        "field assignment requires a named object",
                        Some(*span),
                    ));
                };
                let existing = self.names.get(object_name).copied();
                if existing.is_none() {
                    let candidates = self
                        .class_instances_by_type
                        .values()
                        .filter(|instance| instance.fields.iter().any(|known| known.name == *field))
                        .cloned()
                        .collect::<Vec<_>>();
                    let [instance] = candidates.as_slice() else {
                        return Err(Diagnostic::new(
                            "E000201",
                            format!("unknown binding `{object_name}`"),
                            Some(object.span),
                        ));
                    };
                    let mut fields = Vec::with_capacity(instance.fields.len());
                    for declaration in &instance.fields {
                        if declaration.name == *field {
                            fields.push(self.expression(value, Some(declaration.ty))?);
                        } else {
                            fields.push(self.default_expression(declaration.ty, *span)?);
                        }
                    }
                    let id = self.new_binding_id();
                    let variable = severian_hir::VariableId(id.0);
                    self.names
                        .insert(object_name.clone(), (id, variable, instance.ty));
                    self.declarations.insert(object_name.clone());
                    bindings.push(Binding {
                        id,
                        variable,
                        type_id: instance.ty,
                        value: Expression {
                            id: self.next_id(),
                            type_id: instance.ty,
                            kind: ExpressionKind::Aggregate {
                                class: instance.ty,
                                fields,
                            },
                            span: *span,
                        },
                        mutable: true,
                        preserve_error: false,
                        span: *span,
                    });
                    return Ok(Statement::Binding(id));
                }
                let (binding, _, object_type) = existing.unwrap();
                let instance = self
                    .class_instances_by_type
                    .get(&object_type)
                    .cloned()
                    .ok_or_else(|| {
                        Diagnostic::new(
                            "E000211",
                            "field assignment requires a class value",
                            Some(object.span),
                        )
                    })?;
                let Some((index, declaration)) = instance
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, declaration)| declaration.name == *field)
                else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("class `{}` has no field `{field}`", instance.name),
                        Some(*span),
                    ));
                };
                let value = self.expression(value, Some(declaration.ty))?;
                let value = self.validate_field_value(&instance, index, value, *span)?;
                Ok(Statement::FieldSet {
                    binding,
                    field: index as u32,
                    value,
                })
            }
            AstStatement::IndexAssignment {
                object,
                index,
                value,
                span,
            } => {
                let collection = self.expression(object, None)?;
                let unit = self
                    .types
                    .resolve_name("unit")
                    .expect("bootstrap defines unit");
                if let Some(element) = self.pointer_elements.get(&collection.type_id).copied() {
                    if self.unsafe_depth == 0 {
                        return Err(Diagnostic::new(
                            "E000219",
                            "raw pointer mutation requires an `unsafe` scope",
                            Some(*span),
                        ));
                    }
                    let index = self.expression(index, None)?;
                    let index_name = self
                        .types
                        .definition(index.type_id)
                        .map(|definition| definition.name.as_str());
                    if !matches!(index_name, Some("int" | "i64" | "usize")) {
                        return Err(Diagnostic::new(
                            "E000211",
                            "pointer indices must be integers",
                            Some(index.span),
                        ));
                    }
                    let value = self.expression(value, Some(element))?;
                    let suffix = self.list_runtime_suffix(element, *span)?;
                    return Ok(Statement::Expression(self.runtime_call(
                        &format!("__sev_pointer_set_{suffix}"),
                        &[collection.type_id, index.type_id, element],
                        unit,
                        vec![collection, index, value],
                        *span,
                    )));
                }
                if let Some(element) = self.list_elements.get(&collection.type_id).copied() {
                    let index = self.expression(index, None)?;
                    if !self.integer_primitive(index.type_id) {
                        return Err(Diagnostic::new(
                            "E000211",
                            "list indices must be integers",
                            Some(index.span),
                        ));
                    }
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let index = self.coerce(index, integer, true)?;
                    let value = self.expression(value, Some(element))?;
                    let storage = self.list_storage_expression(collection, *span);
                    let storage_type = storage.type_id;
                    let suffix = self.list_runtime_suffix(element, *span)?;
                    return Ok(Statement::Expression(self.runtime_call(
                        &format!("__sev_list_set_{suffix}"),
                        &[storage_type, index.type_id, element],
                        unit,
                        vec![storage, index, value],
                        *span,
                    )));
                }
                if let Some((key_type, value_type)) =
                    self.map_elements.get(&collection.type_id).copied()
                {
                    let key = self.expression(index, Some(key_type))?;
                    let value = self.expression(value, Some(value_type))?;
                    let keys =
                        self.collection_storage_expression(collection.clone(), 0, *span);
                    let values = self.collection_storage_expression(collection, 1, *span);
                    let storage_type = keys.type_id;
                    let key_suffix = self.list_runtime_suffix(key_type, *span)?;
                    let value_suffix = self.list_runtime_suffix(value_type, *span)?;
                    return Ok(Statement::Expression(self.runtime_call(
                        &format!("__sev_map_set_{key_suffix}_{value_suffix}"),
                        &[storage_type, storage_type, key_type, value_type],
                        unit,
                        vec![keys, values, key, value],
                        *span,
                    )));
                }
                Err(Diagnostic::new(
                    "E000211",
                    "indexed assignment is implemented for lists and maps",
                    Some(*span),
                ))
            }
            AstStatement::Expression(expression) => {
                if let AstExpressionKind::Throw { error } = &expression.kind {
                    if let Some(fallible) = self.fallible_types.get(&result_type).copied() {
                        let error = self.expression(error, Some(fallible.error))?;
                        let result = self.fallible_error_expression(
                            result_type,
                            fallible,
                            error,
                            expression.span,
                        )?;
                        return Ok(Statement::Return(Some(result)));
                    }
                    if self.is_error_type(result_type) {
                        return Ok(Statement::Return(Some(
                            self.expression(error, Some(result_type))?,
                        )));
                    }
                }
                if let AstExpressionKind::Mock { cases, fallback } = &expression.kind {
                    let Some(name) = cases.first().and_then(|case| match &case.call.kind {
                        AstExpressionKind::Call { callee, .. } => callable_path(callee),
                        _ => None,
                    }) else {
                        return Err(Diagnostic::new(
                            "E000217",
                            "mock cases must contain function calls",
                            Some(expression.span),
                        ));
                    };
                    if cases.iter().any(|case| {
                        !matches!(&case.call.kind,
                            AstExpressionKind::Call { callee, .. }
                                if callable_path(callee).as_deref() == Some(name.as_str()))
                    }) {
                        return Err(Diagnostic::new(
                            "E000217",
                            "all cases in one mock must target the same function",
                            Some(expression.span),
                        ));
                    }
                    self.mocks.insert(
                        name,
                        ActiveMock {
                            cases: cases.clone(),
                            fallback: fallback.as_ref().clone(),
                        },
                    );
                    return Ok(Statement::Sequence(Block::default()));
                }
                if let AstExpressionKind::Call { callee, arguments } = &expression.kind {
                    if callable_path(callee).as_deref() == Some("expect") {
                        let [argument] = arguments.as_slice() else {
                            return Err(Diagnostic::new(
                                "E000217",
                                "`expect` requires exactly one condition",
                                Some(expression.span),
                            ));
                        };
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        let condition = self.expression(&argument.value, Some(boolean))?;
                        let string = self
                            .types
                            .resolve_name("string")
                            .expect("bootstrap defines string");
                        let message = self.string_expression(
                            format!(
                                "expectation failed in {}",
                                self.active_function_name
                                    .as_deref()
                                    .unwrap_or("<test>")
                            ),
                            expression.span,
                        );
                        let unit = self
                            .types
                            .resolve_name("unit")
                            .expect("bootstrap defines unit");
                        return Ok(Statement::Expression(self.runtime_call(
                            "__sev_expect",
                            &[boolean, string],
                            unit,
                            vec![condition, message],
                            expression.span,
                        )));
                    }
                    if callable_path(callee).as_deref() == Some("throws") {
                        let [argument] = arguments.as_slice() else {
                            return Err(Diagnostic::new(
                                "E000217",
                                "`throws` requires exactly one expression",
                                Some(expression.span),
                            ));
                        };
                        let boolean_type = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        if let Some(expected_error) = &argument.expected_error {
                            let (expected_name, is_error_value) = match &expected_error.kind {
                                AstExpressionKind::Name(name) => (name.clone(), false),
                                AstExpressionKind::Call { callee, .. } => {
                                    let Some(name) = callable_path(callee) else {
                                        return Err(Diagnostic::new(
                                            "E000217",
                                            "`throws` expects an error type or error value after `->`",
                                            Some(expected_error.span),
                                        ));
                                    };
                                    (name, true)
                                }
                                _ => {
                                    return Err(Diagnostic::new(
                                        "E000217",
                                        "`throws` expects an error type or error value after `->`",
                                        Some(expected_error.span),
                                    ));
                                }
                            };
                            let expected_type = self
                                .class_instances
                                .get(&(expected_name.clone(), Vec::new()))
                                .map(|instance| instance.ty)
                                .or_else(|| self.types.resolve_name(&expected_name))
                                .filter(|ty| self.is_error_type(*ty));
                            if is_error_value && expected_type.is_none() {
                                return Err(Diagnostic::new(
                                    "E000217",
                                    "`throws` expects an error type or error value after `->`",
                                    Some(expected_error.span),
                                ));
                            }
                            if let (Some(expected_type), Some(actual_type)) =
                                (expected_type, self.call_thrown_error(&argument.value))
                            {
                                let core_error = self
                                    .types
                                    .resolve_name("Error")
                                    .expect("bootstrap defines Error");
                                if expected_type != core_error && actual_type != expected_type {
                                    let actual_name = self
                                        .types
                                        .definition(actual_type)
                                        .map(|definition| definition.name.clone())
                                        .or_else(|| {
                                            self.class_instances_by_type
                                                .get(&actual_type)
                                                .map(|instance| instance.name.clone())
                                        })
                                        .unwrap_or_else(|| format!("type#{}", actual_type.0));
                                    let condition = Expression {
                                        id: self.next_id(),
                                        type_id: boolean_type,
                                        kind: ExpressionKind::Literal(LiteralValue::Boolean(false)),
                                        span: expression.span,
                                    };
                                    let message = self.string_expression(
                                        format!("expected `{expected_name}`, got `{actual_name}`"),
                                        expression.span,
                                    );
                                    let string = message.type_id;
                                    let unit = self
                                        .types
                                        .resolve_name("unit")
                                        .expect("bootstrap defines unit");
                                    return Ok(Statement::Expression(self.runtime_call(
                                        "__sev_expect",
                                        &[boolean_type, string],
                                        unit,
                                        vec![condition, message],
                                        expression.span,
                                    )));
                                }
                            }
                        }
                        let action = match self.class_method_update(&argument.value)? {
                            Some(statement) => statement,
                            None => Statement::Expression(self.expression(&argument.value, None)?),
                        };
                        return Ok(Statement::ExpectThrow {
                            body: Block {
                                statements: vec![action],
                            },
                            boolean_type,
                            span: expression.span,
                        });
                    }
                }
                let dropped = explicit_drop_receiver(expression).map(str::to_owned);
                let lowered = match self.class_method_update(expression)? {
                    Some(update) => update,
                    None if dropped.is_some() => {
                        return Err(Diagnostic::new(
                            "E000211",
                            "`drop` requires a class value with a unit `drop()` method",
                            Some(expression.span),
                        ));
                    }
                    None => Statement::Expression(self.expression(expression, None)?),
                };
                if let Some(receiver) = dropped {
                    self.names.remove(&receiver);
                    self.declarations.remove(&receiver);
                }
                Ok(lowered)
            }
            AstStatement::Return { value, span } => {
                let unit = self
                    .types
                    .resolve_name("unit")
                    .expect("bootstrap defines unit");
                let value = match value {
                    Some(value) if result_type == unit => {
                        return Err(Diagnostic::new(
                            "E000210",
                            "a unit function cannot return a value",
                            Some(*span),
                        ))
                    }
                    Some(value) => {
                        if let Some(fallible) = self.fallible_types.get(&result_type).copied() {
                            if let AstExpressionKind::Call { callee, arguments } = &value.kind {
                                if callable_path(callee).as_deref() == Some("error") {
                                    let [argument] = arguments.as_slice() else {
                                        return Err(Diagnostic::new(
                                            "E000206",
                                            "`error` expects exactly one error value",
                                            Some(value.span),
                                        ));
                                    };
                                    let error =
                                        self.expression(&argument.value, Some(fallible.error))?;
                                    return Ok(Statement::Return(Some(
                                        self.fallible_error_expression(
                                            result_type,
                                            fallible,
                                            error,
                                            *span,
                                        )?,
                                    )));
                                }
                            }
                            let value = self.expression(value, Some(fallible.success))?;
                            Some(self.fallible_success_expression(
                                result_type,
                                fallible,
                                value,
                                *span,
                            )?)
                        } else {
                            Some(self.expression(value, Some(result_type))?)
                        }
                    }
                    None if result_type != unit => {
                        return Err(Diagnostic::new(
                            "E000210",
                            "this function must return its declared result",
                            Some(*span),
                        ))
                    }
                    None => None,
                };
                Ok(Statement::Return(value))
            }
            AstStatement::Assert {
                condition,
                message,
                span,
            } => {
                let boolean = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");
                let string = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                Ok(Statement::Assert {
                    condition: self.expression(condition, Some(boolean))?,
                    message: message
                        .as_ref()
                        .map(|message| self.expression(message, Some(string)))
                        .transpose()?,
                    span: *span,
                    condition_span: condition.span,
                })
            }
            AstStatement::If {
                condition: condition_ast,
                then_block,
                else_block,
                ..
            } => {
                let condition = self.condition_expression(condition_ast)?;
                let outer_names = self.names.clone();
                let outer_declarations = self.declarations.clone();
                let outer_substitutions = self.value_substitutions.clone();
                if let AstExpressionKind::Binary {
                    operator: AstBinaryOperator::Identity,
                    left,
                    right,
                } = &condition_ast.kind
                {
                    if let (AstExpressionKind::Name(binding_name), AstExpressionKind::Name(type_name)) =
                        (&left.kind, &right.kind)
                    {
                        if let Some((binding, _, binding_type)) = self.names.get(binding_name).copied()
                        {
                            if let Some(fallible) = self.fallible_types.get(&binding_type).copied() {
                                let target = self
                                    .class_instances
                                    .get(&(type_name.clone(), Vec::new()))
                                    .map(|instance| instance.ty)
                                    .or_else(|| self.types.resolve_name(type_name));
                                let field = if target == Some(fallible.success) {
                                    Some((1, fallible.success))
                                } else if target == Some(fallible.error) {
                                    Some((2, fallible.error))
                                } else {
                                    None
                                };
                                if let Some((index, type_id)) = field {
                                    let binding_id = self.next_id();
                                    let field_id = self.next_id();
                                    let narrowed = Expression {
                                        id: field_id,
                                        type_id,
                                        kind: ExpressionKind::Field {
                                            object: Box::new(Expression {
                                                id: binding_id,
                                                type_id: binding_type,
                                                kind: ExpressionKind::Binding(binding),
                                                span: left.span,
                                            }),
                                            index,
                                        },
                                        span: left.span,
                                    };
                                    self.value_substitutions.insert(
                                        binding_name.clone(),
                                        narrowed,
                                    );
                                }
                            }
                        }
                    }
                }
                let then_block = self.block(then_block, bindings, result_type)?;
                self.names.clone_from(&outer_names);
                self.declarations.clone_from(&outer_declarations);
                self.value_substitutions.clone_from(&outer_substitutions);
                let else_block = self.block(else_block, bindings, result_type)?;
                self.names = outer_names;
                self.declarations = outer_declarations;
                self.value_substitutions = outer_substitutions;
                Ok(Statement::If {
                    condition,
                    then_block,
                    else_block,
                })
            }
            AstStatement::Match {
                subject,
                cases,
                span,
            } => self.match_statement(subject, cases, *span, bindings, result_type),
            AstStatement::Select {
                limit,
                cases,
                error_body,
                span,
            } => self.select_statement(limit, cases, error_body, *span, bindings, result_type),
        }
    }

    fn select_statement(
        &mut self,
        limit: &AstExpression,
        cases: &[severian_ast::SelectCase],
        error_body: &[AstStatement],
        span: severian_source::Span,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        let usize_type = self
            .types
            .resolve_name("usize")
            .expect("bootstrap defines usize");
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let unit = self
            .types
            .resolve_name("unit")
            .expect("bootstrap defines unit");
        let limit = self.expression(limit, Some(usize_type))?;

        // `limit` counts successful receives, independently of bindings used by
        // the program inside a case. Keeping this counter compiler-owned makes
        // select useful without requiring a magic source-level variable.
        let counter = self.new_binding_id();
        let counter_variable = severian_hir::VariableId(counter.0);
        self.mutable_variables.insert(counter_variable);
        bindings.push(Binding {
            id: counter,
            variable: counter_variable,
            type_id: usize_type,
            value: self.integer_expression("0", usize_type, span),
            mutable: true,
            preserve_error: false,
            span,
        });
        let condition = Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Binary {
                operator: BinaryOperator::Less,
                left: Box::new(Expression {
                    id: self.next_id(),
                    type_id: usize_type,
                    kind: ExpressionKind::Binding(counter),
                    span,
                }),
                right: Box::new(limit),
            },
            span,
        };

        let outer_names = self.names.clone();
        let outer_declarations = self.declarations.clone();
        let mut lowered_cases = Vec::with_capacity(cases.len());
        let mut channel_storages = Vec::with_capacity(cases.len());
        for case in cases {
            self.names.clone_from(&outer_names);
            self.declarations.clone_from(&outer_declarations);

            let channel = self.expression(&case.channel, None)?;
            let Some(element) = self.channel_elements.get(&channel.type_id).copied() else {
                self.names = outer_names;
                self.declarations = outer_declarations;
                return Err(Diagnostic::new(
                    "E000205",
                    "a select case requires a channel after `from`",
                    Some(case.channel.span),
                ));
            };
            let storage = self.channel_storage_expression(channel, case.channel.span);
            let storage_type = storage.type_id;
            channel_storages.push(storage.clone());
            let ready = self.runtime_call(
                "__sev_channel_claim",
                &[storage_type],
                boolean,
                vec![storage.clone()],
                case.span,
            );
            let suffix = self.list_runtime_suffix(element, case.span)?;
            let value = self.runtime_call(
                &format!("__sev_channel_recv_{suffix}"),
                &[storage_type],
                element,
                vec![storage],
                case.span,
            );

            let case_binding = self.new_binding_id();
            let case_variable = severian_hir::VariableId(case_binding.0);
            if !self.declarations.insert(case.binding.clone()) {
                self.names = outer_names;
                self.declarations = outer_declarations;
                return Err(Diagnostic::new(
                    "E000203",
                    format!("binding `{}` is already defined", case.binding),
                    Some(case.span),
                ));
            }
            self.names
                .insert(case.binding.clone(), (case_binding, case_variable, element));
            bindings.push(Binding {
                id: case_binding,
                variable: case_variable,
                type_id: element,
                value,
                mutable: false,
                preserve_error: false,
                span: case.span,
            });

            self.loop_depth += 1;
            let lowered_body = self.block(&case.body, bindings, result_type);
            self.loop_depth -= 1;
            let mut body = lowered_body?;
            body.statements.insert(0, Statement::Binding(case_binding));

            let next_counter = self.new_binding_id();
            let increment = Expression {
                id: self.next_id(),
                type_id: usize_type,
                kind: ExpressionKind::Binary {
                    operator: BinaryOperator::Add,
                    left: Box::new(Expression {
                        id: self.next_id(),
                        type_id: usize_type,
                        kind: ExpressionKind::Binding(counter),
                        span: case.span,
                    }),
                    right: Box::new(self.integer_expression("1", usize_type, case.span)),
                },
                span: case.span,
            };
            bindings.push(Binding {
                id: next_counter,
                variable: counter_variable,
                type_id: usize_type,
                value: increment,
                mutable: true,
                preserve_error: false,
                span: case.span,
            });
            let increment = Statement::Binding(next_counter);
            increment_before_continue(&mut body, &increment);
            body.statements.push(increment);
            lowered_cases.push((ready, body));
        }

        self.names.clone_from(&outer_names);
        self.declarations.clone_from(&outer_declarations);
        let yield_call = self.runtime_call("__sev_channel_yield", &[], unit, Vec::new(), span);
        let mut otherwise = Block {
            statements: vec![Statement::Expression(yield_call)],
        };
        if !error_body.is_empty() {
            let closed = channel_storages
                .into_iter()
                .map(|storage| {
                    self.runtime_call(
                        "__sev_channel_is_closed",
                        &[storage.type_id],
                        boolean,
                        vec![storage],
                        span,
                    )
                })
                .collect::<Vec<_>>();
            let mut closed = closed.into_iter();
            let mut all_closed = closed.next().expect("select has at least one case");
            for next in closed {
                all_closed = Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Binary {
                        operator: BinaryOperator::And,
                        left: Box::new(all_closed),
                        right: Box::new(next),
                    },
                    span,
                };
            }
            let error_body = self.block(error_body, bindings, result_type)?;
            otherwise = Block {
                statements: vec![Statement::If {
                    condition: all_closed,
                    then_block: error_body,
                    else_block: otherwise,
                }],
            };
        }
        for (ready, body) in lowered_cases.into_iter().rev() {
            otherwise = Block {
                statements: vec![Statement::If {
                    condition: ready,
                    then_block: body,
                    else_block: otherwise,
                }],
            };
        }
        self.names = outer_names;
        self.declarations = outer_declarations;

        Ok(Statement::Sequence(Block {
            statements: vec![
                Statement::Binding(counter),
                Statement::While {
                    condition,
                    body: otherwise,
                    span,
                },
            ],
        }))
    }

    fn match_statement(
        &mut self,
        subject: &AstExpression,
        cases: &[severian_ast::MatchCase],
        span: severian_source::Span,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        let subject = self.expression(subject, None)?;
        let subject_type = subject.type_id;
        if self
            .enums
            .values()
            .any(|instance| instance.ty == subject_type)
        {
            return self.enum_match_statement(subject, cases, span, bindings, result_type);
        }
        let outer_names = self.names.clone();
        let outer_declarations = self.declarations.clone();
        let mut seen_types = BTreeSet::new();
        let mut catch_all = false;
        let mut arms = Vec::new();
        for case in cases {
            if catch_all {
                return Err(Diagnostic::new(
                    "E000214",
                    "this case is unreachable because a previous case matches every remaining value",
                    Some(case.span),
                ));
            }
            let type_id = case
                .annotation
                .as_ref()
                .map(|annotation| resolve_type_annotation(self.types, annotation))
                .transpose()?;
            if let Some(type_id) = type_id {
                if !seen_types.insert(type_id) {
                    return Err(Diagnostic::new(
                        "E000214",
                        "this match type is handled more than once",
                        Some(case.span),
                    ));
                }
                if type_id != subject_type {
                    return Err(Diagnostic::new(
                        "E000215",
                        "case type cannot occur in this non-union matched expression",
                        Some(case.span),
                    ));
                }
            }
            catch_all = true;

            self.names.clone_from(&outer_names);
            self.declarations.clone_from(&outer_declarations);
            let binding = if let Some(name) = &case.binding {
                let id = self.new_binding_id();
                let binding_type = type_id.unwrap_or(subject_type);
                let variable = severian_hir::VariableId(id.0);
                self.names
                    .insert(name.clone(), (id, variable, binding_type));
                self.declarations.insert(name.clone());
                bindings.push(Binding {
                    id,
                    variable,
                    type_id: binding_type,
                    value: subject.clone(),
                    mutable: false,
                    preserve_error: true,
                    span: case.span,
                });
                Some(id)
            } else {
                None
            };
            arms.push(severian_hir::MatchArm {
                binding,
                type_id,
                body: self.block(&case.body, bindings, result_type)?,
            });
        }
        self.names = outer_names;
        self.declarations = outer_declarations;
        if !catch_all && !seen_types.contains(&subject_type) {
            return Err(Diagnostic::new(
                "E000216",
                "match is not exhaustive; add a compatible typed case or `case _:`",
                Some(span),
            ));
        }
        Ok(Statement::Match { subject, arms })
    }

    fn enum_match_statement(
        &mut self,
        subject: Expression,
        cases: &[severian_ast::MatchCase],
        span: severian_source::Span,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        let instance = self
            .enums
            .values()
            .find(|instance| instance.ty == subject.type_id)
            .cloned()
            .expect("caller selected an enum subject");
        let integer = self.tag_type();
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let outer_names = self.names.clone();
        let outer_declarations = self.declarations.clone();
        let outer_substitutions = self.value_substitutions.clone();
        let mut handled = BTreeSet::new();
        let mut lowered = Vec::new();
        let mut returned = Some(Vec::new());
        for case in cases {
            self.names.clone_from(&outer_names);
            self.declarations.clone_from(&outer_declarations);
            self.value_substitutions.clone_from(&outer_substitutions);
            let pattern = case.binding.as_deref().unwrap_or("_");
            let variant = instance.variants.iter().enumerate().find(|(_, variant)| {
                pattern == variant.name || pattern == format!("{}.{}", instance.name, variant.name)
            });
            let (condition, selected) = if let Some((ordinal, variant)) = variant {
                if !handled.insert(ordinal) {
                    return Err(Diagnostic::new(
                        "E000214",
                        format!("enum variant `{}` is handled more than once", variant.name),
                        Some(case.span),
                    ));
                }
                for (payload_ordinal, payload) in variant.fields.iter().enumerate() {
                    let index = enum_payload_index(&instance.variants, ordinal, payload_ordinal);
                    let field = &instance.fields[index];
                    let id = self.next_id();
                    self.value_substitutions.insert(
                        payload.name.clone(),
                        Expression {
                            id,
                            type_id: field.ty,
                            kind: ExpressionKind::Field {
                                object: Box::new(subject.clone()),
                                index: index as u32,
                            },
                            span: case.span,
                        },
                    );
                }
                let tag = Expression {
                    id: self.next_id(),
                    type_id: integer,
                    kind: ExpressionKind::Field {
                        object: Box::new(subject.clone()),
                        index: 0,
                    },
                    span: case.span,
                };
                let ordinal = self.integer_expression(&ordinal.to_string(), integer, case.span);
                (
                    Some(Expression {
                        id: self.next_id(),
                        type_id: boolean,
                        kind: ExpressionKind::Binary {
                            operator: BinaryOperator::Equal,
                            left: Box::new(tag),
                            right: Box::new(ordinal),
                        },
                        span: case.span,
                    }),
                    true,
                )
            } else if pattern == "_" {
                (None, true)
            } else {
                return Err(Diagnostic::new(
                    "E000215",
                    format!("`{pattern}` is not a variant of `{}`", instance.name),
                    Some(case.span),
                ));
            };
            debug_assert!(selected);
            if let Some(values) = &mut returned {
                if let [AstStatement::Return {
                    value: Some(value), ..
                }] = case.body.as_slice()
                {
                    values.push((
                        condition.clone(),
                        self.expression(value, Some(result_type))?,
                    ));
                } else {
                    returned = None;
                }
            }
            lowered.push((condition, self.block(&case.body, bindings, result_type)?));
        }
        self.names = outer_names;
        self.declarations = outer_declarations;
        self.value_substitutions = outer_substitutions;
        if handled.len() != instance.variants.len()
            && !lowered.iter().any(|(condition, _)| condition.is_none())
        {
            return Err(Diagnostic::new(
                "E000216",
                format!("match does not cover every `{}` variant", instance.name),
                Some(span),
            ));
        }
        if let Some(mut values) = returned {
            if let Ok(suffix) = self.select_runtime_suffix(result_type, span) {
                let Some((_, mut selected)) = values.pop() else {
                    return Err(Diagnostic::new(
                        "E000216",
                        "an enum match requires at least one arm",
                        Some(span),
                    ));
                };
                while let Some((condition, value)) = values.pop() {
                    let Some(condition) = condition else {
                        selected = value;
                        continue;
                    };
                    selected = self.runtime_call(
                        &format!("__sev_select_{suffix}"),
                        &[boolean, result_type, result_type],
                        result_type,
                        vec![condition, value, selected],
                        span,
                    );
                }
                return Ok(Statement::Return(Some(selected)));
            }
        }
        let mut lowered = lowered.into_iter().rev();
        let Some((_last_condition, last_body)) = lowered.next() else {
            return Err(Diagnostic::new(
                "E000216",
                "an enum match requires at least one arm",
                Some(span),
            ));
        };
        let mut tail = last_body;
        let mut nested = false;
        for (condition, body) in lowered {
            if let Some(condition) = condition {
                nested = true;
                tail = Block {
                    statements: vec![Statement::If {
                        condition,
                        then_block: body,
                        else_block: tail,
                    }],
                };
            } else {
                tail = body;
            }
        }
        if nested && tail.statements.len() == 1 {
            return Ok(tail.statements.into_iter().next().unwrap());
        }
        let condition = Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
            span,
        };
        Ok(Statement::If {
            condition,
            then_block: tail.clone(),
            else_block: tail,
        })
    }

    fn block(
        &mut self,
        statements: &[AstStatement],
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Block, Diagnostic> {
        let mut block = Block::default();
        let mut deferred = Vec::new();
        for statement in statements {
            if let AstStatement::Defer { expression, .. } = statement {
                deferred.push(Statement::Expression(self.expression(expression, None)?));
                continue;
            }
            let mut statement = statement.clone();
            let preludes =
                self.lower_statement_comprehensions(&mut statement, bindings, result_type)?;
            block.statements.extend(preludes);
            block
                .statements
                .push(self.statement(&statement, bindings, result_type)?);
        }
        deferred.reverse();
        if !deferred.is_empty() {
            insert_before_returns(&mut block, &deferred);
            if block_flow(statements) == ControlFlow::FallsThrough {
                block.statements.extend(deferred);
            }
        }
        Ok(block)
    }

    fn lower_statement_comprehensions(
        &mut self,
        statement: &mut AstStatement,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Vec<Statement>, Diagnostic> {
        let mut preludes = Vec::new();
        match statement {
            AstStatement::Binding(binding) => self.lower_expression_comprehensions(
                &mut binding.value,
                bindings,
                result_type,
                &mut preludes,
            )?,
            AstStatement::Destructure { value, .. }
            | AstStatement::Expression(value)
            | AstStatement::Defer {
                expression: value, ..
            }
            | AstStatement::FallibleElse { value, .. }
            | AstStatement::Return {
                value: Some(value), ..
            } => self
                .lower_expression_comprehensions(value, bindings, result_type, &mut preludes)?,
            AstStatement::Assert {
                condition, message, ..
            } => {
                self.lower_expression_comprehensions(
                    condition,
                    bindings,
                    result_type,
                    &mut preludes,
                )?;
                if let Some(message) = message {
                    self.lower_expression_comprehensions(
                        message,
                        bindings,
                        result_type,
                        &mut preludes,
                    )?;
                }
            }
            AstStatement::IndexAssignment {
                object,
                index,
                value,
                ..
            } => {
                for expression in [object, index, value] {
                    self.lower_expression_comprehensions(
                        expression,
                        bindings,
                        result_type,
                        &mut preludes,
                    )?;
                }
            }
            AstStatement::FieldAssignment { object, value, .. } => {
                self.lower_expression_comprehensions(
                    object,
                    bindings,
                    result_type,
                    &mut preludes,
                )?;
                self.lower_expression_comprehensions(
                    value,
                    bindings,
                    result_type,
                    &mut preludes,
                )?;
            }
            _ => {}
        }
        Ok(preludes)
    }

    fn lower_expression_comprehensions(
        &mut self,
        expression: &mut AstExpression,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
        preludes: &mut Vec<Statement>,
    ) -> Result<(), Diagnostic> {
        if matches!(
            expression.kind,
            AstExpressionKind::ListComprehension { .. }
                | AstExpressionKind::SetComprehension { .. }
                | AstExpressionKind::MapComprehension { .. }
        ) {
            let (name, mut statements) =
                self.lower_comprehension(expression.clone(), bindings, result_type)?;
            preludes.append(&mut statements);
            expression.kind = AstExpressionKind::Name(name);
            return Ok(());
        }
        match &mut expression.kind {
            AstExpressionKind::List(values)
            | AstExpressionKind::Set(values)
            | AstExpressionKind::Tuple(values) => {
                for value in values {
                    self.lower_expression_comprehensions(
                        value,
                        bindings,
                        result_type,
                        preludes,
                    )?;
                }
            }
            AstExpressionKind::Map(entries) => {
                for entry in entries {
                    self.lower_expression_comprehensions(
                        &mut entry.key,
                        bindings,
                        result_type,
                        preludes,
                    )?;
                    self.lower_expression_comprehensions(
                        &mut entry.value,
                        bindings,
                        result_type,
                        preludes,
                    )?;
                }
            }
            AstExpressionKind::Member { object, .. }
            | AstExpressionKind::TypeApplication { callee: object, .. }
            | AstExpressionKind::Async {
                expression: object, ..
            }
            | AstExpressionKind::Await { expression: object }
            | AstExpressionKind::Throw { error: object }
            | AstExpressionKind::Unary {
                operand: object, ..
            } => self.lower_expression_comprehensions(
                object,
                bindings,
                result_type,
                preludes,
            )?,
            AstExpressionKind::Index { object, index } => {
                self.lower_expression_comprehensions(
                    object,
                    bindings,
                    result_type,
                    preludes,
                )?;
                self.lower_expression_comprehensions(
                    index,
                    bindings,
                    result_type,
                    preludes,
                )?;
            }
            AstExpressionKind::Slice {
                object,
                start,
                end,
                step,
                ..
            } => {
                self.lower_expression_comprehensions(
                    object,
                    bindings,
                    result_type,
                    preludes,
                )?;
                for bound in [start, end, step].into_iter().flatten() {
                    self.lower_expression_comprehensions(
                        bound,
                        bindings,
                        result_type,
                        preludes,
                    )?;
                }
            }
            AstExpressionKind::Call { callee, arguments } => {
                self.lower_expression_comprehensions(
                    callee,
                    bindings,
                    result_type,
                    preludes,
                )?;
                for argument in arguments {
                    self.lower_expression_comprehensions(
                        &mut argument.value,
                        bindings,
                        result_type,
                        preludes,
                    )?;
                }
            }
            AstExpressionKind::Conditional {
                value,
                condition,
                fallback,
            } => {
                self.lower_expression_comprehensions(
                    value,
                    bindings,
                    result_type,
                    preludes,
                )?;
                self.lower_expression_comprehensions(
                    condition,
                    bindings,
                    result_type,
                    preludes,
                )?;
                self.lower_expression_comprehensions(
                    fallback,
                    bindings,
                    result_type,
                    preludes,
                )?;
            }
            AstExpressionKind::Fallback { value, fallback }
            | AstExpressionKind::Binary {
                left: value,
                right: fallback,
                ..
            } => {
                self.lower_expression_comprehensions(
                    value,
                    bindings,
                    result_type,
                    preludes,
                )?;
                self.lower_expression_comprehensions(
                    fallback,
                    bindings,
                    result_type,
                    preludes,
                )?;
            }
            AstExpressionKind::Lambda { body, .. } => self.lower_expression_comprehensions(
                body,
                bindings,
                result_type,
                preludes,
            )?,
            AstExpressionKind::Mock { .. }
            | AstExpressionKind::Literal(_)
            | AstExpressionKind::Name(_)
            | AstExpressionKind::ListComprehension { .. }
            | AstExpressionKind::SetComprehension { .. }
            | AstExpressionKind::MapComprehension { .. } => {}
        }
        Ok(())
    }

    fn lower_comprehension(
        &mut self,
        comprehension: AstExpression,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<(String, Vec<Statement>), Diagnostic> {
        let span = comprehension.span;
        let (kind, first, second, clauses) = match comprehension.kind {
            AstExpressionKind::ListComprehension { value, clauses } => {
                ("list", *value, None, clauses)
            }
            AstExpressionKind::SetComprehension { value, clauses } => {
                ("set", *value, None, clauses)
            }
            AstExpressionKind::MapComprehension {
                key,
                value,
                clauses,
            } => ("map", *key, Some(*value), clauses),
            _ => unreachable!("comprehension lowering requires a comprehension"),
        };
        if clauses.is_empty() {
            return Err(Diagnostic::new(
                "E000211",
                "a comprehension requires at least one `for` clause",
                Some(span),
            ));
        }

        let outer_names = self.names.clone();
        let outer_declarations = self.declarations.clone();
        let inferred = (|| {
            for clause in &clauses {
                let iterable = self.expression(&clause.iterable, None)?;
                let element_types = if let Some(element) =
                    self.list_elements.get(&iterable.type_id).copied()
                {
                    vec![element]
                } else if let Some((key, value)) =
                    self.map_elements.get(&iterable.type_id).copied()
                {
                    vec![key, value]
                } else {
                    return Err(Diagnostic::new(
                        "E000211",
                        "comprehension iteration requires a list or map",
                        Some(clause.iterable.span),
                    ));
                };
                if element_types.len() != clause.bindings.len() {
                    return Err(Diagnostic::new(
                        "E000211",
                        "comprehension binding count does not match its iterable",
                        Some(clause.span),
                    ));
                }
                for (name, ty) in clause.bindings.iter().zip(element_types) {
                    let id = self.new_binding_id();
                    self.names
                        .insert(name.clone(), (id, severian_hir::VariableId(id.0), ty));
                    self.declarations.insert(name.clone());
                }
            }
            let first_type = self.expression(&first, None)?.type_id;
            let second_type = second
                .as_ref()
                .map(|value| self.expression(value, None).map(|value| value.type_id))
                .transpose()?;
            Ok::<_, Diagnostic>((first_type, second_type))
        })();
        self.names = outer_names;
        self.declarations = outer_declarations;
        let (first_type, second_type) = inferred?;

        let name = format!("__comprehension_{}", self.next_comprehension);
        self.next_comprehension += 1;
        let (collection_type, initial) = match kind {
            "list" => {
                let ty = self.instantiate_list_type(first_type);
                (ty, self.empty_list_expression(ty, span)?)
            }
            "set" => {
                let value = self.empty_set_expression(Some(first_type), span)?;
                (value.type_id, value)
            }
            "map" => {
                let value_type = second_type.expect("map comprehension has a value");
                let ty = self.instantiate_map_type(first_type, value_type);
                let storage_type = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines pointer-backed string");
                let keys =
                    self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), span);
                let values =
                    self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), span);
                (
                    ty,
                    Expression {
                        id: self.next_id(),
                        type_id: ty,
                        kind: ExpressionKind::Aggregate {
                            class: ty,
                            fields: vec![keys, values],
                        },
                        span,
                    },
                )
            }
            _ => unreachable!(),
        };
        let id = self.new_binding_id();
        let variable = severian_hir::VariableId(id.0);
        self.names
            .insert(name.clone(), (id, variable, collection_type));
        self.declarations.insert(name.clone());
        self.mutable_variables.insert(variable);
        bindings.push(Binding {
            id,
            variable,
            type_id: collection_type,
            value: initial,
            mutable: true,
            preserve_error: false,
            span,
        });

        let result_name = AstExpression {
            kind: AstExpressionKind::Name(name.clone()),
            span,
        };
        let mut body = match kind {
            "list" | "set" => {
                let operation = if kind == "list" { "append" } else { "add" };
                vec![AstStatement::Expression(AstExpression {
                    kind: AstExpressionKind::Call {
                        callee: Box::new(AstExpression {
                            kind: AstExpressionKind::Member {
                                object: Box::new(result_name.clone()),
                                name: operation.into(),
                            },
                            span,
                        }),
                        arguments: vec![severian_ast::CallArgument {
                            name: None,
                            value: first,
                            expected_error: None,
                            span,
                        }],
                    },
                    span,
                })]
            }
            "map" => vec![AstStatement::IndexAssignment {
                object: result_name,
                index: first,
                value: second.expect("map comprehension has a value"),
                span,
            }],
            _ => unreachable!(),
        };
        for clause in clauses.into_iter().rev() {
            if let Some(condition) = clause.condition {
                body = vec![AstStatement::If {
                    condition,
                    then_block: body,
                    else_block: Vec::new(),
                    span: clause.span,
                }];
            }
            body = vec![AstStatement::For {
                binding: clause.bindings[0].clone(),
                second_binding: clause.bindings.get(1).cloned(),
                iterable: clause.iterable,
                initializer: None,
                body,
                span: clause.span,
            }];
        }
        let loop_statement = self.statement(&body[0], bindings, result_type)?;
        Ok((name, vec![Statement::Binding(id), loop_statement]))
    }

    fn condition_expression(
        &mut self,
        condition: &AstExpression,
    ) -> Result<Expression, Diagnostic> {
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        let value = self.expression(condition, None)?;
        if value.type_id == boolean {
            return Ok(value);
        }
        if value.type_id == string {
            return Ok(self.runtime_call(
                "__sev_string_is_present",
                &[string],
                boolean,
                vec![value],
                condition.span,
            ));
        }
        if self.list_elements.contains_key(&value.type_id) {
            let storage = self.list_storage_expression(value, condition.span);
            let storage_type = storage.type_id;
            let usize_type = self
                .types
                .resolve_name("usize")
                .expect("bootstrap defines usize");
            let length = self.runtime_call(
                "__sev_list_len",
                &[storage_type],
                usize_type,
                vec![storage],
                condition.span,
            );
            let zero = self.integer_expression("0", usize_type, condition.span);
            return Ok(Expression {
                id: self.next_id(),
                type_id: boolean,
                kind: ExpressionKind::Binary {
                    operator: BinaryOperator::NotEqual,
                    left: Box::new(length),
                    right: Box::new(zero),
                },
                span: condition.span,
            });
        }
        self.coerce(value, boolean, false)
    }

    fn lower_function_hooks(
        &mut self,
        function: &severian_ast::FunctionDeclaration,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<(Block, Vec<LoweredHook>), Diagnostic> {
        let mut entry = Block::default();
        let mut lowered = Vec::new();
        let unit = self
            .types
            .resolve_name("unit")
            .expect("bootstrap defines unit");
        for decorator in &function.decorators {
            let Some(hook) = self.namespace_hooks.get(&decorator.name).cloned() else {
                continue;
            };
            let members = if decorator.arguments.is_empty() {
                hook.members.clone()
            } else {
                let mut selected = Vec::new();
                for argument in &decorator.arguments {
                    let severian_ast::DecoratorValue::Name(selector) = &argument.value else {
                        return Err(Diagnostic::new(
                            "E000218",
                            "hook selections must be hook names",
                            Some(argument.span),
                        ));
                    };
                    if argument.name.is_some() {
                        return Err(Diagnostic::new(
                            "E000218",
                            "hook selections are positional and execute in argument order",
                            Some(argument.span),
                        ));
                    }
                    let member = hook
                        .members
                        .iter()
                        .find(|member| {
                            member.method_name == *selector
                                || member.selectors.iter().any(|name| name == selector)
                        })
                        .cloned()
                        .ok_or_else(|| {
                            Diagnostic::new(
                                "E000218",
                                format!(
                                    "hook namespace `@{}` has no hook `{selector}`",
                                    decorator.name
                                ),
                                Some(argument.span),
                            )
                        })?;
                    selected.push(member);
                }
                selected
            };
            for member in members {
                let [(class_name, implementation)] = member.implementations.as_slice() else {
                    let detail = if member.implementations.is_empty() {
                        "has no implementation"
                    } else {
                        "has more than one implementation"
                    };
                    return Err(Diagnostic::new(
                        "E000206",
                        format!(
                            "hook `{}.{}` selected by `@{}` {detail}",
                            hook.trait_name, member.method_name, decorator.name
                        ),
                        Some(decorator.span),
                    ));
                };
                let specification = implementation.hook.as_ref().ok_or_else(|| {
                    Diagnostic::new(
                        "E000218",
                        format!(
                            "hook implementation `{class_name}.{}` lost its hook body",
                            implementation.name
                        ),
                        Some(implementation.span),
                    )
                })?;
                let context_parameter = implementation
                    .parameters
                    .iter()
                    .find(|parameter| parameter.name == specification.context)
                    .expect("hook context validation ran before lowering");
                let context_type = self.resolve_source_type(&context_parameter.annotation)?;
                let instance = self
                    .class_instances_by_type
                    .get(&context_type)
                    .cloned()
                    .ok_or_else(|| {
                        Diagnostic::new(
                            "E000204",
                            "hook context must be a concrete class",
                            Some(context_parameter.annotation.span),
                        )
                    })?;
                let fields = instance
                    .fields
                    .iter()
                    .map(|field| self.default_expression(field.ty, decorator.span))
                    .collect::<Result<Vec<_>, _>>()?;
                let context = self.new_binding_id();
                let variable = severian_hir::VariableId(context.0);
                bindings.push(Binding {
                    id: context,
                    variable,
                    type_id: context_type,
                    value: Expression {
                        id: self.next_id(),
                        type_id: context_type,
                        kind: ExpressionKind::Aggregate {
                            class: context_type,
                            fields,
                        },
                        span: decorator.span,
                    },
                    mutable: true,
                    preserve_error: false,
                    span: decorator.span,
                });
                entry.statements.push(Statement::Binding(context));
                if let Some((field, _)) = instance
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field)| field.name == "function")
                {
                    entry.statements.push(Statement::FieldSet {
                        binding: context,
                        field: field as u32,
                        value: self.string_expression(function.name.clone(), decorator.span),
                    });
                }

                let duration = instance
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field)| field.name == "duration")
                    .map(|(field, definition)| {
                        let started = self.new_binding_id();
                        let variable = severian_hir::VariableId(started.0);
                        bindings.push(Binding {
                            id: started,
                            variable,
                            type_id: definition.ty,
                            value: self.runtime_call(
                                "__sev_time_monotonic",
                                &[],
                                definition.ty,
                                Vec::new(),
                                decorator.span,
                            ),
                            mutable: false,
                            preserve_error: false,
                            span: decorator.span,
                        });
                        entry.statements.push(Statement::Binding(started));
                        let started = Expression {
                            id: self.next_id(),
                            type_id: definition.ty,
                            kind: ExpressionKind::Binding(started),
                            span: decorator.span,
                        };
                        let finished = self.runtime_call(
                            "__sev_time_monotonic",
                            &[],
                            definition.ty,
                            Vec::new(),
                            decorator.span,
                        );
                        let elapsed = Expression {
                            id: self.next_id(),
                            type_id: definition.ty,
                            kind: ExpressionKind::Binary {
                                operator: BinaryOperator::Subtract,
                                left: Box::new(finished),
                                right: Box::new(started),
                            },
                            span: decorator.span,
                        };
                        (field as u32, elapsed)
                    });

                let outer_names = self.names.clone();
                let outer_declarations = self.declarations.clone();
                self.names.insert(
                    specification.context.clone(),
                    (context, variable, context_type),
                );
                self.declarations.insert(specification.context.clone());
                let with_phase = self.block(&specification.with_phase, bindings, unit)?;
                self.names.clone_from(&outer_names);
                self.declarations.clone_from(&outer_declarations);

                self.names.insert(
                    specification.context.clone(),
                    (context, variable, context_type),
                );
                self.declarations.insert(specification.context.clone());
                let without_phase = self.block(&specification.without_phase, bindings, unit)?;
                self.names = outer_names;
                self.declarations = outer_declarations;

                entry.statements.extend(with_phase.statements);
                lowered.push(LoweredHook {
                    context,
                    result_field: instance
                        .fields
                        .iter()
                        .position(|field| {
                            field.name == "result" && self.types.assignable(result_type, field.ty)
                        })
                        .map(|field| field as u32),
                    error_field: instance
                        .fields
                        .iter()
                        .position(|field| field.name == "error")
                        .map(|field| field as u32),
                    duration,
                    without_phase,
                });
            }
        }
        Ok((entry, lowered))
    }

    fn contract_assertion(
        &mut self,
        contract: &severian_ast::FunctionContract,
    ) -> Result<Statement, Diagnostic> {
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let condition = self.expression(&contract.condition, Some(boolean))?;
        if let Some(failure) = &contract.failure {
            let error = self.expression(failure, None)?;
            if self.is_error_type(error.type_id) {
                let unit = self
                    .types
                    .resolve_name("unit")
                    .expect("bootstrap defines unit");
                let success = Expression {
                    id: self.next_id(),
                    type_id: unit,
                    kind: ExpressionKind::Literal(LiteralValue::Unit),
                    span: contract.span,
                };
                let failure = Expression {
                    id: self.next_id(),
                    type_id: unit,
                    kind: ExpressionKind::Throw(Box::new(error)),
                    span: contract.span,
                };
                return Ok(Statement::Expression(Expression {
                    id: self.next_id(),
                    type_id: unit,
                    kind: ExpressionKind::Fallback {
                        condition: Box::new(condition),
                        value: Box::new(success),
                        fallback: Box::new(failure),
                    },
                    span: contract.span,
                }));
            }
        }
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        let message = contract_failure_expression(contract.failure.as_ref())
            .map(|message| self.expression(message, Some(string)))
            .transpose()?;
        Ok(Statement::Assert {
            condition,
            message,
            span: contract.span,
            condition_span: contract.condition.span,
        })
    }

    fn next_id(&mut self) -> HirId {
        let id = HirId(self.next_hir);
        self.next_hir += 1;
        id
    }

    fn apply_parameter_effects(
        &mut self,
        function: FunctionId,
        arguments: Vec<Expression>,
        span: severian_source::Span,
    ) -> Vec<Expression> {
        let effects = self
            .parameter_effects
            .get(&function)
            .cloned()
            .unwrap_or_else(|| vec![ParameterEffect::Shared; arguments.len()]);
        arguments
            .into_iter()
            .zip(effects)
            .map(|(argument, effect)| {
                if matches!(
                    argument.kind,
                    ExpressionKind::Borrow { .. } | ExpressionKind::Move(_)
                ) {
                    return argument;
                }
                let type_id = argument.type_id;
                Expression {
                    id: self.next_id(),
                    type_id,
                    kind: match effect {
                        ParameterEffect::Shared => ExpressionKind::Borrow {
                            operand: Box::new(argument),
                            exclusive: false,
                        },
                        ParameterEffect::Exclusive => ExpressionKind::Borrow {
                            operand: Box::new(argument),
                            exclusive: true,
                        },
                        ParameterEffect::Move => ExpressionKind::Move(Box::new(argument)),
                    },
                    span,
                }
            })
            .collect()
    }

    fn inferred_parameter_effect(&self, body: &Block, parameter: BindingId) -> ParameterEffect {
        body.statements.iter().fold(ParameterEffect::Shared, |effect, statement| {
            effect.max(self.statement_parameter_effect(statement, parameter))
        })
    }

    fn statement_parameter_effect(
        &self,
        statement: &Statement,
        parameter: BindingId,
    ) -> ParameterEffect {
        match statement {
            Statement::Sequence(block) => self.inferred_parameter_effect(block, parameter),
            Statement::Binding(binding) => self
                .active_binding_effect(*binding, parameter)
                .unwrap_or(ParameterEffect::Shared),
            Statement::FieldUpdate { binding, value, .. }
            | Statement::FieldSet { binding, value, .. } => {
                let direct = if *binding == parameter {
                    ParameterEffect::Exclusive
                } else {
                    ParameterEffect::Shared
                };
                direct.max(self.expression_parameter_effect(value, parameter))
            }
            Statement::Expression(expression) => {
                self.expression_parameter_effect(expression, parameter)
            }
            Statement::Return(Some(expression)) => {
                let returned = if expression_is_binding(expression, parameter) {
                    ParameterEffect::Move
                } else {
                    ParameterEffect::Shared
                };
                returned.max(self.expression_parameter_effect(expression, parameter))
            }
            Statement::Return(None) | Statement::Break { .. } | Statement::Continue { .. } => {
                ParameterEffect::Shared
            }
            Statement::Assert {
                condition, message, ..
            } => {
                let mut effect = self.expression_parameter_effect(condition, parameter);
                if let Some(message) = message {
                    effect = effect.max(self.expression_parameter_effect(message, parameter));
                }
                effect
            }
            Statement::ExpectThrow { body, .. } => self.inferred_parameter_effect(body, parameter),
            Statement::Try {
                body, catch_body, ..
            } => self
                .inferred_parameter_effect(body, parameter)
                .max(self.inferred_parameter_effect(catch_body, parameter)),
            Statement::If {
                condition,
                then_block,
                else_block,
            } => self
                .expression_parameter_effect(condition, parameter)
                .max(self.inferred_parameter_effect(then_block, parameter))
                .max(self.inferred_parameter_effect(else_block, parameter)),
            Statement::While {
                condition, body, ..
            } => self
                .expression_parameter_effect(condition, parameter)
                .max(self.inferred_parameter_effect(body, parameter)),
            Statement::Match { subject, arms } => arms.iter().fold(
                self.expression_parameter_effect(subject, parameter),
                |effect, arm| effect.max(self.inferred_parameter_effect(&arm.body, parameter)),
            ),
        }
    }

    fn active_binding_effect(
        &self,
        binding: BindingId,
        parameter: BindingId,
    ) -> Option<ParameterEffect> {
        self.binding_values
            .get(&binding)
            .map(|value| self.expression_parameter_effect(value, parameter))
    }

    fn expression_parameter_effect(
        &self,
        expression: &Expression,
        parameter: BindingId,
    ) -> ParameterEffect {
        match &expression.kind {
            ExpressionKind::Literal(_) | ExpressionKind::Binding(_) | ExpressionKind::Function(_) => {
                ParameterEffect::Shared
            }
            ExpressionKind::Aggregate { fields, .. } => fields.iter().fold(
                ParameterEffect::Shared,
                |effect, field| effect.max(self.expression_parameter_effect(field, parameter)),
            ),
            ExpressionKind::Field { object, .. }
            | ExpressionKind::Await(object)
            | ExpressionKind::Throw(object)
            | ExpressionKind::Convert { operand: object, .. }
            | ExpressionKind::Unary { operand: object, .. } => {
                self.expression_parameter_effect(object, parameter)
            }
            ExpressionKind::Call { callee, arguments } => {
                let mut effect = arguments.iter().fold(
                    ParameterEffect::Shared,
                    |effect, argument| {
                        effect.max(self.expression_parameter_effect(argument, parameter))
                    },
                );
                if self.mutating_runtime_callee(callee)
                    && arguments
                        .iter()
                        .any(|argument| expression_contains_binding(argument, parameter))
                {
                    effect = effect.max(ParameterEffect::Exclusive);
                }
                effect
            }
            ExpressionKind::Async { expression, .. } => {
                self.expression_parameter_effect(expression, parameter)
            }
            ExpressionKind::AsyncFieldUpdate { binding, value, .. } => {
                let direct = if *binding == parameter {
                    ParameterEffect::Exclusive
                } else {
                    ParameterEffect::Shared
                };
                direct.max(self.expression_parameter_effect(value, parameter))
            }
            ExpressionKind::Fallback {
                condition,
                value,
                fallback,
            } => self
                .expression_parameter_effect(condition, parameter)
                .max(self.expression_parameter_effect(value, parameter))
                .max(self.expression_parameter_effect(fallback, parameter)),
            ExpressionKind::Borrow { operand, exclusive } => {
                let nested = self.expression_parameter_effect(operand, parameter);
                if *exclusive && expression_contains_binding(operand, parameter) {
                    nested.max(ParameterEffect::Exclusive)
                } else {
                    nested
                }
            }
            ExpressionKind::Move(operand) => {
                let nested = self.expression_parameter_effect(operand, parameter);
                if expression_contains_binding(operand, parameter) {
                    nested.max(ParameterEffect::Move)
                } else {
                    nested
                }
            }
            ExpressionKind::Binary { left, right, .. } => self
                .expression_parameter_effect(left, parameter)
                .max(self.expression_parameter_effect(right, parameter)),
        }
    }

    fn mutating_runtime_callee(&self, callee: &severian_hir::Callee) -> bool {
        let severian_hir::Callee::Direct { function, .. } = callee else {
            return false;
        };
        self.runtime_definitions.iter().any(|(symbol, definition)| {
            definition == function
                && (symbol == "__sev_list_clear"
                    || symbol.contains("_push_")
                    || symbol.contains("_pop_")
                    || symbol.contains("_append_")
                    || symbol.contains("_remove_")
                    || symbol.contains("_insert_"))
        })
    }

    fn prepare(&mut self, ast: &AstExpression) -> Result<Prepared, Diagnostic> {
        match &ast.kind {
            AstExpressionKind::Literal(AstLiteral::Measured { .. }) => {
                self.expression(ast, None).map(Prepared::Resolved)
            }
            AstExpressionKind::Literal(value) => {
                Ok(Prepared::Literal(universal_literal(value), ast.span))
            }
            _ => self.expression(ast, None).map(Prepared::Resolved),
        }
    }

    fn finish(&mut self, prepared: Prepared, expected: TypeId) -> Result<Expression, Diagnostic> {
        match prepared {
            Prepared::Literal(value, span) => {
                if let Ok(type_id) = self.types.resolve_literal(&value, Some(expected)) {
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id,
                        kind: ExpressionKind::Literal(value),
                        span,
                    });
                }
                let type_id = self
                    .types
                    .resolve_literal(&value, None)
                    .map_err(|error| semantic_error(error.to_string(), span))?;
                let expression = Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Literal(value),
                    span,
                };
                self.coerce(expression, expected, false)
            }
            Prepared::Resolved(expression) => self.coerce(expression, expected, false),
        }
    }

    fn expression(
        &mut self,
        ast: &AstExpression,
        expected: Option<TypeId>,
    ) -> Result<Expression, Diagnostic> {
        let expression = self.expression_inner(ast, expected)?;
        match expected {
            Some(expected) => self.coerce(expression, expected, false),
            None => Ok(expression),
        }
    }

    fn expression_inner(
        &mut self,
        ast: &AstExpression,
        expected: Option<TypeId>,
    ) -> Result<Expression, Diagnostic> {
        match &ast.kind {
            AstExpressionKind::Lambda { .. } => Err(Diagnostic::new(
                "E000205",
                "a lambda must be bound or passed to a function-typed parameter",
                Some(ast.span),
            )),
            AstExpressionKind::Mock { .. } => Err(Diagnostic::new(
                "E000217",
                "`mock` declarations are only valid as test statements",
                Some(ast.span),
            )),
            AstExpressionKind::Set(values) => {
                let Some(first) = values.first() else {
                    return Err(Diagnostic::new(
                        "E000204",
                        "an empty set must be written as `set()`",
                        Some(ast.span),
                    ));
                };
                let element = self.expression(first, None)?.type_id;
                let set = self.empty_set_expression(Some(element), ast.span)?;
                let set_type = set.type_id;
                if expected.is_some_and(|expected| expected != set_type) {
                    return Err(semantic_error(
                        "set literal does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                let storage_type = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines pointer-backed string");
                let suffix = self.list_runtime_suffix(element, ast.span)?;
                let mut storage = self.collection_storage_expression(set, 0, ast.span);
                for value in values {
                    let value = self.expression(value, Some(element))?;
                    storage = self.runtime_call(
                        &format!("__sev_set_append_{suffix}"),
                        &[storage_type, element],
                        storage_type,
                        vec![storage, value],
                        ast.span,
                    );
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: set_type,
                    kind: ExpressionKind::Aggregate {
                        class: set_type,
                        fields: vec![storage],
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::ListComprehension { .. }
            | AstExpressionKind::SetComprehension { .. }
            | AstExpressionKind::MapComprehension { .. } => Err(Diagnostic::new(
                "E000211",
                "collection comprehensions require statement-level lowering",
                Some(ast.span),
            )),
            AstExpressionKind::Map(entries) => {
                if entries.is_empty()
                    && expected.is_none_or(|ty| !self.map_elements.contains_key(&ty))
                {
                    return Err(Diagnostic::new(
                        "E000204",
                        "an empty map literal requires an expected `map[K, V]` type",
                        Some(ast.span),
                    ));
                }
                let (map_type, key_type, value_type) = if let Some(map_type) =
                    expected.filter(|ty| self.map_elements.contains_key(ty))
                {
                    let (key, value) = self.map_elements[&map_type];
                    (map_type, key, value)
                } else {
                    let first = &entries[0];
                    let key = self.expression(&first.key, None)?.type_id;
                    let value = self.expression(&first.value, None)?.type_id;
                    (self.instantiate_map_type(key, value), key, value)
                };
                let storage_type = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines pointer-backed string");
                let mut keys =
                    self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), ast.span);
                let mut values =
                    self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), ast.span);
                let key_symbol = format!(
                    "__sev_list_append_{}",
                    self.list_runtime_suffix(key_type, ast.span)?
                );
                let value_symbol = format!(
                    "__sev_list_append_{}",
                    self.list_runtime_suffix(value_type, ast.span)?
                );
                for entry in entries {
                    let key = self.expression(&entry.key, Some(key_type))?;
                    let value = self.expression(&entry.value, Some(value_type))?;
                    keys = self.runtime_call(
                        &key_symbol,
                        &[storage_type, key_type],
                        storage_type,
                        vec![keys, key],
                        entry.span,
                    );
                    values = self.runtime_call(
                        &value_symbol,
                        &[storage_type, value_type],
                        storage_type,
                        vec![values, value],
                        entry.span,
                    );
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: map_type,
                    kind: ExpressionKind::Aggregate {
                        class: map_type,
                        fields: vec![keys, values],
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Async {
                expression,
                owner,
                locked,
            } => {
                if !matches!(expression.kind, AstExpressionKind::Call { .. }) {
                    return Err(Diagnostic::new(
                        "E000211",
                        "`async` requires a function or method call",
                        Some(ast.span),
                    ));
                }
                if *owner == severian_ast::TaskOwner::Runtime && self.unsafe_depth == 0 {
                    return Err(Diagnostic::new(
                        "E000211",
                        "runtime-owned tasks require an `unsafe` scope",
                        Some(ast.span),
                    ));
                }
                if let Some(update) = self.class_method_update(expression)? {
                    let Statement::FieldUpdate {
                        binding,
                        field,
                        operator,
                        value,
                    } = update
                    else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "async class methods currently require one field update",
                            Some(expression.span),
                        ));
                    };
                    let unit = self
                        .types
                        .resolve_name("unit")
                        .expect("bootstrap defines unit");
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: unit,
                        kind: ExpressionKind::AsyncFieldUpdate {
                            binding,
                            field,
                            operator,
                            value: Box::new(value),
                            owner: match owner {
                                severian_ast::TaskOwner::SelfScope => {
                                    severian_hir::TaskOwner::SelfScope
                                }
                                severian_ast::TaskOwner::Runtime => {
                                    severian_hir::TaskOwner::Runtime
                                }
                                severian_ast::TaskOwner::Inferred => {
                                    severian_hir::TaskOwner::Inferred
                                }
                            },
                            locked: *locked,
                        },
                        span: ast.span,
                    });
                }
                // A task owns the complete result of its call, including a
                // fallible result envelope. Error propagation happens when
                // the task is awaited, not before the call is spawned.
                self.preserve_error_depth += 1;
                let lowered = self.expression(expression, expected);
                self.preserve_error_depth -= 1;
                let expression = lowered?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id: expression.type_id,
                    kind: ExpressionKind::Async {
                        expression: Box::new(expression),
                        owner: match owner {
                            severian_ast::TaskOwner::SelfScope => {
                                severian_hir::TaskOwner::SelfScope
                            }
                            severian_ast::TaskOwner::Runtime => severian_hir::TaskOwner::Runtime,
                            severian_ast::TaskOwner::Inferred => severian_hir::TaskOwner::Inferred,
                        },
                        locked: *locked,
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Await { expression } => {
                // The surrounding expected type describes the awaited value,
                // not the task handle (which may still carry a fallible
                // envelope).
                let expression = self.expression(expression, None)?;
                if let Some(element) = self.channel_elements.get(&expression.type_id).copied() {
                    let storage = self.channel_storage_expression(expression, ast.span);
                    let storage_type = storage.type_id;
                    let suffix = self.list_runtime_suffix(element, ast.span)?;
                    return Ok(self.runtime_call(
                        &format!("__sev_channel_recv_{suffix}"),
                        &[storage_type],
                        element,
                        vec![storage],
                        ast.span,
                    ));
                }
                let awaited = Expression {
                    id: self.next_id(),
                    type_id: expression.type_id,
                    kind: ExpressionKind::Await(Box::new(expression)),
                    span: ast.span,
                };
                if self.preserve_error_depth == 0 {
                    if let Some(fallible) = self.fallible_types.get(&awaited.type_id).copied() {
                        return Ok(self.unwrap_fallible_expression(awaited, fallible, ast.span));
                    }
                }
                Ok(awaited)
            }
            AstExpressionKind::Conditional {
                value,
                condition,
                fallback,
            } => {
                let condition = self.condition_expression(condition)?;
                let value = self.expression(value, expected)?;
                let fallback = self.expression(fallback, Some(value.type_id))?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id: value.type_id,
                    kind: ExpressionKind::Fallback {
                        condition: Box::new(condition),
                        value: Box::new(value),
                        fallback: Box::new(fallback),
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Fallback { value, fallback } => {
                let value = self.expression(value, expected)?;
                let is_string = self
                    .types
                    .definition(value.type_id)
                    .is_some_and(|definition| definition.name == "string");
                if !is_string {
                    return Err(Diagnostic::new(
                        "E000204",
                        "`else` fallback currently requires an optional string value",
                        Some(ast.span),
                    ));
                }
                let boolean = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");
                let condition = self.runtime_call(
                    "__sev_string_is_present",
                    &[value.type_id],
                    boolean,
                    vec![value.clone()],
                    ast.span,
                );
                let fallback = self.expression(fallback, Some(value.type_id))?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id: value.type_id,
                    kind: ExpressionKind::Fallback {
                        condition: Box::new(condition),
                        value: Box::new(value),
                        fallback: Box::new(fallback),
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Throw { error } => {
                let error = self.expression(error, None)?;
                if !self.is_error_type(error.type_id) {
                    return Err(Diagnostic::new(
                        "E000215",
                        "only an error value may be thrown",
                        Some(error.span),
                    ));
                }
                let type_id = expected.unwrap_or_else(|| {
                    self.types
                        .resolve_name("unit")
                        .expect("bootstrap defines unit")
                });
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Throw(Box::new(error)),
                    span: ast.span,
                })
            }
            AstExpressionKind::Literal(value) => {
                if matches!(value, AstLiteral::None)
                    && expected.is_some_and(|expected| self.optional_types.contains(&expected))
                {
                    return self.default_expression(expected.unwrap(), ast.span);
                }
                if let AstLiteral::Measured { magnitude, suffix } = value {
                    let (type_name, value) = measured_literal(magnitude, suffix, ast.span)?;
                    let type_id = self.types.resolve_name(type_name).ok_or_else(|| {
                        Diagnostic::new(
                            "E000204",
                            format!("the `{suffix}` unit type is unavailable"),
                            Some(ast.span),
                        )
                    })?;
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id,
                        kind: ExpressionKind::Literal(value),
                        span: ast.span,
                    });
                }
                let value = if matches!(value, AstLiteral::None)
                    && expected.is_some_and(|expected| {
                        self.types
                            .definition(expected)
                            .is_some_and(|definition| definition.name == "string")
                    }) {
                    LiteralValue::String(String::new())
                } else {
                    universal_literal(value)
                };
                let literal_expected = expected.filter(|expected| {
                    self.types.primitive(*expected).is_some_and(|primitive| {
                        primitive.category.literal_kind() == Some(value.kind())
                    })
                });
                let type_id = self
                    .types
                    .resolve_literal(&value, literal_expected)
                    .map_err(|error| semantic_error(error.to_string(), ast.span))?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Literal(value),
                    span: ast.span,
                })
            }
            AstExpressionKind::List(values) => {
                let (list_type, element) = if let Some(list_type) =
                    expected.filter(|ty| self.list_elements.contains_key(ty))
                {
                    (list_type, self.list_elements[&list_type])
                } else if values.is_empty() {
                    let element = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    (self.instantiate_list_type(element), element)
                } else {
                    let first = self.expression(&values[0], None)?;
                    let element = first.type_id;
                    (self.instantiate_list_type(element), element)
                };
                let list = self.empty_list_expression(list_type, ast.span)?;
                if values.is_empty() {
                    return Ok(list);
                }
                let storage_type = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines pointer-backed string");
                let suffix = self.list_runtime_suffix(element, ast.span)?;
                let symbol = format!("__sev_list_append_{suffix}");
                let mut storage = self.list_storage_expression(list, ast.span);
                for value in values {
                    let value = self.expression(value, Some(element))?;
                    storage = self.runtime_call(
                        &symbol,
                        &[storage_type, element],
                        storage_type,
                        vec![storage, value],
                        ast.span,
                    );
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: list_type,
                    kind: ExpressionKind::Aggregate {
                        class: list_type,
                        fields: vec![storage],
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Tuple(values) => {
                if values.is_empty() {
                    return self.expression(
                        &AstExpression {
                            kind: AstExpressionKind::Literal(AstLiteral::Unit),
                            span: ast.span,
                        },
                        expected,
                    );
                }
                let mut fields = Vec::with_capacity(values.len());
                for value in values {
                    fields.push(self.expression(value, None)?);
                }
                let element_types = fields.iter().map(|field| field.type_id).collect::<Vec<_>>();
                let tuple_type = self.instantiate_tuple_type(&element_types);
                if expected.is_some_and(|expected| expected != tuple_type) {
                    return Err(semantic_error(
                        "tuple does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: tuple_type,
                    kind: ExpressionKind::Aggregate {
                        class: tuple_type,
                        fields,
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Name(name) => {
                if name == "absent" {
                    let Some(expected) = expected else {
                        return Err(Diagnostic::new(
                            "E000204",
                            "`absent` requires an optional expected type",
                            Some(ast.span),
                        ));
                    };
                    if self
                        .types
                        .definition(expected)
                        .is_some_and(|definition| definition.name == "string")
                    {
                        return Ok(Expression {
                            id: self.next_id(),
                            type_id: expected,
                            kind: ExpressionKind::Literal(LiteralValue::String(String::new())),
                            span: ast.span,
                        });
                    }
                }
                if self.enum_variants.contains_key(name) {
                    return self.enum_constructor(name, &[], expected, ast.span);
                }
                if let Some(value) = self.value_substitutions.get(name).cloned() {
                    if expected
                        .is_some_and(|expected| !self.types.assignable(value.type_id, expected))
                    {
                        return Err(semantic_error(
                            "substituted value does not satisfy the expected type".into(),
                            ast.span,
                        ));
                    }
                    return Ok(value);
                }
                let Some((binding, _, type_id)) = self.names.get(name).copied() else {
                    return Err(Diagnostic::new(
                        "E000201",
                        format!("unknown binding `{name}`"),
                        Some(ast.span),
                    ));
                };
                if expected.is_some_and(|expected| !self.types.assignable(type_id, expected)) {
                    return Err(semantic_error(
                        "binding type does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Binding(binding),
                    span: ast.span,
                })
            }
            AstExpressionKind::Member { object, name } => {
                if let Some(path) = callable_path(ast) {
                    if self.enum_variants.contains_key(&path) {
                        return self.enum_constructor(&path, &[], expected, ast.span);
                    }
                    if let Some(mut value) = self.value_substitutions.get(&path).cloned() {
                        if expected
                            .is_some_and(|expected| !self.types.assignable(value.type_id, expected))
                        {
                            return Err(semantic_error(
                                "package constant does not satisfy the expected type".into(),
                                ast.span,
                            ));
                        }
                        value.id = self.next_id();
                        value.span = ast.span;
                        return Ok(value);
                    }
                }
                let object = self.expression(object, None)?;
                if let Some(fallible) = self.fallible_types.get(&object.type_id).copied() {
                    if self.types.resolve_name("Error") == Some(fallible.error)
                        && (name == "message" || name == "call_stack")
                    {
                        let string = self
                            .types
                            .resolve_name("string")
                            .expect("bootstrap defines string");
                        if expected.is_some_and(|expected| expected != string) {
                            return Err(semantic_error(
                                "error property does not satisfy the expected type".into(),
                                ast.span,
                            ));
                        }
                        let error = Expression {
                            id: self.next_id(),
                            type_id: fallible.error,
                            kind: ExpressionKind::Field {
                                object: Box::new(object),
                                index: 2,
                            },
                            span: ast.span,
                        };
                        let symbol = if name == "message" {
                            "__sev_error_message"
                        } else {
                            "__sev_error_call_stack"
                        };
                        return Ok(self.runtime_call(
                            symbol,
                            &[fallible.error],
                            string,
                            vec![error],
                            ast.span,
                        ));
                    }
                    if let Some(instance) = self.class_instances_by_type.get(&fallible.error) {
                        if let Some((index, field)) = instance
                            .fields
                            .iter()
                            .enumerate()
                            .find(|(_, field)| field.name == *name)
                        {
                            let field_type = field.ty;
                            let error = Expression {
                                id: self.next_id(),
                                type_id: fallible.error,
                                kind: ExpressionKind::Field {
                                    object: Box::new(object),
                                    index: 2,
                                },
                                span: ast.span,
                            };
                            return Ok(Expression {
                                id: self.next_id(),
                                type_id: field_type,
                                kind: ExpressionKind::Field {
                                    object: Box::new(error),
                                    index: index as u32,
                                },
                                span: ast.span,
                            });
                        }
                    }
                }
                if self.types.resolve_name("Error") == Some(object.type_id)
                    && (name == "message" || name == "call_stack")
                {
                    let string = self
                        .types
                        .resolve_name("string")
                        .expect("bootstrap defines string");
                    if expected.is_some_and(|expected| expected != string) {
                        return Err(semantic_error(
                            "error property does not satisfy the expected type".into(),
                            ast.span,
                        ));
                    }
                    let error_type = object.type_id;
                    let symbol = if name == "message" {
                        "__sev_error_message"
                    } else {
                        "__sev_error_call_stack"
                    };
                    return Ok(self.runtime_call(
                        symbol,
                        &[error_type],
                        string,
                        vec![object],
                        ast.span,
                    ));
                }
                let Some(instance) = self.class_instances_by_type.get(&object.type_id) else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("type has no field `{name}`"),
                        Some(ast.span),
                    ));
                };
                if name.starts_with("__") {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("field `{name}` is private to class `{}`", instance.name),
                        Some(ast.span),
                    )
                    .with_help("access private state through a public class method"));
                }
                let Some((index, field)) = instance
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, field)| field.name == *name)
                else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("class `{}` has no field `{name}`", instance.name),
                        Some(ast.span),
                    ));
                };
                let field_type = field.ty;
                if expected.is_some_and(|expected| !self.types.assignable(field_type, expected)) {
                    return Err(semantic_error(
                        "field type does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                Ok(Expression {
                    id: self.next_id(),
                    type_id: field_type,
                    kind: ExpressionKind::Field {
                        object: Box::new(object),
                        index: index as u32,
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Index { object, index } => {
                let object = self.expression(object, None)?;
                if let Some(element) = self.pointer_elements.get(&object.type_id).copied() {
                    if self.unsafe_depth == 0 {
                        return Err(Diagnostic::new(
                            "E000219",
                            "raw pointer access requires an `unsafe` scope",
                            Some(ast.span),
                        ));
                    }
                    if expected.is_some_and(|expected| expected != element) {
                        return Err(semantic_error(
                            "pointed value does not satisfy the expected type".into(),
                            ast.span,
                        ));
                    }
                    let index = self.expression(index, None)?;
                    let index_name = self
                        .types
                        .definition(index.type_id)
                        .map(|definition| definition.name.as_str());
                    if !matches!(index_name, Some("int" | "i64" | "usize")) {
                        return Err(Diagnostic::new(
                            "E000211",
                            "pointer indices must be integers",
                            Some(index.span),
                        ));
                    }
                    let suffix = self.list_runtime_suffix(element, ast.span)?;
                    return Ok(self.runtime_call(
                        &format!("__sev_pointer_index_{suffix}"),
                        &[object.type_id, index.type_id],
                        element,
                        vec![object, index],
                        ast.span,
                    ));
                }
                let string = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                if object.type_id == string {
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let index = self.expression(index, Some(integer))?;
                    return Ok(self.runtime_call(
                        "__sev_string_index",
                        &[string, integer],
                        string,
                        vec![object, index],
                        ast.span,
                    ));
                }
                if let Some(elements) = self.tuple_elements.get(&object.type_id).cloned() {
                    let index_span = index.span;
                    let AstExpressionKind::Literal(AstLiteral::Integer(index)) = &index.kind else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "tuple indices must be integer literals",
                            Some(index.span),
                        ));
                    };
                    let index = index.parse::<usize>().map_err(|_| {
                        Diagnostic::new("E000211", "invalid tuple index", Some(index_span))
                    })?;
                    let Some(element) = elements.get(index).copied() else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "tuple index is out of bounds",
                            Some(ast.span),
                        ));
                    };
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: element,
                        kind: ExpressionKind::Field {
                            object: Box::new(object),
                            index: index as u32,
                        },
                        span: ast.span,
                    });
                }
                if let Some((key_type, value_type)) =
                    self.map_elements.get(&object.type_id).copied()
                {
                    if expected
                        .is_some_and(|expected| !self.types.assignable(value_type, expected))
                    {
                        return Err(semantic_error(
                            "map value does not satisfy the expected type".into(),
                            ast.span,
                        ));
                    }
                    let key = self.expression(index, Some(key_type))?;
                    let keys = self.collection_storage_expression(object.clone(), 0, ast.span);
                    let values = self.collection_storage_expression(object, 1, ast.span);
                    let storage_type = keys.type_id;
                    let key_suffix = self.list_runtime_suffix(key_type, ast.span)?;
                    let value_suffix = self.list_runtime_suffix(value_type, ast.span)?;
                    return Ok(self.runtime_call(
                        &format!("__sev_map_get_{key_suffix}_{value_suffix}"),
                        &[storage_type, storage_type, key_type],
                        value_type,
                        vec![keys, values, key],
                        ast.span,
                    ));
                }
                let Some(element) = self.list_elements.get(&object.type_id).copied() else {
                    return Err(Diagnostic::new(
                        "E000211",
                        "indexing is not implemented for this type",
                        Some(ast.span),
                    ));
                };
                if expected.is_some_and(|expected| !self.types.assignable(element, expected)) {
                    return Err(semantic_error(
                        "indexed value does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                let index = self.expression(index, None)?;
                if !self.integer_primitive(index.type_id) {
                    return Err(Diagnostic::new(
                        "E000211",
                        "list indices must be integers",
                        Some(index.span),
                    ));
                }
                let integer = self
                    .types
                    .resolve_name("int")
                    .expect("bootstrap defines int");
                let index = self.coerce(index, integer, true)?;
                let storage = self.list_storage_expression(object, ast.span);
                let storage_type = storage.type_id;
                let suffix = self.list_runtime_suffix(element, ast.span)?;
                Ok(self.runtime_call(
                    &format!("__sev_list_index_{suffix}"),
                    &[storage_type, index.type_id],
                    element,
                    vec![storage, index],
                    ast.span,
                ))
            }
            AstExpressionKind::Slice {
                object,
                start,
                end,
                step,
                start_exclusive,
                end_inclusive,
            } => {
                let object_value = self.expression(object, None)?;
                if let Some(elements) = self.tuple_elements.get(&object_value.type_id).cloned() {
                    let reverse = start.is_none()
                        && end.is_none()
                        && step.as_deref().is_some_and(|step| {
                            matches!(
                                &step.kind,
                                AstExpressionKind::Unary {
                                    operator: AstUnaryOperator::Negative,
                                    operand
                                } if matches!(operand.kind, AstExpressionKind::Literal(AstLiteral::Integer(ref value)) if value == "1")
                            )
                        });
                    if !reverse {
                        return Err(Diagnostic::new(
                            "E000211",
                            "tuple slicing currently supports `[::-1]`",
                            Some(ast.span),
                        ));
                    }
                    let reversed_types = elements.iter().rev().copied().collect::<Vec<_>>();
                    let result_type = self.instantiate_tuple_type(&reversed_types);
                    let mut fields = Vec::with_capacity(elements.len());
                    for (index, element) in elements.iter().enumerate().rev() {
                        fields.push(Expression {
                            id: self.next_id(),
                            type_id: *element,
                            kind: ExpressionKind::Field {
                                object: Box::new(object_value.clone()),
                                index: index as u32,
                            },
                            span: ast.span,
                        });
                    }
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: result_type,
                        kind: ExpressionKind::Aggregate {
                            class: result_type,
                            fields,
                        },
                        span: ast.span,
                    });
                }
                let string = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                let integer = self
                    .types
                    .resolve_name("int")
                    .expect("bootstrap defines int");
                let list_element = self.list_elements.get(&object_value.type_id).copied();
                if object_value.type_id != string && list_element.is_none() {
                    return Err(Diagnostic::new(
                        "E000211",
                        "slicing is not implemented for this type",
                        Some(ast.span),
                    ));
                }
                let result_type = object_value.type_id;
                if expected
                    .is_some_and(|expected| !self.types.assignable(result_type, expected))
                {
                    return Err(semantic_error(
                        "slice result does not satisfy the expected type".into(),
                        ast.span,
                    ));
                }
                let start_value = match start {
                    Some(start) => self.expression(start, Some(integer))?,
                    None => self.integer_expression("0", integer, ast.span),
                };
                let end_value = match end {
                    Some(end) => self.expression(end, Some(integer))?,
                    None => self.integer_expression("0", integer, ast.span),
                };
                let step = match step {
                    Some(step) => self.expression(step, Some(integer))?,
                    None => self.integer_expression("1", integer, ast.span),
                };
                let boolean = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");
                let mut flag = |value| Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Literal(LiteralValue::Boolean(value)),
                    span: ast.span,
                };
                let flags = vec![
                    flag(start.is_some()),
                    flag(end.is_some()),
                    flag(*start_exclusive),
                    flag(*end_inclusive),
                ];
                let (symbol, receiver) = if list_element.is_some() {
                    (
                        "__sev_list_slice",
                        self.list_storage_expression(object_value, ast.span),
                    )
                } else {
                    ("__sev_string_slice_ex", object_value)
                };
                let receiver_type = receiver.type_id;
                let mut arguments = vec![receiver, start_value, end_value, step];
                arguments.extend(flags);
                let sliced = self.runtime_call(
                    symbol,
                    &[
                        receiver_type,
                        integer,
                        integer,
                        integer,
                        boolean,
                        boolean,
                        boolean,
                        boolean,
                    ],
                    receiver_type,
                    arguments,
                    ast.span,
                );
                if list_element.is_some() {
                    Ok(Expression {
                        id: self.next_id(),
                        type_id: result_type,
                        kind: ExpressionKind::Aggregate {
                            class: result_type,
                            fields: vec![sliced],
                        },
                        span: ast.span,
                    })
                } else {
                    Ok(sliced)
                }
            }
            AstExpressionKind::TypeApplication { callee, arguments } => {
                if matches!(&callee.kind, AstExpressionKind::Name(name) if matches!(name.as_str(), "channel" | "Channel"))
                    && arguments.len() == 1
                {
                    return self.channel_constructor(&arguments[0], None, ast.span);
                }
                Err(Diagnostic::new(
                    "E000211",
                    "a generic type application must be constructed",
                    Some(ast.span),
                ))
            }
            AstExpressionKind::Call { callee, arguments } => {
                if let Some(call) = self.system_surface_call(callee, arguments, ast.span)? {
                    return Ok(call);
                }
                if let AstExpressionKind::TypeApplication {
                    callee: application,
                    arguments: type_arguments,
                } = &callee.kind
                {
                    if let AstExpressionKind::Name(name) = &application.kind {
                        if name == "set" && type_arguments.len() == 1 && arguments.is_empty() {
                            let element = self.resolve_source_type(&type_arguments[0])?;
                            return self.empty_set_expression(Some(element), ast.span);
                        }
                        if name == "allocate" {
                            if self.unsafe_depth == 0 {
                                return Err(Diagnostic::new(
                                    "E000219",
                                    "raw allocation requires an `unsafe` scope",
                                    Some(ast.span),
                                ));
                            }
                            let ([element], [count]) =
                                (type_arguments.as_slice(), arguments.as_slice())
                            else {
                                return Err(Diagnostic::new(
                                    "E000206",
                                    "`allocate[T]` expects one type and one element count",
                                    Some(ast.span),
                                ));
                            };
                            if count.name.is_some() {
                                return Err(Diagnostic::new(
                                    "E000206",
                                    "the allocation count must be positional",
                                    Some(count.value.span),
                                ));
                            }
                            let element = self.resolve_source_type(element)?;
                            let pointer = self.instantiate_pointer_type(element);
                            let usize_type = self
                                .types
                                .resolve_name("usize")
                                .expect("bootstrap defines usize");
                            let count = self.expression(&count.value, Some(usize_type))?;
                            return Ok(self.runtime_call(
                                "__sev_allocate",
                                &[usize_type],
                                pointer,
                                vec![count],
                                ast.span,
                            ));
                        }
                        if matches!(name.as_str(), "bytes" | "alignment" | "offset") {
                            let [queried] = type_arguments.as_slice() else {
                                return Err(Diagnostic::new(
                                    "E000206",
                                    format!("`{name}[T]` expects exactly one type"),
                                    Some(callee.span),
                                ));
                            };
                            let queried = self.resolve_source_type(queried)?;
                            let (size, alignment) = self.type_layout(queried, ast.span)?;
                            let value = match name.as_str() {
                                "bytes" | "alignment" if arguments.is_empty() => {
                                    if name == "bytes" { size } else { alignment }
                                }
                                "offset" => {
                                    let [field] = arguments.as_slice() else {
                                        return Err(Diagnostic::new(
                                            "E000206",
                                            "`offset[T]` expects one field name",
                                            Some(ast.span),
                                        ));
                                    };
                                    let AstExpressionKind::Literal(AstLiteral::String(field)) =
                                        &field.value.kind
                                    else {
                                        return Err(Diagnostic::new(
                                            "E000206",
                                            "`offset[T]` requires a string literal field name",
                                            Some(field.value.span),
                                        ));
                                    };
                                    self.type_field_offset(queried, field, ast.span)?
                                }
                                _ => {
                                    return Err(Diagnostic::new(
                                        "E000206",
                                        format!("`{name}[T]()` does not accept value arguments"),
                                        Some(ast.span),
                                    ));
                                }
                            };
                            let data_size = self
                                .types
                                .resolve_name("data_size")
                                .expect("bootstrap defines data_size");
                            return Ok(Expression {
                                id: self.next_id(),
                                type_id: data_size,
                                kind: ExpressionKind::Literal(LiteralValue::Float(format!(
                                    "{value}.0"
                                ))),
                                span: ast.span,
                            });
                        }
                    }
                }
                if callable_path(callee).as_deref() == Some("free") {
                    if self.unsafe_depth == 0 {
                        return Err(Diagnostic::new(
                            "E000219",
                            "freeing raw memory requires an `unsafe` scope",
                            Some(ast.span),
                        ));
                    }
                    let [argument] = arguments.as_slice() else {
                        return Err(Diagnostic::new(
                            "E000206",
                            "`free` expects exactly one raw pointer",
                            Some(ast.span),
                        ));
                    };
                    if argument.name.is_some() {
                        return Err(Diagnostic::new(
                            "E000206",
                            "the pointer passed to `free` must be positional",
                            Some(argument.value.span),
                        ));
                    }
                    let pointer = self.expression(&argument.value, None)?;
                    if !self.pointer_elements.contains_key(&pointer.type_id) {
                        return Err(Diagnostic::new(
                            "E000204",
                            "`free` expects a raw pointer",
                            Some(argument.value.span),
                        ));
                    }
                    let unit = self
                        .types
                        .resolve_name("unit")
                        .expect("bootstrap defines unit");
                    return Ok(self.runtime_call(
                        "__sev_free",
                        &[pointer.type_id],
                        unit,
                        vec![pointer],
                        ast.span,
                    ));
                }
                if matches!(callable_path(callee).as_deref(), Some("any" | "all"))
                    && arguments.len() == 1
                {
                    let boolean = self
                        .types
                        .resolve_name("bool")
                        .expect("bootstrap defines bool");
                    let list_type = self.instantiate_list_type(boolean);
                    let collection =
                        self.expression(&arguments[0].value, Some(list_type))?;
                    if self.list_elements.get(&collection.type_id).copied() != Some(boolean) {
                        return Err(Diagnostic::new(
                            "E000206",
                            "`any` and `all` expect a list of bool values",
                            Some(ast.span),
                        ));
                    }
                    let storage = self.list_storage_expression(collection, ast.span);
                    let storage_type = storage.type_id;
                    let name = callable_path(callee).expect("matched callable name");
                    return Ok(self.runtime_call(
                        &format!("__sev_list_{name}"),
                        &[storage_type],
                        boolean,
                        vec![storage],
                        ast.span,
                    ));
                }
                if callable_path(callee).as_deref() == Some("abs") && arguments.len() == 1 {
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let value = self.expression(&arguments[0].value, Some(integer))?;
                    return Ok(self.runtime_call(
                        "__sev_abs_i64",
                        &[integer],
                        integer,
                        vec![value],
                        ast.span,
                    ));
                }
                if matches!(callable_path(callee).as_deref(), Some("min" | "max"))
                    && arguments.len() == 2
                {
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let left = self.expression(&arguments[0].value, Some(integer))?;
                    let right = self.expression(&arguments[1].value, Some(integer))?;
                    let name = callable_path(callee).expect("matched callable name");
                    return Ok(self.runtime_call(
                        &format!("__sev_{name}_i64"),
                        &[integer, integer],
                        integer,
                        vec![left, right],
                        ast.span,
                    ));
                }
                if callable_path(callee).as_deref() == Some("divmod") && arguments.len() == 2 {
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let left = self.expression(&arguments[0].value, Some(integer))?;
                    let right = self.expression(&arguments[1].value, Some(integer))?;
                    let quotient = self.runtime_call(
                        "__sev_div_i64",
                        &[integer, integer],
                        integer,
                        vec![left.clone(), right.clone()],
                        ast.span,
                    );
                    let remainder = self.runtime_call(
                        "__sev_mod_i64",
                        &[integer, integer],
                        integer,
                        vec![left, right],
                        ast.span,
                    );
                    let tuple_type = self.instantiate_tuple_type(&[integer, integer]);
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: tuple_type,
                        kind: ExpressionKind::Aggregate {
                            class: tuple_type,
                            fields: vec![quotient, remainder],
                        },
                        span: ast.span,
                    });
                }
                if callable_path(callee).as_deref() == Some("range")
                    && (1..=3).contains(&arguments.len())
                    && arguments.iter().all(|argument| argument.name.is_none())
                {
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let mut values = arguments
                        .iter()
                        .map(|argument| self.expression(&argument.value, Some(integer)))
                        .collect::<Result<Vec<_>, Diagnostic>>()?;
                    let (start, end, step) = match values.len() {
                        1 => (
                            self.integer_expression("0", integer, ast.span),
                            values.remove(0),
                            self.integer_expression("1", integer, ast.span),
                        ),
                        2 => (
                            values.remove(0),
                            values.remove(0),
                            self.integer_expression("1", integer, ast.span),
                        ),
                        3 => (values.remove(0), values.remove(0), values.remove(0)),
                        _ => unreachable!(),
                    };
                    let storage_type = self
                        .types
                        .resolve_name("string")
                        .expect("bootstrap defines pointer-backed string");
                    let storage = self.runtime_call(
                        "__sev_range",
                        &[integer, integer, integer],
                        storage_type,
                        vec![start, end, step],
                        ast.span,
                    );
                    let result_type = self.instantiate_list_type(integer);
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: result_type,
                        kind: ExpressionKind::Aggregate {
                            class: result_type,
                            fields: vec![storage],
                        },
                        span: ast.span,
                    });
                }
                if callable_path(callee).as_deref() == Some("indices")
                    && arguments.len() == 1
                    && arguments[0].name.is_none()
                {
                    let collection = self.expression(&arguments[0].value, None)?;
                    if !self.list_elements.contains_key(&collection.type_id) {
                        return Err(Diagnostic::new(
                            "E000206",
                            "`indices` expects a list",
                            Some(ast.span),
                        ));
                    }
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let result_type = self.instantiate_list_type(integer);
                    let storage = self.list_storage_expression(collection, ast.span);
                    let storage_type = storage.type_id;
                    let indices = self.runtime_call(
                        "__sev_list_indices",
                        &[storage_type],
                        storage_type,
                        vec![storage],
                        ast.span,
                    );
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: result_type,
                        kind: ExpressionKind::Aggregate {
                            class: result_type,
                            fields: vec![indices],
                        },
                        span: ast.span,
                    });
                }
                if callable_path(callee).as_deref() == Some("enumerate")
                    && arguments.len() == 1
                    && arguments[0].name.is_none()
                {
                    let collection = self.expression(&arguments[0].value, None)?;
                    let Some(element) = self.list_elements.get(&collection.type_id).copied()
                    else {
                        return Err(Diagnostic::new(
                            "E000206",
                            "`enumerate` expects a list",
                            Some(ast.span),
                        ));
                    };
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let map_type = self.instantiate_map_type(integer, element);
                    let values = self.list_storage_expression(collection, ast.span);
                    let storage_type = values.type_id;
                    let keys = self.runtime_call(
                        "__sev_list_indices",
                        &[storage_type],
                        storage_type,
                        vec![values.clone()],
                        ast.span,
                    );
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: map_type,
                        kind: ExpressionKind::Aggregate {
                            class: map_type,
                            fields: vec![keys, values],
                        },
                        span: ast.span,
                    });
                }
                if callable_path(callee).as_deref() == Some("zip")
                    && arguments.len() == 2
                    && arguments.iter().all(|argument| argument.name.is_none())
                {
                    let left = self.expression(&arguments[0].value, None)?;
                    let right = self.expression(&arguments[1].value, None)?;
                    let Some(left_element) = self.list_elements.get(&left.type_id).copied() else {
                        return Err(Diagnostic::new("E000206", "`zip` expects lists", Some(ast.span)));
                    };
                    let Some(right_element) = self.list_elements.get(&right.type_id).copied() else {
                        return Err(Diagnostic::new("E000206", "`zip` expects lists", Some(ast.span)));
                    };
                    let map_type = self.instantiate_map_type(left_element, right_element);
                    let left = self.list_storage_expression(left, ast.span);
                    let right = self.list_storage_expression(right, ast.span);
                    let storage_type = left.type_id;
                    let keys = self.runtime_call(
                        "__sev_list_zip_left",
                        &[storage_type, storage_type],
                        storage_type,
                        vec![left.clone(), right.clone()],
                        ast.span,
                    );
                    let values = self.runtime_call(
                        "__sev_list_zip_right",
                        &[storage_type, storage_type],
                        storage_type,
                        vec![left, right],
                        ast.span,
                    );
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: map_type,
                        kind: ExpressionKind::Aggregate {
                            class: map_type,
                            fields: vec![keys, values],
                        },
                        span: ast.span,
                    });
                }
                if matches!(&callee.kind, AstExpressionKind::Name(name) if name == "set")
                    && arguments.is_empty()
                {
                    let element = expected.and_then(|expected| {
                        (self.set_type == Some(expected)).then_some(self.set_element).flatten()
                    });
                    return self.empty_set_expression(element, ast.span);
                }
                if let AstExpressionKind::Member { object, name } = &callee.kind {
                    if name == "clear" && arguments.is_empty() {
                        let collection = self.expression(object, None)?;
                        if self.list_elements.contains_key(&collection.type_id) {
                            let storage = self.list_storage_expression(collection, ast.span);
                            let storage_type = storage.type_id;
                            let unit = self
                                .types
                                .resolve_name("unit")
                                .expect("bootstrap defines unit");
                            return Ok(self.runtime_call(
                                "__sev_list_clear",
                                &[storage_type],
                                unit,
                                vec![storage],
                                ast.span,
                            ));
                        }
                    }
                    if matches!(name.as_str(), "add" | "append")
                        && arguments.len() == 1
                        && arguments[0].name.is_none()
                    {
                        let collection = self.expression(object, None)?;
                        if self.set_type == Some(collection.type_id) && name == "add" {
                            let value = self.expression(&arguments[0].value, None)?;
                            let suffix = self.list_runtime_suffix(value.type_id, ast.span)?;
                            let storage =
                                self.collection_storage_expression(collection, 0, ast.span);
                            let storage_type = storage.type_id;
                            let unit = self
                                .types
                                .resolve_name("unit")
                                .expect("bootstrap defines unit");
                            return Ok(self.runtime_call(
                                &format!("__sev_set_add_{suffix}"),
                                &[storage_type, value.type_id],
                                unit,
                                vec![storage, value],
                                ast.span,
                            ));
                        }
                        if let Some(element) = self.list_elements.get(&collection.type_id).copied()
                        {
                            let value = self.expression(&arguments[0].value, Some(element))?;
                            let suffix = self.list_runtime_suffix(element, ast.span)?;
                            let storage = self.list_storage_expression(collection, ast.span);
                            let storage_type = storage.type_id;
                            let unit = self
                                .types
                                .resolve_name("unit")
                                .expect("bootstrap defines unit");
                            return Ok(self.runtime_call(
                                &format!("__sev_list_push_{suffix}"),
                                &[storage_type, element],
                                unit,
                                vec![storage, value],
                                ast.span,
                            ));
                        }
                    }
                    if name == "pop" && arguments.is_empty() {
                        let list = self.expression(object, None)?;
                        if let Some(element) = self.list_elements.get(&list.type_id).copied() {
                            if expected
                                .is_some_and(|expected| !self.types.assignable(element, expected))
                            {
                                return Err(semantic_error(
                                    "list element does not satisfy the expected type".into(),
                                    ast.span,
                                ));
                            }
                            let storage = self.list_storage_expression(list, ast.span);
                            let storage_type = storage.type_id;
                            let suffix = self.list_runtime_suffix(element, ast.span)?;
                            return Ok(self.runtime_call(
                                &format!("__sev_list_pop_{suffix}"),
                                &[storage_type],
                                element,
                                vec![storage],
                                ast.span,
                            ));
                        }
                    }
                    if matches!(
                        name.as_str(),
                        "appendleft" | "extend" | "popleft" | "insert" | "remove"
                            | "heap_push" | "heap_pop"
                    ) && !callable_path(callee)
                        .is_some_and(|path| self.functions.contains_key(&path))
                    {
                        let list = self.expression(object, None)?;
                        if let Some(element) = self.list_elements.get(&list.type_id).copied() {
                            let suffix = self.list_runtime_suffix(element, ast.span)?;
                            let storage = self.list_storage_expression(list, ast.span);
                            let storage_type = storage.type_id;
                            let integer = self
                                .types
                                .resolve_name("int")
                                .expect("bootstrap defines int");
                            let unit = self
                                .types
                                .resolve_name("unit")
                                .expect("bootstrap defines unit");
                            let (symbol, parameters, result, mut values) = match name.as_str() {
                                "appendleft" | "heap_push" if arguments.len() == 1 => {
                                    let value =
                                        self.expression(&arguments[0].value, Some(element))?;
                                    (
                                        format!("__sev_list_{name}_{suffix}"),
                                        vec![storage_type, element],
                                        unit,
                                        vec![storage, value],
                                    )
                                }
                                "extend" if arguments.len() == 1 => {
                                    let list_type = self.instantiate_list_type(element);
                                    let other = self
                                        .expression(&arguments[0].value, Some(list_type))?;
                                    let other = self.list_storage_expression(other, ast.span);
                                    (
                                        "__sev_list_extend".into(),
                                        vec![storage_type, storage_type],
                                        unit,
                                        vec![storage, other],
                                    )
                                }
                                "popleft" | "heap_pop" if arguments.is_empty() => (
                                    format!("__sev_list_{name}_{suffix}"),
                                    vec![storage_type],
                                    element,
                                    vec![storage],
                                ),
                                "insert" if arguments.len() == 2 => {
                                    let index =
                                        self.expression(&arguments[0].value, Some(integer))?;
                                    let value =
                                        self.expression(&arguments[1].value, Some(element))?;
                                    (
                                        format!("__sev_list_insert_{suffix}"),
                                        vec![storage_type, integer, element],
                                        unit,
                                        vec![storage, index, value],
                                    )
                                }
                                "remove" if arguments.len() == 1 => {
                                    let value =
                                        self.expression(&arguments[0].value, Some(element))?;
                                    (
                                        format!("__sev_list_remove_{suffix}"),
                                        vec![storage_type, element],
                                        unit,
                                        vec![storage, value],
                                    )
                                }
                                _ => {
                                    return Err(Diagnostic::new(
                                        "E000206",
                                        format!("list method `{name}` received incompatible arguments"),
                                        Some(ast.span),
                                    ));
                                }
                            };
                            return Ok(self.runtime_call(
                                &symbol,
                                &parameters,
                                result,
                                std::mem::take(&mut values),
                                ast.span,
                            ));
                        }
                    }
                    if name == "pop" && arguments.len() == 1 {
                        let list = self.expression(object, None)?;
                        if let Some(element) = self.list_elements.get(&list.type_id).copied() {
                            let integer = self
                                .types
                                .resolve_name("int")
                                .expect("bootstrap defines int");
                            let index = self.expression(&arguments[0].value, Some(integer))?;
                            let suffix = self.list_runtime_suffix(element, ast.span)?;
                            let storage = self.list_storage_expression(list, ast.span);
                            let storage_type = storage.type_id;
                            return Ok(self.runtime_call(
                                &format!("__sev_list_pop_at_{suffix}"),
                                &[storage_type, integer],
                                element,
                                vec![storage, index],
                                ast.span,
                            ));
                        }
                    }
                }
                if let Some(call) = self.callable_call(callee, arguments, expected, ast.span)? {
                    return Ok(call);
                }
                if self.source_call_has_callable_parameter(ast) {
                    if let Some(inlined) = self.inline_source_call(ast, expected)? {
                        return Ok(inlined);
                    }
                }
                if let Some(builder) = self.class_builder_expression(ast, expected)? {
                    return Ok(builder);
                }
                if callable_path(callee).as_deref() == Some("Error") {
                    let [argument] = arguments.as_slice() else {
                        return Err(Diagnostic::new(
                            "E000206",
                            "`Error` expects exactly one message",
                            Some(ast.span),
                        ));
                    };
                    let string = self
                        .types
                        .resolve_name("string")
                        .expect("bootstrap defines string");
                    let error_type = self
                        .types
                        .resolve_name("Error")
                        .expect("bootstrap defines Error");
                    let message = self.expression(&argument.value, Some(string))?;
                    let function = Expression {
                        id: self.next_id(),
                        type_id: string,
                        kind: ExpressionKind::Literal(LiteralValue::String(
                            self.active_function_name
                                .clone()
                                .unwrap_or_else(|| "<module>".into()),
                        )),
                        span: ast.span,
                    };
                    return Ok(self.runtime_call(
                        "__sev_error_create",
                        &[string, string],
                        error_type,
                        vec![message, function],
                        ast.span,
                    ));
                }
                if callable_path(callee).as_deref() == Some("approximate") {
                    return self.approximate_call(arguments, expected, ast.span);
                }
                if callable_path(callee).as_deref() == Some("string") {
                    let [argument] = arguments.as_slice() else {
                        return Err(Diagnostic::new(
                            "E000206",
                            "`string` expects exactly one value",
                            Some(ast.span),
                        ));
                    };
                    let value = self.expression(&argument.value, None)?;
                    return self.display_string(value, ast.span);
                }
                if let Some(mocked) = self.mocked_call(ast, expected)? {
                    return Ok(mocked);
                }
                if !self.mocks.is_empty() {
                    if let Some(inlined) = self.inline_source_call(ast, expected)? {
                        return Ok(inlined);
                    }
                }
                if let [argument] = arguments.as_slice() {
                    let runtime_type = match callable_path(callee).as_deref() {
                        Some("runtime_string") => self.types.resolve_name("string"),
                        Some("runtime_int") => self.types.resolve_name("int"),
                        _ => None,
                    };
                    if let Some(runtime_type) = runtime_type {
                        if expected
                            .is_some_and(|expected| !self.types.assignable(runtime_type, expected))
                        {
                            return Err(semantic_error(
                                "runtime value does not satisfy the expected type".into(),
                                ast.span,
                            ));
                        }
                        return self.expression(&argument.value, Some(runtime_type));
                    }
                }
                if arguments.len() == 1 {
                    if let Some(name) = callable_path(callee) {
                        if let Some(target) = self.types.resolve_name(&name) {
                            if self.numeric_primitive(target) {
                                let operand = self.expression(&arguments[0].value, None)?;
                                return self.coerce(operand, target, true);
                            }
                        }
                    }
                }
                if let AstExpressionKind::Member { object, name } = &callee.kind {
                    if name == "default" && arguments.len() == 1 && arguments[0].name.is_none() {
                        let result = self.expression(object, None)?;
                        if let Some(fallible) = self.fallible_types.get(&result.type_id).copied() {
                            let boolean = self
                                .types
                                .resolve_name("bool")
                                .expect("bootstrap defines bool");
                            let condition = Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Field {
                                    object: Box::new(result.clone()),
                                    index: 0,
                                },
                                span: ast.span,
                            };
                            let value = Expression {
                                id: self.next_id(),
                                type_id: fallible.success,
                                kind: ExpressionKind::Field {
                                    object: Box::new(result),
                                    index: 1,
                                },
                                span: ast.span,
                            };
                            let fallback =
                                self.expression(&arguments[0].value, Some(fallible.success))?;
                            return Ok(Expression {
                                id: self.next_id(),
                                type_id: fallible.success,
                                kind: ExpressionKind::Fallback {
                                    condition: Box::new(condition),
                                    value: Box::new(value),
                                    fallback: Box::new(fallback),
                                },
                                span: ast.span,
                            });
                        }
                    }
                    if name == "zero" && arguments.is_empty() {
                        if let AstExpressionKind::Name(type_name) = &object.kind {
                            if let Some(type_id) = self
                                .types
                                .resolve_name(type_name)
                                .or_else(|| self.active_type_aliases.get(type_name).copied())
                            {
                                if self.numeric_primitive(type_id) {
                                    return self.default_expression(type_id, ast.span);
                                }
                            }
                        }
                    }
                    if arguments.len() == 1 && arguments[0].name.is_none() {
                        if let Some(operator) = primitive_method_operator(name) {
                            if let Ok(left) = self.expression(object, None) {
                                if self.types.supports_binary(operator, left.type_id) {
                                    let right =
                                        self.expression(&arguments[0].value, Some(left.type_id))?;
                                    let result = if matches!(
                                        operator,
                                        BinaryOperator::Equal
                                            | BinaryOperator::NotEqual
                                            | BinaryOperator::Less
                                            | BinaryOperator::LessEqual
                                            | BinaryOperator::Greater
                                            | BinaryOperator::GreaterEqual
                                            | BinaryOperator::Contains
                                    ) {
                                        self.types
                                            .resolve_name("bool")
                                            .expect("bootstrap defines bool")
                                    } else {
                                        left.type_id
                                    };
                                    if expected.is_some_and(|expected| {
                                        !self.types.assignable(result, expected)
                                    }) {
                                        return Err(semantic_error(
                                            "method result does not satisfy the expected type"
                                                .into(),
                                            ast.span,
                                        ));
                                    }
                                    return Ok(Expression {
                                        id: self.next_id(),
                                        type_id: result,
                                        kind: ExpressionKind::Binary {
                                            operator,
                                            left: Box::new(left),
                                            right: Box::new(right),
                                        },
                                        span: ast.span,
                                    });
                                }
                            }
                        }
                    }
                }
                if let AstExpressionKind::Member { object, name } = &callee.kind {
                    if name == "send" && arguments.len() == 1 {
                        let channel = self.expression(object, None)?;
                        if let Some(element) = self.channel_elements.get(&channel.type_id).copied()
                        {
                            let value = self.expression(&arguments[0].value, Some(element))?;
                            let storage = self.channel_storage_expression(channel, ast.span);
                            let storage_type = storage.type_id;
                            let suffix = self.list_runtime_suffix(element, ast.span)?;
                            let unit = self
                                .types
                                .resolve_name("unit")
                                .expect("bootstrap defines unit");
                            return Ok(self.runtime_call(
                                &format!("__sev_channel_send_{suffix}"),
                                &[storage_type, element],
                                unit,
                                vec![storage, value],
                                ast.span,
                            ));
                        }
                    }
                    if name == "size" && arguments.is_empty() {
                        let value = self.expression(object, None)?;
                        if self.list_elements.contains_key(&value.type_id) {
                            let storage = self.list_storage_expression(value, ast.span);
                            let storage_type = storage.type_id;
                            let result = self
                                .types
                                .resolve_name("usize")
                                .expect("bootstrap defines usize");
                            return Ok(self.runtime_call(
                                "__sev_list_len",
                                &[storage_type],
                                result,
                                vec![storage],
                                ast.span,
                            ));
                        }
                    }
                }
                if let AstExpressionKind::TypeApplication {
                    callee: application,
                    arguments: type_arguments,
                } = &callee.kind
                {
                    if matches!(&application.kind, AstExpressionKind::Name(name) if matches!(name.as_str(), "channel" | "Channel"))
                        && type_arguments.len() == 1
                    {
                        return self.channel_constructor(
                            &type_arguments[0],
                            arguments.first().map(|argument| &argument.value),
                            ast.span,
                        );
                    }
                }
                if let Some(path) = callable_path(callee) {
                    if self.enum_variants.contains_key(&path) {
                        return self.enum_constructor(&path, arguments, expected, ast.span);
                    }
                    if self.enums.get(&path).is_some_and(|instance| {
                        instance
                            .variants
                            .iter()
                            .any(|variant| !variant.accepted_values.is_empty())
                    }) {
                        return self.enum_value_constructor(
                            &path,
                            arguments,
                            expected,
                            ast.span,
                        );
                    }
                    if self
                        .class_instances
                        .contains_key(&(path.clone(), Vec::new()))
                    {
                        return self.class_constructor(
                            &path,
                            &[],
                            arguments,
                            expected,
                            ast.span,
                        );
                    }
                }
                if let Some((class, type_arguments)) = class_application(callee) {
                    return self.class_constructor(
                        class,
                        type_arguments,
                        arguments,
                        expected,
                        ast.span,
                    );
                }
                if let AstExpressionKind::Name(class) = &callee.kind {
                    if self
                        .classes
                        .get(class)
                        .is_some_and(|declaration| !declaration.type_parameters.is_empty())
                    {
                        return self
                            .inferred_class_constructor(class, arguments, expected, ast.span);
                    }
                    if self.classes.contains_key(class) {
                        return self.class_constructor(class, &[], arguments, expected, ast.span);
                    }
                }
                if matches!(callable_path(callee).as_deref(), Some("size" | "len"))
                    && arguments.len() == 1
                {
                    let value = self.expression(&arguments[0].value, None)?;
                    if self.list_elements.contains_key(&value.type_id) {
                        let storage = self.list_storage_expression(value, ast.span);
                        let storage_type = storage.type_id;
                        let result = self
                            .types
                            .resolve_name("usize")
                            .expect("bootstrap defines usize");
                        return Ok(self.runtime_call(
                            "__sev_list_len",
                            &[storage_type],
                            result,
                            vec![storage],
                            ast.span,
                        ));
                    }
                    if self.map_elements.contains_key(&value.type_id) {
                        let storage = self.collection_storage_expression(value, 0, ast.span);
                        let storage_type = storage.type_id;
                        let result = self
                            .types
                            .resolve_name("usize")
                            .expect("bootstrap defines usize");
                        return Ok(self.runtime_call(
                            "__sev_list_len",
                            &[storage_type],
                            result,
                            vec![storage],
                            ast.span,
                        ));
                    }
                    let string = self
                        .types
                        .resolve_name("string")
                        .expect("bootstrap defines string");
                    if value.type_id == string {
                        let result = self
                            .types
                            .resolve_name("usize")
                            .expect("bootstrap defines usize");
                        return Ok(self.runtime_call(
                            "__sev_string_length",
                            &[string],
                            result,
                            vec![value],
                            ast.span,
                        ));
                    }
                }
                if callable_path(callee).as_deref() == Some("print")
                    && !arguments.is_empty()
                    && arguments.iter().all(|argument| argument.name.is_none())
                    && arguments.len() > 1 {
                        let values = arguments
                            .iter()
                            .map(|argument| self.expression(&argument.value, None))
                            .collect::<Result<Vec<_>, Diagnostic>>()?;
                        let string = self
                            .types
                            .resolve_name("string")
                            .expect("bootstrap defines string");
                        let mut values = values.into_iter();
                        let first = values.next().expect("print has multiple arguments");
                        let mut rendered = self.display_string(first, ast.span)?;
                        for value in values {
                            let space = Expression {
                                id: self.next_id(),
                                type_id: string,
                                kind: ExpressionKind::Literal(LiteralValue::String(" ".into())),
                                span: ast.span,
                            };
                            rendered = self.runtime_call(
                                "__sev_string_concat",
                                &[string, string],
                                string,
                                vec![rendered, space],
                                ast.span,
                            );
                            let value = self.display_string(value, ast.span)?;
                            rendered = self.runtime_call(
                                "__sev_string_concat",
                                &[string, string],
                                string,
                                vec![rendered, value],
                                ast.span,
                            );
                        }
                        let result = self
                            .types
                            .resolve_name("i32")
                            .expect("bootstrap defines i32");
                        return Ok(self.runtime_call(
                            "__sev_print_string",
                            &[string],
                            result,
                            vec![rendered],
                            ast.span,
                        ));
                    }
                if callable_path(callee).as_deref() == Some("print")
                    && arguments.len() == 1
                    && arguments[0].name.is_none()
                {
                    let value = self.expression(&arguments[0].value, None)?;
                    let rendered = self.display_string(value, ast.span)?;
                    let string = rendered.type_id;
                    let result = self
                        .types
                        .resolve_name("i32")
                        .expect("bootstrap defines i32");
                    return Ok(self.runtime_call(
                        "__sev_print_string",
                        &[string],
                        result,
                        vec![rendered],
                        ast.span,
                    ));
                }
                if let Some(call) =
                    self.trait_namespace_call(callee, arguments, expected, ast.span)?
                {
                    return Ok(call);
                }
                if let Some(method) =
                    self.class_method_call(callee, arguments, expected, ast.span)?
                {
                    return Ok(method);
                }
                if matches!(callee.kind, AstExpressionKind::Member { .. })
                    && !callable_path(callee).is_some_and(|path| self.functions.contains_key(&path))
                {
                    return Err(Diagnostic::new(
                        "E000211",
                        "class method is unknown or cannot be used as a value",
                        Some(callee.span),
                    ));
                }
                let Some(name) = callable_path(callee) else {
                    return Err(Diagnostic::new(
                        "E000206",
                        "call target must resolve to a function declaration",
                        Some(callee.span),
                    ));
                };
                let candidates = self.functions.get(&name).cloned().unwrap_or_default();
                let mut matches = Vec::new();
                for function in candidates {
                    let signature = self.signatures[&function].clone();
                    let exposed_result = self
                        .fallible_types
                        .get(&signature.result)
                        .map_or(signature.result, |fallible| fallible.success);
                    if expected
                        .is_some_and(|expected| !self.types.assignable(exposed_result, expected))
                    {
                        continue;
                    }
                    let mut ordered = vec![None; signature.parameters.len()];
                    let mut positional = 0usize;
                    let mut named = false;
                    let mut valid = true;
                    for argument in arguments {
                        let index = if let Some(argument_name) = &argument.name {
                            named = true;
                            signature
                                .parameters
                                .iter()
                                .position(|parameter| parameter.name == *argument_name)
                        } else if named {
                            None
                        } else {
                            let index = positional;
                            positional += 1;
                            (index < ordered.len()).then_some(index)
                        };
                        let Some(index) = index else {
                            valid = false;
                            break;
                        };
                        if ordered[index].replace(&argument.value).is_some() {
                            valid = false;
                            break;
                        }
                    }
                    if !valid {
                        continue;
                    }
                    let ordered = ordered
                        .into_iter()
                        .zip(&signature.parameters)
                        .map(|(argument, parameter)| argument.or(parameter.default.as_ref()))
                        .collect::<Option<Vec<_>>>();
                    let Some(ordered) = ordered else {
                        continue;
                    };
                    let resolved = ordered
                        .iter()
                        .zip(&signature.parameters)
                        .map(|(argument, parameter)| {
                            if matches!(
                                argument.kind,
                                AstExpressionKind::List(_)
                                    | AstExpressionKind::Set(_)
                                    | AstExpressionKind::Map(_)
                            ) {
                                return self.expression(argument, Some(parameter.type_id));
                            }
                            match self.prepare(argument) {
                                Ok(prepared) => self.finish(prepared, parameter.type_id),
                                Err(_) => self.expression(argument, Some(parameter.type_id)),
                            }
                        })
                        .collect::<Result<Vec<_>, _>>();
                    if let Ok(arguments) = resolved {
                        let conversions = arguments
                            .iter()
                            .zip(&signature.parameters)
                            .map(|(argument, parameter)| {
                                if matches!(
                                    argument.kind,
                                    ExpressionKind::Aggregate { class, .. }
                                        if class == parameter.type_id
                                            && self.union_types.contains_key(&class)
                                ) {
                                    Some(ConversionRank::General)
                                } else {
                                    expression_conversion_rank(
                                        self.types,
                                        argument,
                                        parameter.type_id,
                                    )
                                }
                            })
                            .collect::<Option<Vec<_>>>();
                        let Some(conversions) = conversions else {
                            continue;
                        };
                        matches.push((
                            conversions,
                            self.function_specificity[&function],
                            function,
                            signature.result,
                            arguments,
                        ));
                    }
                }
                let best = matches
                    .iter()
                    .enumerate()
                    .filter(|(index, candidate)| {
                        !matches.iter().enumerate().any(|(other_index, other)| {
                            other_index != *index && dominates(&other.0, &candidate.0)
                        })
                    })
                    .map(|(_, candidate)| candidate)
                    .collect::<Vec<_>>();
                let specificity = best.iter().map(|candidate| candidate.1).min();
                let best = best
                    .into_iter()
                    .filter(|candidate| Some(candidate.1) == specificity)
                    .collect::<Vec<_>>();
                let [(_, _, function, result, arguments)] = best.as_slice() else {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("call to `{name}` has no unique compatible declaration"),
                        Some(ast.span),
                    ));
                };
                let arguments =
                    self.apply_parameter_effects(*function, (*arguments).clone(), ast.span);
                let call = Expression {
                    id: self.next_id(),
                    type_id: *result,
                    kind: ExpressionKind::Call {
                        callee: severian_hir::Callee::Direct {
                            function: self.function_definitions[function],
                            substitution: self.function_substitutions[function].clone(),
                        },
                        arguments,
                    },
                    span: ast.span,
                };
                if self.is_error_type(*result) {
                    let unit = self
                        .types
                        .resolve_name("unit")
                        .expect("bootstrap defines unit");
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: expected.unwrap_or(unit),
                        kind: ExpressionKind::Throw(Box::new(call)),
                        span: ast.span,
                    });
                }
                if self.preserve_error_depth == 0 {
                    if let Some(fallible) = self.fallible_types.get(result).copied() {
                        return Ok(self.unwrap_fallible_expression(call, fallible, ast.span));
                    }
                }
                Ok(call)
            }
            AstExpressionKind::Unary { operator, operand } => {
                if *operator == AstUnaryOperator::AddressOf {
                    if self.unsafe_depth == 0 {
                        return Err(Diagnostic::new(
                            "E000219",
                            "raw addresses require an `unsafe` scope",
                            Some(ast.span),
                        ));
                    }
                    let AstExpressionKind::Index { object, index } = &operand.kind else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "raw address-of currently requires an indexed list element",
                            Some(operand.span),
                        ));
                    };
                    let collection = self.expression(object, None)?;
                    let Some(element) = self.list_elements.get(&collection.type_id).copied()
                    else {
                        return Err(Diagnostic::new(
                            "E000211",
                            "raw address-of requires a list element",
                            Some(operand.span),
                        ));
                    };
                    let pointer = self.instantiate_pointer_type(element);
                    if expected.is_some_and(|expected| expected != pointer) {
                        return Err(semantic_error(
                            "raw address does not satisfy the expected pointer type".into(),
                            ast.span,
                        ));
                    }
                    let index = self.expression(index, None)?;
                    let index_name = self
                        .types
                        .definition(index.type_id)
                        .map(|definition| definition.name.as_str());
                    if !matches!(index_name, Some("int" | "i64" | "usize")) {
                        return Err(Diagnostic::new(
                            "E000211",
                            "pointer offsets must be integers",
                            Some(index.span),
                        ));
                    }
                    let storage = self.list_storage_expression(collection, ast.span);
                    let storage_type = storage.type_id;
                    return Ok(self.runtime_call(
                        "__sev_list_address",
                        &[storage_type, index.type_id],
                        pointer,
                        vec![storage, index],
                        ast.span,
                    ));
                }
                if matches!(
                    operator,
                    AstUnaryOperator::Borrow | AstUnaryOperator::BorrowMut
                ) {
                    let operand = self.expression(operand, expected)?;
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: operand.type_id,
                        kind: ExpressionKind::Borrow {
                            operand: Box::new(operand),
                            exclusive: *operator == AstUnaryOperator::BorrowMut,
                        },
                        span: ast.span,
                    });
                }
                if *operator == AstUnaryOperator::Copy {
                    let operand = self.expression(operand, expected)?;
                    let Some(element) = self.list_elements.get(&operand.type_id).copied() else {
                        return Ok(operand);
                    };
                    let list_type = operand.type_id;
                    let storage = self.list_storage_expression(operand, ast.span);
                    let storage_type = storage.type_id;
                    let suffix = self.list_runtime_suffix(element, ast.span)?;
                    let copied = self.runtime_call(
                        &format!("__sev_list_copy_{suffix}"),
                        &[storage_type],
                        storage_type,
                        vec![storage],
                        ast.span,
                    );
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: list_type,
                        kind: ExpressionKind::Aggregate {
                            class: list_type,
                            fields: vec![copied],
                        },
                        span: ast.span,
                    });
                }
                if *operator == AstUnaryOperator::Move {
                    let operand = self.expression(operand, expected)?;
                    return Ok(Expression {
                        id: self.next_id(),
                        type_id: operand.type_id,
                        kind: ExpressionKind::Move(Box::new(operand)),
                        span: ast.span,
                    });
                }
                if *operator == AstUnaryOperator::Not {
                    let operand = self.expression(operand, None)?;
                    if self
                        .types
                        .definition(operand.type_id)
                        .is_some_and(|definition| definition.name == "string")
                    {
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        return Ok(self.runtime_call(
                            "__sev_string_is_empty",
                            &[operand.type_id],
                            boolean,
                            vec![operand],
                            ast.span,
                        ));
                    }
                }
                let operator = universal_unary(*operator);
                let prepared = self.prepare(operand)?;
                let resolved = self
                    .types
                    .resolve_unary(operator, prepared.constraint(), expected)
                    .map_err(|error| semantic_error(error.to_string(), ast.span))?;
                let operand = self.finish(prepared, resolved.operand)?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id: resolved.result,
                    kind: ExpressionKind::Unary {
                        operator,
                        operand: Box::new(operand),
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Binary {
                operator,
                left,
                right,
            } => {
                if matches!(
                    operator,
                    AstBinaryOperator::Less
                        | AstBinaryOperator::LessEqual
                        | AstBinaryOperator::Greater
                        | AstBinaryOperator::GreaterEqual
                ) {
                    if let AstExpressionKind::Binary {
                        operator: first_operator,
                        left: first_left,
                        right: middle,
                    } = &left.kind
                    {
                        if matches!(
                            first_operator,
                            AstBinaryOperator::Less
                                | AstBinaryOperator::LessEqual
                                | AstBinaryOperator::Greater
                                | AstBinaryOperator::GreaterEqual
                        ) {
                            let chained = AstExpression {
                                kind: AstExpressionKind::Binary {
                                    operator: AstBinaryOperator::And,
                                    left: Box::new(AstExpression {
                                        kind: AstExpressionKind::Binary {
                                            operator: *first_operator,
                                            left: first_left.clone(),
                                            right: middle.clone(),
                                        },
                                        span: left.span,
                                    }),
                                    right: Box::new(AstExpression {
                                        kind: AstExpressionKind::Binary {
                                            operator: *operator,
                                            left: middle.clone(),
                                            right: right.clone(),
                                        },
                                        span: ast.span,
                                    }),
                                },
                                span: ast.span,
                            };
                            return self.expression(&chained, expected);
                        }
                    }
                }
                let operator_spelling = ast_binary_spelling(*operator);
                if self
                    .active_operator_namespaces
                    .contains_key(operator_spelling)
                {
                    return self.trait_namespace_operator(
                        operator_spelling,
                        left,
                        right,
                        expected,
                        ast.span,
                    );
                }
                if matches!(
                    operator,
                    AstBinaryOperator::Equal | AstBinaryOperator::NotEqual
                ) {
                    if let AstExpressionKind::Name(type_name) = &right.kind {
                        let target = self
                            .class_instances
                            .get(&(type_name.clone(), Vec::new()))
                            .map(|instance| instance.ty)
                            .or_else(|| self.types.resolve_name(type_name));
                        if let Some(target) = target.filter(|target| self.is_error_type(*target)) {
                            let left = self.expression(left, None)?;
                            if self.is_error_type(left.type_id) {
                                let boolean = self
                                    .types
                                    .resolve_name("bool")
                                    .expect("bootstrap defines bool");
                                let equal = left.type_id == target;
                                return Ok(Expression {
                                    id: self.next_id(),
                                    type_id: boolean,
                                    kind: ExpressionKind::Literal(LiteralValue::Boolean(
                                        if *operator == AstBinaryOperator::Equal {
                                            equal
                                        } else {
                                            !equal
                                        },
                                    )),
                                    span: ast.span,
                                });
                            }
                        }
                    }
                }
                if *operator == AstBinaryOperator::Identity {
                    if let AstExpressionKind::Name(type_name) = &right.kind {
                        let target = self
                            .class_instances
                            .get(&(type_name.clone(), Vec::new()))
                            .map(|instance| instance.ty)
                            .or_else(|| self.types.resolve_name(type_name));
                        if let Some(target) = target {
                            let left = self.expression(left, None)?;
                            if let Some(fallible) = self.fallible_types.get(&left.type_id).copied()
                            {
                                let boolean = self
                                    .types
                                    .resolve_name("bool")
                                    .expect("bootstrap defines bool");
                                let ok = Expression {
                                    id: self.next_id(),
                                    type_id: boolean,
                                    kind: ExpressionKind::Field {
                                        object: Box::new(left),
                                        index: 0,
                                    },
                                    span: ast.span,
                                };
                                if target == fallible.success {
                                    return Ok(ok);
                                }
                                if target == fallible.error {
                                    return Ok(Expression {
                                        id: self.next_id(),
                                        type_id: boolean,
                                        kind: ExpressionKind::Unary {
                                            operator: UnaryOperator::Not,
                                            operand: Box::new(ok),
                                        },
                                        span: ast.span,
                                    });
                                }
                                return Ok(Expression {
                                    id: self.next_id(),
                                    type_id: boolean,
                                    kind: ExpressionKind::Literal(LiteralValue::Boolean(false)),
                                    span: ast.span,
                                });
                            }
                        }
                    }
                }
                if *operator == AstBinaryOperator::Contains {
                    let haystack = self.expression(right, None)?;
                    if let Some(element) = self.list_elements.get(&haystack.type_id).copied() {
                        let needle = self.expression(left, Some(element))?;
                        let suffix = self.list_runtime_suffix(element, ast.span)?;
                        let storage = self.list_storage_expression(haystack, ast.span);
                        let storage_type = storage.type_id;
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        return Ok(self.runtime_call(
                            &format!("__sev_list_contains_{suffix}"),
                            &[storage_type, element],
                            boolean,
                            vec![storage, needle],
                            ast.span,
                        ));
                    }
                    if self.set_type == Some(haystack.type_id) {
                        let needle = self.expression(left, None)?;
                        let suffix = self.list_runtime_suffix(needle.type_id, ast.span)?;
                        let storage = self.collection_storage_expression(haystack, 0, ast.span);
                        let storage_type = storage.type_id;
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        return Ok(self.runtime_call(
                            &format!("__sev_set_contains_{suffix}"),
                            &[storage_type, needle.type_id],
                            boolean,
                            vec![storage, needle],
                            ast.span,
                        ));
                    }
                    let string = self
                        .types
                        .resolve_name("string")
                        .expect("bootstrap defines string");
                    let needle = self.expression(left, Some(string))?;
                    let haystack = self.coerce(haystack, string, false)?;
                    let boolean = self
                        .types
                        .resolve_name("bool")
                        .expect("bootstrap defines bool");
                    return Ok(self.runtime_call(
                        "__sev_string_contains",
                        &[string, string],
                        boolean,
                        vec![haystack, needle],
                        ast.span,
                    ));
                }
                if *operator == AstBinaryOperator::Power {
                    let left = self.expression(left, None)?;
                    let name = self
                        .types
                        .definition(left.type_id)
                        .map(|definition| definition.name.as_str());
                    if matches!(name, Some("float" | "f64")) {
                        if matches!(
                            right.kind,
                            AstExpressionKind::Literal(AstLiteral::Integer(_))
                        ) {
                            let integer = self
                                .types
                                .resolve_name("int")
                                .expect("bootstrap defines int");
                            let right = self.expression(right, Some(integer))?;
                            return Ok(self.runtime_call(
                                "__sev_pow_f64_i64",
                                &[left.type_id, integer],
                                left.type_id,
                                vec![left, right],
                                ast.span,
                            ));
                        }
                        let right = self.expression(right, Some(left.type_id))?;
                        return Ok(self.runtime_call(
                            "__sev_pow_f64_f64",
                            &[left.type_id, left.type_id],
                            left.type_id,
                            vec![left, right],
                            ast.span,
                        ));
                    }
                    if matches!(name, Some("int" | "i64")) {
                        if matches!(right.kind, AstExpressionKind::Literal(AstLiteral::Float(_))) {
                            let float = self
                                .types
                                .resolve_name("float")
                                .expect("bootstrap defines float");
                            let left = self.coerce(left, float, false)?;
                            let right = self.expression(right, Some(float))?;
                            return Ok(self.runtime_call(
                                "__sev_pow_f64_f64",
                                &[float, float],
                                float,
                                vec![left, right],
                                ast.span,
                            ));
                        }
                        let right = self.expression(right, Some(left.type_id))?;
                        return Ok(self.runtime_call(
                            "__sev_pow_i64_i64",
                            &[left.type_id, left.type_id],
                            left.type_id,
                            vec![left, right],
                            ast.span,
                        ));
                    }
                }
                if matches!(
                    operator,
                    AstBinaryOperator::Equal
                        | AstBinaryOperator::NotEqual
                        | AstBinaryOperator::Identity
                ) {
                    let resolved_left = self.expression(left, None)?;
                    if matches!(
                        operator,
                        AstBinaryOperator::Equal | AstBinaryOperator::NotEqual
                    ) {
                        if let Some(fallible) =
                            self.fallible_types.get(&resolved_left.type_id).copied()
                        {
                            let boolean = self
                                .types
                                .resolve_name("bool")
                                .expect("bootstrap defines bool");
                            let ok = Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Field {
                                    object: Box::new(resolved_left.clone()),
                                    index: 0,
                                },
                                span: ast.span,
                            };
                            let value = Expression {
                                id: self.next_id(),
                                type_id: fallible.success,
                                kind: ExpressionKind::Field {
                                    object: Box::new(resolved_left),
                                    index: 1,
                                },
                                span: ast.span,
                            };
                            let right = self.expression(right, Some(fallible.success))?;
                            let equal = Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Binary {
                                    operator: BinaryOperator::Equal,
                                    left: Box::new(value),
                                    right: Box::new(right),
                                },
                                span: ast.span,
                            };
                            let successful_equal = Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Binary {
                                    operator: BinaryOperator::And,
                                    left: Box::new(ok),
                                    right: Box::new(equal),
                                },
                                span: ast.span,
                            };
                            if *operator == AstBinaryOperator::NotEqual {
                                return Ok(Expression {
                                    id: self.next_id(),
                                    type_id: boolean,
                                    kind: ExpressionKind::Unary {
                                        operator: UnaryOperator::Not,
                                        operand: Box::new(successful_equal),
                                    },
                                    span: ast.span,
                                });
                            }
                            return Ok(successful_equal);
                        }
                        if let Some(elements) =
                            self.tuple_elements.get(&resolved_left.type_id).cloned()
                        {
                            let resolved_right =
                                self.expression(right, Some(resolved_left.type_id))?;
                            let boolean = self
                                .types
                                .resolve_name("bool")
                                .expect("bootstrap defines bool");
                            let mut comparison = Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
                                span: ast.span,
                            };
                            for (index, element) in elements.into_iter().enumerate() {
                                let left_field = Expression {
                                    id: self.next_id(),
                                    type_id: element,
                                    kind: ExpressionKind::Field {
                                        object: Box::new(resolved_left.clone()),
                                        index: index as u32,
                                    },
                                    span: ast.span,
                                };
                                let right_field = Expression {
                                    id: self.next_id(),
                                    type_id: element,
                                    kind: ExpressionKind::Field {
                                        object: Box::new(resolved_right.clone()),
                                        index: index as u32,
                                    },
                                    span: ast.span,
                                };
                                let equal = Expression {
                                    id: self.next_id(),
                                    type_id: boolean,
                                    kind: ExpressionKind::Binary {
                                        operator: BinaryOperator::Equal,
                                        left: Box::new(left_field),
                                        right: Box::new(right_field),
                                    },
                                    span: ast.span,
                                };
                                comparison = Expression {
                                    id: self.next_id(),
                                    type_id: boolean,
                                    kind: ExpressionKind::Binary {
                                        operator: BinaryOperator::And,
                                        left: Box::new(comparison),
                                        right: Box::new(equal),
                                    },
                                    span: ast.span,
                                };
                            }
                            if *operator == AstBinaryOperator::NotEqual {
                                return Ok(Expression {
                                    id: self.next_id(),
                                    type_id: boolean,
                                    kind: ExpressionKind::Unary {
                                        operator: UnaryOperator::Not,
                                        operand: Box::new(comparison),
                                    },
                                    span: ast.span,
                                });
                            }
                            return Ok(comparison);
                        }
                        if self
                            .class_instances_by_type
                            .contains_key(&resolved_left.type_id)
                        {
                            let resolved_right =
                                self.expression(right, Some(resolved_left.type_id))?;
                            let comparison = self.structural_class_equality(
                                resolved_left,
                                resolved_right,
                                ast.span,
                                &mut BTreeSet::new(),
                            )?;
                            if *operator == AstBinaryOperator::NotEqual {
                                return Ok(Expression {
                                    id: self.next_id(),
                                    type_id: comparison.type_id,
                                    kind: ExpressionKind::Unary {
                                        operator: UnaryOperator::Not,
                                        operand: Box::new(comparison),
                                    },
                                    span: ast.span,
                                });
                            }
                            return Ok(comparison);
                        }
                    }
                    if let Some(element) = self.list_elements.get(&resolved_left.type_id).copied() {
                        let list_type = resolved_left.type_id;
                        let resolved_right = self.expression(right, Some(list_type))?;
                        let left_storage = self.list_storage_expression(resolved_left, ast.span);
                        let right_storage = self.list_storage_expression(resolved_right, ast.span);
                        let storage_type = left_storage.type_id;
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        let symbol = if *operator == AstBinaryOperator::Identity {
                            "__sev_list_identity".to_owned()
                        } else {
                            format!(
                                "__sev_list_equal_{}",
                                self.list_runtime_suffix(element, ast.span)?
                            )
                        };
                        let comparison = self.runtime_call(
                            &symbol,
                            &[storage_type, storage_type],
                            boolean,
                            vec![left_storage, right_storage],
                            ast.span,
                        );
                        if *operator == AstBinaryOperator::NotEqual {
                            return Ok(Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Unary {
                                    operator: UnaryOperator::Not,
                                    operand: Box::new(comparison),
                                },
                                span: ast.span,
                            });
                        }
                        return Ok(comparison);
                    }
                    if self.set_type == Some(resolved_left.type_id) {
                        let set_type = resolved_left.type_id;
                        let resolved_right = self.expression(right, Some(set_type))?;
                        let left_storage =
                            self.collection_storage_expression(resolved_left, 0, ast.span);
                        let right_storage =
                            self.collection_storage_expression(resolved_right, 0, ast.span);
                        let storage_type = left_storage.type_id;
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        let element = self.set_element.ok_or_else(|| {
                            Diagnostic::new(
                                "E000211",
                                "set element type is unknown",
                                Some(ast.span),
                            )
                        })?;
                        let suffix = self.list_runtime_suffix(element, ast.span)?;
                        let comparison = self.runtime_call(
                            &format!("__sev_set_equal_{suffix}"),
                            &[storage_type, storage_type],
                            boolean,
                            vec![left_storage, right_storage],
                            ast.span,
                        );
                        if *operator == AstBinaryOperator::NotEqual {
                            return Ok(Expression {
                                id: self.next_id(),
                                type_id: boolean,
                                kind: ExpressionKind::Unary {
                                    operator: UnaryOperator::Not,
                                    operand: Box::new(comparison),
                                },
                                span: ast.span,
                            });
                        }
                        return Ok(comparison);
                    }
                    if *operator == AstBinaryOperator::Identity {
                        let resolved_right = self.expression(right, Some(resolved_left.type_id))?;
                        return Ok(Expression {
                            id: self.next_id(),
                            type_id: self
                                .types
                                .resolve_name("bool")
                                .expect("bootstrap defines bool"),
                            kind: ExpressionKind::Binary {
                                operator: BinaryOperator::Equal,
                                left: Box::new(resolved_left),
                                right: Box::new(resolved_right),
                            },
                            span: ast.span,
                        });
                    }
                }
                if matches!(
                    operator,
                    AstBinaryOperator::Less
                        | AstBinaryOperator::LessEqual
                        | AstBinaryOperator::Greater
                        | AstBinaryOperator::GreaterEqual
                ) {
                    let left_value = self.expression(left, None)?;
                    let right_value = self.expression(right, None)?;
                    if left_value.type_id != right_value.type_id
                        && self.integer_primitive(left_value.type_id)
                        && self.integer_primitive(right_value.type_id)
                    {
                        let right_value = self.coerce(right_value, left_value.type_id, true)?;
                        let boolean = self
                            .types
                            .resolve_name("bool")
                            .expect("bootstrap defines bool");
                        return Ok(Expression {
                            id: self.next_id(),
                            type_id: boolean,
                            kind: ExpressionKind::Binary {
                                operator: universal_binary(*operator),
                                left: Box::new(left_value),
                                right: Box::new(right_value),
                            },
                            span: ast.span,
                        });
                    }
                }
                if *operator == AstBinaryOperator::Remainder {
                    let left = self.expression(left, None)?;
                    if self.types.primitive(left.type_id).is_some_and(|primitive| {
                        primitive.category == severian_universal::PrimitiveCategory::Measured
                    }) {
                        let right = self.expression(right, Some(left.type_id))?;
                        return Ok(Expression {
                            id: self.next_id(),
                            type_id: left.type_id,
                            kind: ExpressionKind::Binary {
                                operator: BinaryOperator::Remainder,
                                left: Box::new(left),
                                right: Box::new(right),
                            },
                            span: ast.span,
                        });
                    }
                }
                let operator = universal_binary(*operator);
                // Both operands remain constraints until a single signature is
                // selected; neither side gets an early default literal type.
                let left = self.prepare(left)?;
                let right = self.prepare(right)?;
                let left_constraint = left.constraint();
                let right_constraint = right.constraint();
                let resolved = self
                    .types
                    .resolve_binary(operator, left_constraint, right_constraint, expected)
                    .map_err(|error| {
                        self.binary_operator_error(
                            error,
                            operator,
                            left_constraint,
                            right_constraint,
                            ast.span,
                        )
                    })?;
                let left = self.finish(left, resolved.left)?;
                let right = self.finish(right, resolved.right)?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id: resolved.result,
                    kind: ExpressionKind::Binary {
                        operator,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span: ast.span,
                })
            }
        }
    }

    fn binary_operator_error(
        &self,
        error: TypeError,
        operator: BinaryOperator,
        left: TypeConstraint,
        right: TypeConstraint,
        span: severian_source::Span,
    ) -> Diagnostic {
        if error == TypeError::NoMatchingOperator(operator) {
            let left = self.constraint_name(left);
            let right = self.constraint_name(right);
            return Diagnostic::new(
                "E000202",
                format!("no `{operator}` operator accepts `{left}` and `{right}`"),
                Some(span),
            )
            .with_label(span, "these operands have no compatible operator overload")
            .with_note(format!("left operand: `{left}`; right operand: `{right}`"))
            .with_help("convert one operand explicitly or use operands with compatible types");
        }
        semantic_error(error.to_string(), span)
    }

    fn numeric_primitive(&self, ty: TypeId) -> bool {
        self.types.primitive(ty).is_some_and(|primitive| {
            matches!(
                primitive.category,
                severian_universal::PrimitiveCategory::Integer
                    | severian_universal::PrimitiveCategory::Float
                    | severian_universal::PrimitiveCategory::Measured
            )
        })
    }

    fn integer_primitive(&self, ty: TypeId) -> bool {
        self.types.primitive(ty).is_some_and(|primitive| {
            primitive.category == severian_universal::PrimitiveCategory::Integer
        })
    }

    fn tag_type(&self) -> TypeId {
        ["int", "u32", "i32", "u64", "i64", "usize"]
            .into_iter()
            .find_map(|name| self.types.resolve_name(name))
            .expect("a runtime union requires an integer representation type")
    }

    fn coerce(
        &mut self,
        expression: Expression,
        expected: TypeId,
        explicit: bool,
    ) -> Result<Expression, Diagnostic> {
        if expression.type_id == expected {
            return Ok(expression);
        }
        if let Some(members) = self.union_types.get(&expected).cloned() {
            if !explicit && members.contains(&expression.type_id) {
                return self.union_expression(expected, &members, expression);
            }
        }
        if explicit {
            if let Some(members) = self.union_types.get(&expression.type_id).cloned() {
                return self.convert_union_expression(expression, &members, expected);
            }
            if self.types.resolve_name("string") == Some(expression.type_id)
                && matches!(
                    self.types
                        .definition(expected)
                        .map(|definition| definition.name.as_str()),
                    Some("float" | "f64")
                )
            {
                let string = expression.type_id;
                let span = expression.span;
                return Ok(self.runtime_call(
                    "__sev_float_from_string",
                    &[string],
                    expected,
                    vec![expression],
                    span,
                ));
            }
        }
        let numeric =
            self.numeric_primitive(expression.type_id) && self.numeric_primitive(expected);
        if !numeric {
            if !explicit && self.types.assignable(expression.type_id, expected) {
                return Ok(expression);
            }
            return Err(semantic_error(
                if explicit {
                    "explicit conversion requires numeric primitive types".into()
                } else {
                    "expression does not satisfy the expected type".into()
                },
                expression.span,
            ));
        }
        if !explicit && !self.types.assignable(expression.type_id, expected) {
            return Err(semantic_error(
                "expression does not satisfy the expected type".into(),
                expression.span,
            ));
        }
        let from = expression.type_id;
        let span = expression.span;
        Ok(Expression {
            id: self.next_id(),
            type_id: expected,
            kind: ExpressionKind::Convert {
                operand: Box::new(expression),
                conversion: severian_hir::Conversion {
                    from,
                    to: expected,
                    kind: if explicit {
                        severian_hir::ConversionKind::NumericCast
                    } else {
                        severian_hir::ConversionKind::NumericWidening
                    },
                },
            },
            span,
        })
    }

    fn constraint_name(&self, constraint: TypeConstraint) -> String {
        match constraint {
            TypeConstraint::Known(ty) => self
                .types
                .definition(ty)
                .map_or_else(|| format!("{ty:?}"), |definition| definition.name.clone()),
            TypeConstraint::Literal(kind) => format!("{kind:?} literal").to_lowercase(),
        }
    }

    fn class_builder_expression(
        &mut self,
        ast: &AstExpression,
        expected: Option<TypeId>,
    ) -> Result<Option<Expression>, Diagnostic> {
        let mut current = ast;
        let mut updates = Vec::new();
        while let AstExpressionKind::Call { callee, arguments } = &current.kind {
            let AstExpressionKind::Member { object, name } = &callee.kind else {
                break;
            };
            if name != "set" || arguments.len() != 2 {
                break;
            }
            updates.push((arguments[0].value.clone(), arguments[1].value.clone()));
            current = object;
        }
        if updates.is_empty() {
            return Ok(None);
        }
        let AstExpressionKind::Call { callee, .. } = &current.kind else {
            return Ok(None);
        };
        let AstExpressionKind::Name(class_name) = &callee.kind else {
            return Ok(None);
        };
        if !self.classes.contains_key(class_name) {
            return Ok(None);
        }
        let object = self.expression(current, expected)?;
        let Some(instance) = self.class_instances_by_type.get(&object.type_id).cloned() else {
            return Ok(None);
        };
        let ExpressionKind::Aggregate { class, mut fields } = object.kind else {
            return Ok(None);
        };
        for (key, value) in updates.into_iter().rev() {
            let field_name = match &key.kind {
                AstExpressionKind::Name(name)
                | AstExpressionKind::Literal(AstLiteral::String(name)) => name.as_str(),
                _ => {
                    return Err(Diagnostic::new(
                        "E000211",
                        "builder field names must be identifiers or string literals",
                        Some(key.span),
                    ))
                }
            };
            let Some((field, declaration)) = instance
                .fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == field_name && !field.name.starts_with('_'))
            else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!(
                        "class `{}` has no writable field `{field_name}`",
                        instance.name
                    ),
                    Some(key.span),
                ));
            };
            let value = self.expression(&value, Some(declaration.ty))?;
            fields[field] = self.validate_field_value(&instance, field, value, ast.span)?;
        }
        Ok(Some(Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Aggregate { class, fields },
            span: ast.span,
        }))
    }

    fn class_constructor(
        &mut self,
        class: &str,
        type_arguments: &[TypeAnnotation],
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let instance = if type_arguments.is_empty() {
            self.class_instances
                .get(&(class.to_owned(), Vec::new()))
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| self.instantiate_class(class, type_arguments, span))?
        } else {
            self.instantiate_class(class, type_arguments, span)?
        };
        if expected.is_some_and(|expected| expected != instance.ty) {
            return Err(semantic_error(
                "constructed class does not satisfy the expected type".into(),
                span,
            ));
        }
        let constructor = instance
            .constructors
            .iter()
            .find(|constructor| constructor.parameters.len() == arguments.len())
            .cloned();
        let implicit_defaults =
            constructor.is_none() && arguments.is_empty() && !instance.fields.is_empty();
        let fields = if self.is_error_type(instance.ty)
            && instance.fields.len() == 1
            && instance.fields[0].name == "__error"
            && instance.constructors.is_empty()
            && arguments.len() == 1
        {
            let string = self
                .types
                .resolve_name("string")
                .expect("bootstrap defines string");
            let message = self.expression(&arguments[0].value, Some(string))?;
            let function = self.string_expression(
                self.active_function_name
                    .clone()
                    .unwrap_or_else(|| "<module>".into()),
                span,
            );
            let error = self
                .types
                .resolve_name("Error")
                .expect("bootstrap defines Error");
            vec![self.runtime_call(
                "__sev_error_create",
                &[string, string],
                error,
                vec![message, function],
                span,
            )]
        } else if let Some(constructor) = constructor {
            self.class_constructor_values(&instance, &constructor, arguments, span)?
        } else if !instance.constructors.is_empty() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "constructor `{class}` has no overload accepting {} argument(s)",
                    arguments.len()
                ),
                Some(span),
            ));
        } else if arguments.is_empty() && !instance.fields.is_empty() {
            instance
                .fields
                .iter()
                .map(|field| self.default_expression(field.ty, span))
                .collect::<Result<Vec<_>, _>>()?
        } else if arguments.len() != instance.fields.len() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "constructor `{class}` expects {} field value(s), received {}",
                    instance.fields.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        } else {
            arguments
                .iter()
                .zip(&instance.fields)
                .map(|(argument, field)| self.expression(&argument.value, Some(field.ty)))
                .collect::<Result<Vec<_>, _>>()?
        };
        let fields = if implicit_defaults {
            fields
        } else {
            fields
                .into_iter()
                .enumerate()
                .map(|(field, value)| self.validate_field_value(&instance, field, value, span))
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Aggregate {
                class: instance.ty,
                fields,
            },
            span,
        })
    }

    fn inferred_class_constructor(
        &mut self,
        class: &str,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let declaration = self.classes.get(class).cloned().ok_or_else(|| {
            Diagnostic::new(
                "E000204",
                format!("unknown generic class `{class}`"),
                Some(span),
            )
        })?;
        if arguments.len() != declaration.fields.len() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "constructor `{class}` expects {} field value(s), received {}",
                    declaration.fields.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }

        let parameters = declaration
            .type_parameters
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut inferred = BTreeMap::<String, TypeId>::new();
        let mut fields = Vec::with_capacity(arguments.len());
        for (argument, field) in arguments.iter().zip(&declaration.fields) {
            let field_type_name = field.annotation.simple_name().ok_or_else(|| {
                Diagnostic::new(
                    "E000204",
                    "generic constructor inference currently requires named field types",
                    Some(field.annotation.span),
                )
            })?;
            if parameters.contains(field_type_name) {
                let expected_field = inferred.get(field_type_name).copied();
                let value = self.expression(&argument.value, expected_field)?;
                inferred
                    .entry(field_type_name.to_owned())
                    .or_insert(value.type_id);
                fields.push(value);
            } else {
                let field_type = resolve_type_annotation(self.types, &field.annotation)?;
                fields.push(self.expression(&argument.value, Some(field_type))?);
            }
        }
        let concrete = declaration
            .type_parameters
            .iter()
            .map(|parameter| {
                inferred.get(parameter).copied().ok_or_else(|| {
                    Diagnostic::new(
                        "E000206",
                        format!(
                            "cannot infer type parameter `{parameter}` for constructor `{class}`"
                        ),
                        Some(span),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let instance = self.instantiate_class_types(class, &concrete, span)?;
        if expected.is_some_and(|expected| expected != instance.ty) {
            return Err(semantic_error(
                "constructed class does not satisfy the expected type".into(),
                span,
            ));
        }
        let fields = fields
            .into_iter()
            .enumerate()
            .map(|(field, value)| self.validate_field_value(&instance, field, value, span))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Aggregate {
                class: instance.ty,
                fields,
            },
            span,
        })
    }

    fn instantiate_class(
        &mut self,
        name: &str,
        arguments: &[TypeAnnotation],
        span: severian_source::Span,
    ) -> Result<ClassInstance, Diagnostic> {
        let Some(declaration) = self.classes.get(name).cloned() else {
            return Err(Diagnostic::new(
                "E000204",
                format!("unknown generic class `{name}`"),
                Some(span),
            ));
        };
        if declaration.type_parameters.len() != arguments.len() {
            return Err(Diagnostic::new(
                "E000204",
                format!(
                    "class `{name}` expects {} type argument(s), received {}",
                    declaration.type_parameters.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }
        let concrete = arguments
            .iter()
            .map(|argument| self.resolve_source_type(argument))
            .collect::<Result<Vec<_>, _>>()?;
        self.instantiate_class_types(name, &concrete, span)
    }

    fn instantiate_class_types(
        &mut self,
        name: &str,
        concrete: &[TypeId],
        span: severian_source::Span,
    ) -> Result<ClassInstance, Diagnostic> {
        let declaration = self.classes.get(name).cloned().ok_or_else(|| {
            Diagnostic::new(
                "E000204",
                format!("unknown generic class `{name}`"),
                Some(span),
            )
        })?;
        if declaration.type_parameters.len() != concrete.len() {
            return Err(Diagnostic::new(
                "E000204",
                format!(
                    "class `{name}` expects {} type argument(s), received {}",
                    declaration.type_parameters.len(),
                    concrete.len()
                ),
                Some(span),
            ));
        }
        let key = (name.to_owned(), concrete.to_vec());
        if let Some(instance) = self.class_instances.get(&key) {
            return Ok(instance.clone());
        }
        let substitution = declaration
            .type_parameters
            .iter()
            .cloned()
            .zip(concrete.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let mut fields = Vec::with_capacity(declaration.fields.len());
        for field in &declaration.fields {
            let ty = self.resolve_instantiated_type(&field.annotation, &substitution)?;
            fields.push(HirClassFieldDeclaration {
                name: field.name.clone(),
                ty,
            });
        }
        let ty = TypeId(self.next_class_type);
        self.next_class_type = self.next_class_type.saturating_sub(1);
        let instance = ClassInstance {
            ty,
            name: name.to_owned(),
            fields: fields.clone(),
            source_fields: declaration.fields.clone(),
            constructors: declaration.constructors.clone(),
            methods: declaration.methods.clone(),
        };
        self.class_instances.insert(key, instance.clone());
        self.class_instances_by_type.insert(ty, instance.clone());
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!(
                "{}[{}]",
                name,
                concrete
                    .iter()
                    .map(|ty| {
                        self.types
                            .definition(*ty)
                            .map(|definition| definition.name.clone())
                            .unwrap_or_else(|| format!("type#{}", ty.0))
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            fields,
        });
        Ok(instance)
    }

    fn resolve_instantiated_type(
        &mut self,
        annotation: &TypeAnnotation,
        substitution: &BTreeMap<String, TypeId>,
    ) -> Result<TypeId, Diagnostic> {
        let Some((name, arguments)) = annotation.named_parts() else {
            return Err(Diagnostic::new(
                "E000204",
                "union class fields are not yet supported",
                Some(annotation.span),
            ));
        };
        if arguments.is_empty() {
            if let Some(ty) = substitution.get(name) {
                return Ok(*ty);
            }
            return resolve_type_annotation(self.types, annotation);
        }
        if name == "list" && arguments.len() == 1 {
            let element = self.resolve_instantiated_type(&arguments[0], substitution)?;
            return Ok(self.instantiate_list_type(element));
        }
        if name == "pointer" && arguments.len() == 1 {
            let element = self.resolve_instantiated_type(&arguments[0], substitution)?;
            return Ok(self.instantiate_pointer_type(element));
        }
        if name == "map" && arguments.len() == 2 {
            let key = self.resolve_instantiated_type(&arguments[0], substitution)?;
            let value = self.resolve_instantiated_type(&arguments[1], substitution)?;
            return Ok(self.instantiate_map_type(key, value));
        }
        Err(Diagnostic::new(
            "E000204",
            format!("generic field type `{name}` is not yet supported"),
            Some(annotation.span),
        ))
    }

    fn instantiate_list_type(&mut self, element: TypeId) -> TypeId {
        if let Some(ty) = self.list_types.get(&element) {
            return *ty;
        }
        let ty = list_type_id(element);
        let storage = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        let element_name = self
            .types
            .definition(element)
            .map(|definition| definition.name.clone())
            .unwrap_or_else(|| format!("type#{}", element.0));
        self.list_types.insert(element, ty);
        self.list_elements.insert(ty, element);
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("list[{element_name}]"),
            fields: vec![HirClassFieldDeclaration {
                name: "storage".into(),
                ty: storage,
            }],
        });
        ty
    }

    fn instantiate_pointer_type(&mut self, element: TypeId) -> TypeId {
        if let Some(ty) = self.pointer_types.get(&element) {
            return *ty;
        }
        let ty = pointer_type_id(element);
        self.pointer_types.insert(element, ty);
        self.pointer_elements.insert(ty, element);
        ty
    }

    fn instantiate_map_type(&mut self, key: TypeId, value: TypeId) -> TypeId {
        if let Some(ty) = self.map_types.get(&(key, value)) {
            return *ty;
        }
        let ty = map_type_id(key, value);
        let storage = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        let type_name = |id: TypeId| {
            self.types
                .definition(id)
                .map(|definition| definition.name.clone())
                .unwrap_or_else(|| format!("type#{}", id.0))
        };
        self.map_types.insert((key, value), ty);
        self.map_elements.insert(ty, (key, value));
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("map[{}, {}]", type_name(key), type_name(value)),
            fields: vec![
                HirClassFieldDeclaration {
                    name: "keys".into(),
                    ty: storage,
                },
                HirClassFieldDeclaration {
                    name: "values".into(),
                    ty: storage,
                },
            ],
        });
        ty
    }

    fn instantiate_fallible_type(&mut self, success: TypeId, error: TypeId) -> TypeId {
        let ty = fallible_type_id(success, error);
        if self.fallible_types.contains_key(&ty) {
            return ty;
        }
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        self.fallible_types
            .insert(ty, FallibleType { success, error });
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("result[type#{}, type#{}]", success.0, error.0),
            fields: vec![
                HirClassFieldDeclaration {
                    name: "ok".into(),
                    ty: boolean,
                },
                HirClassFieldDeclaration {
                    name: "value".into(),
                    ty: success,
                },
                HirClassFieldDeclaration {
                    name: "error".into(),
                    ty: error,
                },
            ],
        });
        ty
    }

    fn instantiate_union_type(&mut self, members: &[TypeId]) -> TypeId {
        let mut members = members.to_vec();
        members.sort();
        members.dedup();
        let ty = union_type_id(&members);
        if self.union_types.contains_key(&ty) {
            return ty;
        }
        let integer = self.tag_type();
        let mut fields = vec![HirClassFieldDeclaration {
            name: "__tag".into(),
            ty: integer,
        }];
        fields.extend(
            members
                .iter()
                .enumerate()
                .map(|(index, member)| HirClassFieldDeclaration {
                    name: format!("__value_{index}"),
                    ty: *member,
                }),
        );
        self.union_types.insert(ty, members.clone());
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!(
                "union[{}]",
                members
                    .iter()
                    .map(|member| format!("type#{}", member.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            fields,
        });
        ty
    }

    fn instantiate_function_type(&mut self, parameters: &[TypeId], result: TypeId) -> TypeId {
        let ty = function_type_id(parameters, result);
        if self.function_types.contains_key(&ty) {
            return ty;
        }
        self.function_types.insert(
            ty,
            FunctionType {
                parameters: parameters.to_vec(),
                result,
            },
        );
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!(
                "function[({}) -> type#{}]",
                parameters
                    .iter()
                    .map(|parameter| format!("type#{}", parameter.0))
                    .collect::<Vec<_>>()
                    .join(", "),
                result.0
            ),
            fields: Vec::new(),
        });
        ty
    }

    fn instantiate_lambda_type(&mut self, captures: &[(String, TypeId)]) -> TypeId {
        let ty = TypeId(self.next_class_type);
        self.next_class_type = self.next_class_type.saturating_sub(1);
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("lambda#{}", ty.0),
            fields: captures
                .iter()
                .map(|(name, ty)| HirClassFieldDeclaration {
                    name: name.clone(),
                    ty: *ty,
                })
                .collect(),
        });
        ty
    }

    fn union_expression(
        &mut self,
        union: TypeId,
        members: &[TypeId],
        value: Expression,
    ) -> Result<Expression, Diagnostic> {
        let Some(tag) = members.iter().position(|member| *member == value.type_id) else {
            return Err(semantic_error(
                "expression does not satisfy the expected union type".into(),
                value.span,
            ));
        };
        let span = value.span;
        let integer = self.tag_type();
        let mut fields = vec![self.integer_expression(&tag.to_string(), integer, span)];
        for (index, member) in members.iter().enumerate() {
            if index == tag {
                fields.push(value.clone());
            } else {
                fields.push(self.default_expression(*member, span)?);
            }
        }
        Ok(Expression {
            id: self.next_id(),
            type_id: union,
            kind: ExpressionKind::Aggregate {
                class: union,
                fields,
            },
            span,
        })
    }

    fn convert_union_expression(
        &mut self,
        union: Expression,
        members: &[TypeId],
        expected: TypeId,
    ) -> Result<Expression, Diagnostic> {
        let span = union.span;
        let integer = self.tag_type();
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let mut selected = None;
        for (index, member) in members.iter().copied().enumerate().rev() {
            let field = Expression {
                id: self.next_id(),
                type_id: member,
                kind: ExpressionKind::Field {
                    object: Box::new(union.clone()),
                    index: index as u32 + 1,
                },
                span,
            };
            let value = self.coerce(field, expected, true)?;
            let Some(fallback) = selected else {
                selected = Some(value);
                continue;
            };
            let tag = Expression {
                id: self.next_id(),
                type_id: integer,
                kind: ExpressionKind::Field {
                    object: Box::new(union.clone()),
                    index: 0,
                },
                span,
            };
            let ordinal = self.integer_expression(&index.to_string(), integer, span);
            let condition = Expression {
                id: self.next_id(),
                type_id: boolean,
                kind: ExpressionKind::Binary {
                    operator: BinaryOperator::Equal,
                    left: Box::new(tag),
                    right: Box::new(ordinal),
                },
                span,
            };
            selected = Some(Expression {
                id: self.next_id(),
                type_id: expected,
                kind: ExpressionKind::Fallback {
                    condition: Box::new(condition),
                    value: Box::new(value),
                    fallback: Box::new(fallback),
                },
                span,
            });
        }
        selected
            .ok_or_else(|| Diagnostic::new("E000204", "cannot convert an empty union", Some(span)))
    }

    fn channel_constructor(
        &mut self,
        element: &TypeAnnotation,
        capacity: Option<&AstExpression>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let element = self.resolve_source_type(element)?;
        let channel_type = self.instantiate_channel_type(element);
        let usize_type = self
            .types
            .resolve_name("usize")
            .expect("bootstrap defines usize");
        let capacity = match capacity {
            Some(capacity) => self.expression(capacity, Some(usize_type))?,
            None => self.integer_expression("0", usize_type, span),
        };
        let storage_type = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        let storage = self.runtime_call(
            "__sev_channel_create",
            &[usize_type],
            storage_type,
            vec![capacity],
            span,
        );
        Ok(Expression {
            id: self.next_id(),
            type_id: channel_type,
            kind: ExpressionKind::Aggregate {
                class: channel_type,
                fields: vec![storage],
            },
            span,
        })
    }

    fn instantiate_channel_type(&mut self, element: TypeId) -> TypeId {
        if let Some(ty) = self.channel_types.get(&element) {
            return *ty;
        }
        let ty = TypeId(self.next_class_type);
        self.next_class_type = self.next_class_type.saturating_sub(1);
        let storage = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        self.channel_types.insert(element, ty);
        self.channel_elements.insert(ty, element);
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("channel[type#{}]", element.0),
            fields: vec![HirClassFieldDeclaration {
                name: "storage".into(),
                ty: storage,
            }],
        });
        ty
    }

    fn instantiate_tuple_type(&mut self, elements: &[TypeId]) -> TypeId {
        if let Some(ty) = self.tuple_types.get(elements) {
            return *ty;
        }
        let ty = tuple_type_id(elements);
        let element_names = elements
            .iter()
            .map(|element| {
                self.types
                    .definition(*element)
                    .map(|definition| definition.name.clone())
                    .unwrap_or_else(|| format!("type#{}", element.0))
            })
            .collect::<Vec<_>>();
        let fields = elements
            .iter()
            .enumerate()
            .map(|(index, element)| HirClassFieldDeclaration {
                name: index.to_string(),
                ty: *element,
            })
            .collect::<Vec<_>>();
        self.tuple_types.insert(elements.to_vec(), ty);
        self.tuple_elements.insert(ty, elements.to_vec());
        self.lowered_classes.push(HirClassDeclaration {
            id: ty,
            name: format!("({})", element_names.join(", ")),
            fields,
        });
        ty
    }

    fn class_constructor_values(
        &mut self,
        instance: &ClassInstance,
        constructor: &severian_ast::FunctionDeclaration,
        arguments: &[severian_ast::CallArgument],
        span: severian_source::Span,
    ) -> Result<Vec<Expression>, Diagnostic> {
        let body = constructor.body.as_ref().ok_or_else(|| {
            Diagnostic::new(
                "E000211",
                format!("constructor `{}` has no implementation", constructor.name),
                Some(constructor.span),
            )
        })?;
        let resolved_arguments = arguments
            .iter()
            .zip(&constructor.parameters)
            .map(|(argument, parameter)| {
                let parameter_type = resolve_type_annotation(self.types, &parameter.annotation)?;
                self.expression(&argument.value, Some(parameter_type))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let previous = self.value_substitutions.clone();
        for (parameter, argument) in constructor.parameters.iter().zip(resolved_arguments) {
            self.value_substitutions
                .insert(parameter.name.clone(), argument);
        }
        let values = (|| {
            let mut values = Vec::with_capacity(instance.fields.len());
            for field in &instance.fields {
                let initializer = body.iter().find_map(|statement| {
                    let AstStatement::Binding(binding) = statement else {
                        return None;
                    };
                    (binding.name == field.name).then_some(&binding.value)
                });
                let Some(initializer) = initializer else {
                    return Err(Diagnostic::new(
                        "E000221",
                        format!(
                            "constructor `{}` does not initialize field `{}`",
                            constructor.name, field.name
                        ),
                        Some(span),
                    ));
                };
                values.push(self.expression(initializer, Some(field.ty))?);
            }
            Ok(values)
        })();
        self.value_substitutions = previous;
        values
    }

    fn empty_list_expression(
        &mut self,
        list_type: TypeId,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let storage_type = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        let storage = self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), span);
        Ok(Expression {
            id: self.next_id(),
            type_id: list_type,
            kind: ExpressionKind::Aggregate {
                class: list_type,
                fields: vec![storage],
            },
            span,
        })
    }

    fn empty_set_expression(
        &mut self,
        element: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let set_type = self.ensure_set_type_optional(element, span)?;
        let storage_type = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        let storage = self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), span);
        Ok(Expression {
            id: self.next_id(),
            type_id: set_type,
            kind: ExpressionKind::Aggregate {
                class: set_type,
                fields: vec![storage],
            },
            span,
        })
    }

    fn ensure_set_type(
        &mut self,
        element: TypeId,
        span: severian_source::Span,
    ) -> Result<TypeId, Diagnostic> {
        self.ensure_set_type_optional(Some(element), span)
    }

    fn ensure_set_type_optional(
        &mut self,
        element: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<TypeId, Diagnostic> {
        let set_type = set_type_id();
        let storage_type = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        if self.set_type.is_none() {
            self.set_type = Some(set_type);
            self.lowered_classes.push(HirClassDeclaration {
                id: set_type,
                name: "set".into(),
                fields: vec![HirClassFieldDeclaration {
                    name: "storage".into(),
                    ty: storage_type,
                }],
            });
        }
        if let Some(element) = element {
            if self.set_element.is_some_and(|known| known != element) {
                return Err(Diagnostic::new(
                    "E000204",
                    "set element types must agree within a compilation unit",
                    Some(span),
                ));
            }
            self.set_element = Some(element);
        }
        Ok(set_type)
    }

    fn enum_constructor(
        &mut self,
        path: &str,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let (enum_name, ordinal) = self.enum_variants[path].clone();
        let instance = self.enums[&enum_name].clone();
        let variant = &instance.variants[ordinal];
        if arguments.len() != variant.fields.len() {
            return Err(Diagnostic::new(
                "E000221",
                format!(
                    "enum variant `{}` expects {} payload value(s), received {}",
                    variant.name,
                    variant.fields.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }
        if expected.is_some_and(|expected| expected != instance.ty) {
            return Err(semantic_error(
                "enum variant does not satisfy the expected type".into(),
                span,
            ));
        }
        let integer = self
            .types
            .resolve_name("int")
            .expect("bootstrap defines int");
        let mut values = Vec::with_capacity(instance.fields.len());
        values.push(self.integer_expression(&ordinal.to_string(), integer, span));
        for (known_ordinal, known_variant) in instance.variants.iter().enumerate() {
            for (payload_ordinal, payload) in known_variant.fields.iter().enumerate() {
                let index = enum_payload_index(&instance.variants, known_ordinal, payload_ordinal);
                let field = &instance.fields[index];
                if known_ordinal == ordinal {
                    let argument = arguments
                        .iter()
                        .find(|argument| argument.name.as_deref() == Some(payload.name.as_str()))
                        .or_else(|| arguments.get(payload_ordinal))
                        .expect("enum arity was checked before payload lowering");
                    values.push(self.expression(&argument.value, Some(field.ty))?);
                } else {
                    values.push(self.default_expression(field.ty, span)?);
                }
            }
        }
        Ok(Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Aggregate {
                class: instance.ty,
                fields: values,
            },
            span,
        })
    }

    fn enum_value_constructor(
        &mut self,
        enum_name: &str,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let [argument] = arguments else {
            return Err(Diagnostic::new(
                "E000221",
                format!("enum conversion `{enum_name}` expects exactly one value"),
                Some(span),
            ));
        };
        if argument.name.is_some() {
            return Err(Diagnostic::new(
                "E000221",
                format!("enum conversion `{enum_name}` does not accept named arguments"),
                Some(argument.span),
            ));
        }
        let instance = self.enums[enum_name].clone();
        if expected.is_some_and(|expected| expected != instance.ty) {
            return Err(semantic_error(
                "enum conversion does not satisfy the expected type".into(),
                span,
            ));
        }

        let value = self.expression(&argument.value, None)?;
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let error = self.core_error_expression(
            format!("value is not accepted by enum `{enum_name}`"),
            span,
        );
        let mut converted = Expression {
            id: self.next_id(),
            type_id: instance.ty,
            kind: ExpressionKind::Throw(Box::new(error)),
            span,
        };
        for variant in instance.variants.iter().rev() {
            for accepted in variant.accepted_values.iter().rev() {
                let accepted = AstExpression {
                    kind: AstExpressionKind::Literal(accepted.clone()),
                    span,
                };
                let Ok(accepted) = self.expression(&accepted, Some(value.type_id)) else {
                    continue;
                };
                let condition = Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Binary {
                        operator: BinaryOperator::Equal,
                        left: Box::new(value.clone()),
                        right: Box::new(accepted),
                    },
                    span,
                };
                let canonical = self.enum_constructor(
                    &format!("{enum_name}.{}", variant.name),
                    &[],
                    Some(instance.ty),
                    span,
                )?;
                converted = Expression {
                    id: self.next_id(),
                    type_id: instance.ty,
                    kind: ExpressionKind::Fallback {
                        condition: Box::new(condition),
                        value: Box::new(canonical),
                        fallback: Box::new(converted),
                    },
                    span,
                };
            }
        }
        Ok(converted)
    }

    fn default_expression(
        &mut self,
        type_id: TypeId,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        if self.list_elements.contains_key(&type_id) {
            return self.empty_list_expression(type_id, span);
        }
        if self.map_elements.contains_key(&type_id) {
            let storage_type = self
                .types
                .resolve_name("string")
                .expect("bootstrap defines pointer-backed string");
            let keys =
                self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), span);
            let values =
                self.runtime_call("__sev_list_create", &[], storage_type, Vec::new(), span);
            return Ok(Expression {
                id: self.next_id(),
                type_id,
                kind: ExpressionKind::Aggregate {
                    class: type_id,
                    fields: vec![keys, values],
                },
                span,
            });
        }
        if let Some(elements) = self.tuple_elements.get(&type_id).cloned() {
            let fields = elements
                .into_iter()
                .map(|element| self.default_expression(element, span))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Expression {
                id: self.next_id(),
                type_id,
                kind: ExpressionKind::Aggregate {
                    class: type_id,
                    fields,
                },
                span,
            });
        }
        if let Some(instance) = self.class_instances_by_type.get(&type_id).cloned() {
            let fields = instance
                .fields
                .iter()
                .map(|field| self.default_expression(field.ty, span))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Expression {
                id: self.next_id(),
                type_id,
                kind: ExpressionKind::Aggregate {
                    class: type_id,
                    fields,
                },
                span,
            });
        }
        let literal = match self
            .types
            .primitive(type_id)
            .map(|primitive| primitive.representation)
        {
            Some(severian_universal::PrimitiveRepresentation::String) => {
                LiteralValue::String(String::new())
            }
            Some(severian_universal::PrimitiveRepresentation::Float { .. }) => {
                LiteralValue::Float("0.0".into())
            }
            Some(severian_universal::PrimitiveRepresentation::Boolean) => {
                LiteralValue::Boolean(false)
            }
            Some(severian_universal::PrimitiveRepresentation::Character) => {
                LiteralValue::Character('\0')
            }
            Some(severian_universal::PrimitiveRepresentation::None) => LiteralValue::None,
            Some(severian_universal::PrimitiveRepresentation::Unit) => LiteralValue::Unit,
            Some(
                severian_universal::PrimitiveRepresentation::Integer { .. }
                | severian_universal::PrimitiveRepresentation::PointerInteger { .. },
            ) => LiteralValue::Integer("0".into()),
            _ => {
                return Err(Diagnostic::new(
                    "E000221",
                    "this value requires an explicit default representation",
                    Some(span),
                ))
            }
        };
        Ok(Expression {
            id: self.next_id(),
            type_id,
            kind: ExpressionKind::Literal(literal),
            span,
        })
    }

    fn fallible_success_expression(
        &mut self,
        result_type: TypeId,
        fallible: FallibleType,
        value: Expression,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let ok = Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
            span,
        };
        let error = self.default_expression(fallible.error, span)?;
        Ok(Expression {
            id: self.next_id(),
            type_id: result_type,
            kind: ExpressionKind::Aggregate {
                class: result_type,
                fields: vec![ok, value, error],
            },
            span,
        })
    }

    fn fallible_error_expression(
        &mut self,
        result_type: TypeId,
        fallible: FallibleType,
        error: Expression,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let ok = Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Literal(LiteralValue::Boolean(false)),
            span,
        };
        let value = self.default_expression(fallible.success, span)?;
        Ok(Expression {
            id: self.next_id(),
            type_id: result_type,
            kind: ExpressionKind::Aggregate {
                class: result_type,
                fields: vec![ok, value, error],
            },
            span,
        })
    }

    fn unwrap_fallible_expression(
        &mut self,
        result: Expression,
        fallible: FallibleType,
        span: severian_source::Span,
    ) -> Expression {
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let condition = Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Field {
                object: Box::new(result.clone()),
                index: 0,
            },
            span,
        };
        let value = Expression {
            id: self.next_id(),
            type_id: fallible.success,
            kind: ExpressionKind::Field {
                object: Box::new(result.clone()),
                index: 1,
            },
            span,
        };
        let error = Expression {
            id: self.next_id(),
            type_id: fallible.error,
            kind: ExpressionKind::Field {
                object: Box::new(result),
                index: 2,
            },
            span,
        };
        let fallback = Expression {
            id: self.next_id(),
            type_id: fallible.success,
            kind: ExpressionKind::Throw(Box::new(error)),
            span,
        };
        Expression {
            id: self.next_id(),
            type_id: fallible.success,
            kind: ExpressionKind::Fallback {
                condition: Box::new(condition),
                value: Box::new(value),
                fallback: Box::new(fallback),
            },
            span,
        }
    }

    fn tuple_string(
        &mut self,
        tuple: Expression,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let elements = self.tuple_elements[&tuple.type_id].clone();
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        let mut rendered = Expression {
            id: self.next_id(),
            type_id: string,
            kind: ExpressionKind::Literal(LiteralValue::String("(".into())),
            span,
        };
        for (index, element) in elements.iter().enumerate() {
            if index != 0 {
                let comma = Expression {
                    id: self.next_id(),
                    type_id: string,
                    kind: ExpressionKind::Literal(LiteralValue::String(",".into())),
                    span,
                };
                rendered = self.runtime_call(
                    "__sev_string_concat",
                    &[string, string],
                    string,
                    vec![rendered, comma],
                    span,
                );
            }
            let field = Expression {
                id: self.next_id(),
                type_id: *element,
                kind: ExpressionKind::Field {
                    object: Box::new(tuple.clone()),
                    index: index as u32,
                },
                span,
            };
            let field = self.display_string(field, span)?;
            rendered = self.runtime_call(
                "__sev_string_concat",
                &[string, string],
                string,
                vec![rendered, field],
                span,
            );
        }
        let close = Expression {
            id: self.next_id(),
            type_id: string,
            kind: ExpressionKind::Literal(LiteralValue::String(")".into())),
            span,
        };
        Ok(self.runtime_call(
            "__sev_string_concat",
            &[string, string],
            string,
            vec![rendered, close],
            span,
        ))
    }

    fn display_string(
        &mut self,
        value: Expression,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        if let Some(fallible) = self.fallible_types.get(&value.type_id).copied() {
            let condition = Expression {
                id: self.next_id(),
                type_id: self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool"),
                kind: ExpressionKind::Field {
                    object: Box::new(value.clone()),
                    index: 0,
                },
                span,
            };
            let success = Expression {
                id: self.next_id(),
                type_id: fallible.success,
                kind: ExpressionKind::Field {
                    object: Box::new(value),
                    index: 1,
                },
                span,
            };
            let success = self.display_string(success, span)?;
            let failure = self.string_expression("Error", span);
            return Ok(Expression {
                id: self.next_id(),
                type_id: success.type_id,
                kind: ExpressionKind::Fallback {
                    condition: Box::new(condition),
                    value: Box::new(success),
                    fallback: Box::new(failure),
                },
                span,
            });
        }
        if self.tuple_elements.contains_key(&value.type_id) {
            return self.tuple_string(value, span);
        }
        if let Some(element) = self.list_elements.get(&value.type_id).copied() {
            let storage = self.list_storage_expression(value, span);
            let storage_type = storage.type_id;
            let string = self
                .types
                .resolve_name("string")
                .expect("bootstrap defines string");
            let suffix = self.list_runtime_suffix(element, span)?;
            return Ok(self.runtime_call(
                &format!("__sev_list_string_{suffix}"),
                &[storage_type],
                string,
                vec![storage],
                span,
            ));
        }
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        let name = self
            .types
            .definition(value.type_id)
            .map(|definition| definition.name.as_str());
        if self.integer_primitive(value.type_id) && name != Some("usize") {
            let integer = self
                .types
                .resolve_name("int")
                .expect("bootstrap defines int");
            let value = self.coerce(value, integer, true)?;
            return Ok(self.runtime_call(
                "__sev_string_from_int",
                &[integer],
                string,
                vec![value],
                span,
            ));
        }
        if self.types.primitive(value.type_id).is_some_and(|primitive| {
            matches!(
                primitive.category,
                severian_universal::PrimitiveCategory::Float
                    | severian_universal::PrimitiveCategory::Measured
            )
        }) {
            let float = self
                .types
                .resolve_name("float")
                .expect("bootstrap defines float");
            let value = self.coerce(value, float, true)?;
            return Ok(self.runtime_call(
                "__sev_string_from_float",
                &[float],
                string,
                vec![value],
                span,
            ));
        }
        match name {
            Some("string") => Ok(value),
            Some("usize") => Ok(self.runtime_call(
                "__sev_string_from_usize",
                &[value.type_id],
                string,
                vec![value],
                span,
            )),
            Some("bool") => Ok(self.runtime_call(
                "__sev_string_from_bool",
                &[value.type_id],
                string,
                vec![value],
                span,
            )),
            Some("char") => Ok(self.runtime_call(
                "__sev_string_from_char",
                &[value.type_id],
                string,
                vec![value],
                span,
            )),
            _ => Err(Diagnostic::new(
                "E000211",
                "print does not support this argument type",
                Some(span),
            )),
        }
    }

    fn system_surface_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        span: severian_source::Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        if let AstExpressionKind::TypeApplication {
            callee: application,
            arguments: type_arguments,
        } = &callee.kind
        {
            if callable_path(application).as_deref() == Some("platform.layout") {
                let ([queried], [target]) = (type_arguments.as_slice(), arguments) else {
                    return Err(Diagnostic::new(
                        "E000206",
                        "`platform.layout[T]` expects one type and one target",
                        Some(span),
                    ));
                };
                if target.name.is_some() {
                    return Err(Diagnostic::new(
                        "E000206",
                        "the platform target must be positional",
                        Some(target.value.span),
                    ));
                }
                let target_type = self
                    .class_instances
                    .get(&("platform.Target".into(), Vec::new()))
                    .map(|instance| instance.ty)
                    .ok_or_else(|| {
                        Diagnostic::new(
                            "E000204",
                            "platform.Target is unavailable",
                            Some(target.value.span),
                        )
                    })?;
                let _target = self.expression(&target.value, Some(target_type))?;
                let queried_type = self.resolve_source_type(queried)?;
                let instance = self.instantiate_class(
                    "platform.Layout",
                    std::slice::from_ref(queried),
                    span,
                )?;
                let (size, alignment) = self.type_layout(queried_type, span)?;
                let data_size = self
                    .types
                    .resolve_name("data_size")
                    .expect("bootstrap defines data_size");
                let measured = |value: u64, id: HirId| Expression {
                    id,
                    type_id: data_size,
                    kind: ExpressionKind::Literal(LiteralValue::Float(format!("{value}.0"))),
                    span,
                };
                let size = measured(size, self.next_id());
                let alignment = measured(alignment, self.next_id());
                self.platform_layout_types.insert(instance.ty, queried_type);
                return Ok(Some(Expression {
                    id: self.next_id(),
                    type_id: instance.ty,
                    kind: ExpressionKind::Aggregate {
                        class: instance.ty,
                        fields: vec![size, alignment],
                    },
                    span,
                }));
            }
        }
        if callable_path(callee).as_deref() == Some("process.arguments")
            && arguments.is_empty()
        {
            let string = self
                .types
                .resolve_name("string")
                .expect("bootstrap defines string");
            let list = self.instantiate_list_type(string);
            let storage = self.runtime_call(
                "__sev_process_arguments",
                &[],
                string,
                Vec::new(),
                span,
            );
            return Ok(Some(Expression {
                id: self.next_id(),
                type_id: list,
                kind: ExpressionKind::Aggregate {
                    class: list,
                    fields: vec![storage],
                },
                span,
            }));
        }
        if callable_path(callee).as_deref() == Some("io.write_all")
            && arguments.len() == 2
            && arguments.iter().all(|argument| argument.name.is_none())
        {
            let integer = self
                .types
                .resolve_name("int")
                .expect("bootstrap defines int");
            let byte = self
                .types
                .resolve_name("u8")
                .expect("bootstrap defines u8");
            let bytes_type = self.instantiate_list_type(byte);
            let writer = self.expression(&arguments[0].value, Some(integer))?;
            let bytes = self.expression(&arguments[1].value, Some(bytes_type))?;
            let storage = self.list_storage_expression(bytes, span);
            let storage_type = storage.type_id;
            let data_size = self
                .types
                .resolve_name("data_size")
                .expect("bootstrap defines data_size");
            return Ok(Some(self.runtime_call(
                "__sev_io_write_all",
                &[integer, storage_type],
                data_size,
                vec![writer, storage],
                span,
            )));
        }
        let AstExpressionKind::Member { object, name } = &callee.kind else {
            return Ok(None);
        };
        if name == "contains" && arguments.len() == 1 && arguments[0].name.is_none() {
            if let Ok(collection) = self.expression(object, None) {
                if let Some(element) = self.list_elements.get(&collection.type_id).copied() {
                    let needle = self.expression(&arguments[0].value, Some(element))?;
                    let suffix = self.list_runtime_suffix(element, span)?;
                    let storage = self.list_storage_expression(collection, span);
                    let storage_type = storage.type_id;
                    let boolean = self
                        .types
                        .resolve_name("bool")
                        .expect("bootstrap defines bool");
                    return Ok(Some(self.runtime_call(
                        &format!("__sev_list_contains_{suffix}"),
                        &[storage_type, element],
                        boolean,
                        vec![storage, needle],
                        span,
                    )));
                }
            }
        }
        if name == "field" && arguments.len() == 1 && arguments[0].name.is_none() {
            if let AstExpressionKind::Literal(AstLiteral::String(field_name)) =
                &arguments[0].value.kind
            {
                if let Ok(layout) = self.expression(object, None) {
                    if let Some(queried) = self.platform_layout_types.get(&layout.type_id).copied()
                    {
                        let offset = self.type_field_offset(queried, field_name, span)?;
                        let class = self.class_instances_by_type[&queried].clone();
                        let field = class
                            .fields
                            .iter()
                            .find(|field| field.name == *field_name)
                            .ok_or_else(|| {
                                Diagnostic::new(
                                    "E000204",
                                    format!("class `{}` has no field `{field_name}`", class.name),
                                    Some(arguments[0].value.span),
                                )
                            })?;
                        let (_, alignment) = self.type_layout(field.ty, span)?;
                        let field_layout = self
                            .class_instances
                            .get(&("platform.FieldLayout".into(), Vec::new()))
                            .cloned()
                            .ok_or_else(|| {
                                Diagnostic::new(
                                    "E000204",
                                    "platform.FieldLayout is unavailable",
                                    Some(span),
                                )
                            })?;
                        let data_size = self
                            .types
                            .resolve_name("data_size")
                            .expect("bootstrap defines data_size");
                        let value = |magnitude: u64, id: HirId| Expression {
                            id,
                            type_id: data_size,
                            kind: ExpressionKind::Literal(LiteralValue::Float(format!(
                                "{magnitude}.0"
                            ))),
                            span,
                        };
                        let offset = value(offset, self.next_id());
                        let alignment = value(alignment, self.next_id());
                        return Ok(Some(Expression {
                            id: self.next_id(),
                            type_id: field_layout.ty,
                            kind: ExpressionKind::Aggregate {
                                class: field_layout.ty,
                                fields: vec![offset, alignment],
                            },
                            span,
                        }));
                    }
                }
            }
        }
        if name == "bytes" && arguments.is_empty() {
            let string = self
                .types
                .resolve_name("string")
                .expect("bootstrap defines string");
            let value = self.expression(object, Some(string))?;
            let storage = self.runtime_call(
                "__sev_string_bytes",
                &[string],
                string,
                vec![value],
                span,
            );
            let byte = self
                .types
                .resolve_name("u8")
                .expect("bootstrap defines u8");
            let bytes = self.instantiate_list_type(byte);
            return Ok(Some(Expression {
                id: self.next_id(),
                type_id: bytes,
                kind: ExpressionKind::Aggregate {
                    class: bytes,
                    fields: vec![storage],
                },
                span,
            }));
        }
        Ok(None)
    }

    fn list_storage_expression(
        &mut self,
        list: Expression,
        span: severian_source::Span,
    ) -> Expression {
        self.collection_storage_expression(list, 0, span)
    }

    fn collection_storage_expression(
        &mut self,
        collection: Expression,
        index: u32,
        span: severian_source::Span,
    ) -> Expression {
        let storage_type = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        Expression {
            id: self.next_id(),
            type_id: storage_type,
            kind: ExpressionKind::Field {
                object: Box::new(collection),
                index,
            },
            span,
        }
    }

    fn channel_storage_expression(
        &mut self,
        channel: Expression,
        span: severian_source::Span,
    ) -> Expression {
        let storage = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines pointer-backed string");
        Expression {
            id: self.next_id(),
            type_id: storage,
            kind: ExpressionKind::Field {
                object: Box::new(channel),
                index: 0,
            },
            span,
        }
    }

    fn list_runtime_suffix(
        &self,
        element: TypeId,
        span: severian_source::Span,
    ) -> Result<&'static str, Diagnostic> {
        let name = self
            .types
            .definition(element)
            .map(|definition| definition.name.as_str());
        match name {
            Some("int") | Some("i64") | Some("usize") => Ok("i64"),
            Some("u8") => Ok("u8"),
            Some("bool") => Ok("bool"),
            Some("string") => Ok("ptr"),
            _ if self.list_elements.contains_key(&element) => Ok("list"),
            _ if self.tuple_elements.get(&element).is_some_and(|elements| {
                elements.len() == 2
                    && elements.iter().all(|element| {
                        self.types.definition(*element).is_some_and(|definition| {
                            matches!(definition.name.as_str(), "int" | "i64" | "usize")
                        })
                    })
            }) =>
            {
                Ok("pair_i64")
            }
            _ if self.class_instances_by_type.contains_key(&element) => Ok("aggregate"),
            _ => Err(Diagnostic::new(
                "E000211",
                "native list lowering does not yet support this element representation",
                Some(span),
            )),
        }
    }

    fn type_layout(
        &self,
        ty: TypeId,
        span: severian_source::Span,
    ) -> Result<(u64, u64), Diagnostic> {
        self.type_layout_inner(ty, span, &mut BTreeSet::new())
    }

    fn type_layout_inner(
        &self,
        ty: TypeId,
        span: severian_source::Span,
        visiting: &mut BTreeSet<TypeId>,
    ) -> Result<(u64, u64), Diagnostic> {
        use severian_universal::{FloatFormat, IntegerWidth, PrimitiveRepresentation};

        if self.pointer_elements.contains_key(&ty) {
            return Ok((8, 8));
        }
        if let Some(primitive) = self.types.primitive(ty) {
            let bytes = match primitive.representation {
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Fixed(bits),
                    ..
                }
                | PrimitiveRepresentation::Float {
                    format: FloatFormat::Ieee(bits),
                } => u64::from(bits).div_ceil(8).max(1),
                PrimitiveRepresentation::Integer {
                    bits: IntegerWidth::Machine,
                    ..
                }
                | PrimitiveRepresentation::PointerInteger { .. }
                | PrimitiveRepresentation::Float {
                    format: FloatFormat::Machine,
                }
                | PrimitiveRepresentation::String
                | PrimitiveRepresentation::Bytes
                | PrimitiveRepresentation::Arguments => 8,
                PrimitiveRepresentation::Float {
                    format: FloatFormat::BrainFloat16,
                } => 2,
                PrimitiveRepresentation::Boolean => 1,
                PrimitiveRepresentation::Character => 4,
                PrimitiveRepresentation::None | PrimitiveRepresentation::Unit => 0,
            };
            return Ok((bytes, bytes.max(1)));
        }
        let Some(class) = self.class_instances_by_type.get(&ty) else {
            return Err(Diagnostic::new(
                "E000204",
                "layout is unavailable for this type",
                Some(span),
            ));
        };
        if !visiting.insert(ty) {
            return Err(Diagnostic::new(
                "E000204",
                format!("class `{}` has a recursive inline layout", class.name),
                Some(span),
            ));
        }
        let mut size = 0u64;
        let mut aggregate_alignment = 1u64;
        for field in &class.fields {
            let (field_size, field_alignment) =
                self.type_layout_inner(field.ty, span, visiting)?;
            aggregate_alignment = aggregate_alignment.max(field_alignment);
            size = align_layout(size, field_alignment);
            size = size.saturating_add(field_size);
        }
        visiting.remove(&ty);
        Ok((align_layout(size, aggregate_alignment), aggregate_alignment))
    }

    fn type_field_offset(
        &self,
        ty: TypeId,
        requested: &str,
        span: severian_source::Span,
    ) -> Result<u64, Diagnostic> {
        let class = self.class_instances_by_type.get(&ty).ok_or_else(|| {
            Diagnostic::new(
                "E000204",
                "field offsets are available only for class types",
                Some(span),
            )
        })?;
        let mut offset = 0u64;
        let mut visiting = BTreeSet::new();
        for field in &class.fields {
            let (size, alignment) = self.type_layout_inner(field.ty, span, &mut visiting)?;
            offset = align_layout(offset, alignment);
            if field.name == requested {
                return Ok(offset);
            }
            offset = offset.saturating_add(size);
        }
        Err(Diagnostic::new(
            "E000211",
            format!("class `{}` has no field `{requested}`", class.name),
            Some(span),
        ))
    }

    fn select_runtime_suffix(
        &self,
        result: TypeId,
        span: severian_source::Span,
    ) -> Result<&'static str, Diagnostic> {
        let name = self
            .types
            .definition(result)
            .map(|definition| definition.name.as_str());
        match name {
            Some("string") => Ok("string"),
            Some("float" | "f64") => Ok("f64"),
            Some("f32") => Ok("f32"),
            Some("bool") => Ok("bool"),
            Some("int" | "i64" | "usize") => Ok("i64"),
            _ => Err(Diagnostic::new(
                "E000211",
                "enum return selection does not yet support this result type",
                Some(span),
            )),
        }
    }

    fn structural_class_equality(
        &mut self,
        left: Expression,
        right: Expression,
        span: severian_source::Span,
        visiting: &mut BTreeSet<TypeId>,
    ) -> Result<Expression, Diagnostic> {
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        if !visiting.insert(left.type_id) {
            return Err(Diagnostic::new(
                "E000211",
                "recursive class equality requires an explicit operator",
                Some(span),
            ));
        }
        let fields = self
            .class_instances_by_type
            .get(&left.type_id)
            .expect("caller checked the source class")
            .fields
            .clone();
        let mut comparison = Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
            span,
        };
        for (index, field) in fields.into_iter().enumerate() {
            let left_field = Expression {
                id: self.next_id(),
                type_id: field.ty,
                kind: ExpressionKind::Field {
                    object: Box::new(left.clone()),
                    index: index as u32,
                },
                span,
            };
            let right_field = Expression {
                id: self.next_id(),
                type_id: field.ty,
                kind: ExpressionKind::Field {
                    object: Box::new(right.clone()),
                    index: index as u32,
                },
                span,
            };
            let equal = if self.class_instances_by_type.contains_key(&field.ty) {
                self.structural_class_equality(left_field, right_field, span, visiting)?
            } else {
                Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Binary {
                        operator: BinaryOperator::Equal,
                        left: Box::new(left_field),
                        right: Box::new(right_field),
                    },
                    span,
                }
            };
            comparison = Expression {
                id: self.next_id(),
                type_id: boolean,
                kind: ExpressionKind::Binary {
                    operator: BinaryOperator::And,
                    left: Box::new(comparison),
                    right: Box::new(equal),
                },
                span,
            };
        }
        visiting.remove(&left.type_id);
        Ok(comparison)
    }

    fn runtime_call(
        &mut self,
        symbol: &str,
        parameter_types: &[TypeId],
        result_type: TypeId,
        arguments: Vec<Expression>,
        span: severian_source::Span,
    ) -> Expression {
        let definition = self.ensure_runtime_function(symbol, parameter_types, result_type);
        Expression {
            id: self.next_id(),
            type_id: result_type,
            kind: ExpressionKind::Call {
                callee: severian_hir::Callee::Direct {
                    function: definition,
                    substitution: severian_universal::Substitution::default(),
                },
                arguments,
            },
            span,
        }
    }

    fn approximate_call(
        &mut self,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        if expected.is_some_and(|expected| !self.types.assignable(boolean, expected)) {
            return Err(semantic_error(
                "`approximate` does not satisfy the expected type".into(),
                span,
            ));
        }
        if !(2..=4).contains(&arguments.len()) {
            return Err(Diagnostic::new(
                "E000206",
                "`approximate` expects actual, expected, and optional `atol`/`rtol` values",
                Some(span),
            ));
        }
        let default_float = self
            .types
            .resolve_name("f64")
            .or_else(|| self.types.resolve_name("float"))
            .expect("bootstrap defines a machine float");
        let float = default_float;
        let actual = self.expression(&arguments[0].value, None)?;
        let actual = self.coerce(actual, float, true)?;
        let expected_value = self.expression(&arguments[1].value, None)?;
        let expected_value = self.coerce(expected_value, float, true)?;
        let mut atol = self.default_expression(float, span)?;
        let mut rtol = self.default_expression(float, span)?;
        for (index, argument) in arguments.iter().enumerate().skip(2) {
            let target = match argument.name.as_deref() {
                Some("atol") => &mut atol,
                Some("rtol") => &mut rtol,
                Some(name) => {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("unknown `approximate` tolerance `{name}`"),
                        Some(argument.span),
                    ))
                }
                None if index == 2 => &mut atol,
                None => &mut rtol,
            };
            let value = self.expression(&argument.value, None)?;
            *target = self.coerce(value, float, true)?;
        }
        Ok(self.runtime_call(
            "__sev_approximate_f64",
            &[float, float, float, float],
            boolean,
            vec![actual, expected_value, atol, rtol],
            span,
        ))
    }

    fn integer_expression(
        &mut self,
        spelling: &str,
        type_id: TypeId,
        span: severian_source::Span,
    ) -> Expression {
        Expression {
            id: self.next_id(),
            type_id,
            kind: ExpressionKind::Literal(LiteralValue::Integer(spelling.into())),
            span,
        }
    }

    fn string_expression(
        &mut self,
        value: impl Into<String>,
        span: severian_source::Span,
    ) -> Expression {
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        Expression {
            id: self.next_id(),
            type_id: string,
            kind: ExpressionKind::Literal(LiteralValue::String(value.into())),
            span,
        }
    }

    fn throw_expression(
        &mut self,
        message: impl Into<String>,
        result: TypeId,
        span: severian_source::Span,
    ) -> Expression {
        let error = self.string_expression(message, span);
        Expression {
            id: self.next_id(),
            type_id: result,
            kind: ExpressionKind::Throw(Box::new(error)),
            span,
        }
    }

    fn core_error_expression(
        &mut self,
        message: impl Into<String>,
        span: severian_source::Span,
    ) -> Expression {
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        let error = self
            .types
            .resolve_name("Error")
            .expect("bootstrap defines Error");
        let message = self.string_expression(message, span);
        let function = self.string_expression(
            self.active_function_name
                .clone()
                .unwrap_or_else(|| "<module>".into()),
            span,
        );
        self.runtime_call(
            "__sev_error_create",
            &[string, string],
            error,
            vec![message, function],
            span,
        )
    }

    fn field_condition(
        &mut self,
        mut key: Expression,
        field: &str,
        span: severian_source::Span,
    ) -> Expression {
        key.id = self.next_id();
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let literal = self.string_expression(field, span);
        Expression {
            id: self.next_id(),
            type_id: boolean,
            kind: ExpressionKind::Binary {
                operator: BinaryOperator::Equal,
                left: Box::new(key),
                right: Box::new(literal),
            },
            span,
        }
    }

    fn validate_field_value(
        &mut self,
        instance: &ClassInstance,
        field: usize,
        value: Expression,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let Some(source_field) = instance.source_fields.get(field) else {
            return Ok(value);
        };
        if source_field.constraints.is_empty() {
            return Ok(value);
        }
        let field_name = &instance.fields[field].name;
        let previous = self
            .value_substitutions
            .insert(field_name.clone(), value.clone());
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let lowered = source_field
            .constraints
            .iter()
            .map(|constraint| {
                let condition = self.expression(&constraint.condition, Some(boolean))?;
                let failure = match constraint.failure.as_ref() {
                    Some(failure) => {
                        let error_message = match &failure.kind {
                            AstExpressionKind::Call { callee, arguments }
                                if callable_path(callee).as_deref() == Some("Error") =>
                            {
                                arguments.first().map(|argument| &argument.value)
                            }
                            _ => None,
                        };
                        let error = if let Some(message) = error_message {
                            let string = self
                                .types
                                .resolve_name("string")
                                .expect("bootstrap defines string");
                            self.expression(message, Some(string))?
                        } else if let Some(inlined) = self.inline_source_call(failure, None)? {
                            inlined
                        } else {
                            self.expression(failure, None)?
                        };
                        Some(error)
                    }
                    None => None,
                };
                Ok((condition, failure, constraint.span))
            })
            .collect::<Result<Vec<_>, Diagnostic>>();
        if let Some(previous) = previous {
            self.value_substitutions
                .insert(field_name.clone(), previous);
        } else {
            self.value_substitutions.remove(field_name);
        }
        let mut validated = value;
        for (condition, failure, constraint_span) in lowered?.into_iter().rev() {
            let rejects = failure.is_some();
            let condition = if rejects {
                Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Unary {
                        operator: severian_universal::UnaryOperator::Not,
                        operand: Box::new(condition),
                    },
                    span: constraint_span,
                }
            } else {
                condition
            };
            let error = failure.unwrap_or_else(|| {
                self.string_expression(
                    format!("constraint failed for `{}.{field_name}`", instance.name),
                    constraint_span,
                )
            });
            let fallback = Expression {
                id: self.next_id(),
                type_id: validated.type_id,
                kind: ExpressionKind::Throw(Box::new(error)),
                span: constraint_span,
            };
            validated = Expression {
                id: self.next_id(),
                type_id: validated.type_id,
                kind: ExpressionKind::Fallback {
                    condition: Box::new(condition),
                    value: Box::new(validated),
                    fallback: Box::new(fallback),
                },
                span,
            };
        }
        Ok(validated)
    }

    fn source_call_has_callable_parameter(&self, ast: &AstExpression) -> bool {
        let AstExpressionKind::Call { callee, arguments } = &ast.kind else {
            return false;
        };
        let Some(name) = callable_path(callee) else {
            return false;
        };
        self.source_functions.get(&name).is_some_and(|functions| {
            functions.iter().any(|function| {
                function.parameters.len() == arguments.len()
                    && function.parameters.iter().any(|parameter| {
                        matches!(
                            parameter.annotation.kind,
                            severian_ast::TypeAnnotationKind::Function { .. }
                        )
                    })
            })
        })
    }

    fn function_annotation(
        &mut self,
        annotation: &TypeAnnotation,
    ) -> Result<Option<FunctionType>, Diagnostic> {
        let severian_ast::TypeAnnotationKind::Function { parameters, result } = &annotation.kind
        else {
            return Ok(None);
        };
        let parameters = parameters
            .iter()
            .map(|parameter| self.resolve_source_type(parameter))
            .collect::<Result<Vec<_>, _>>()?;
        let result = self.resolve_source_type(result)?;
        Ok(Some(FunctionType { parameters, result }))
    }

    fn resolve_callable_value(
        &self,
        ast: &AstExpression,
        signature: &FunctionType,
    ) -> Result<ResolvedCallable, Diagnostic> {
        let AstExpressionKind::Name(name) = &ast.kind else {
            return Err(Diagnostic::new(
                "E000205",
                "a callable argument must name a function or bound lambda",
                Some(ast.span),
            ));
        };
        if let Some(callable) = self.callable_substitutions.get(name) {
            if callable.signature == *signature {
                return Ok(callable.clone());
            }
        }
        if let Some((_, variable, _)) = self.names.get(name) {
            if let Some(value) = self.callable_bindings.get(variable) {
                if matches!(value, CallableValue::Lambda { parameters, .. } if parameters.len() == signature.parameters.len())
                {
                    return Ok(ResolvedCallable {
                        value: value.clone(),
                        signature: signature.clone(),
                    });
                }
            }
        }
        let matches = self
            .functions
            .get(name)
            .into_iter()
            .flatten()
            .filter(|function| {
                let candidate = &self.signatures[function];
                candidate.parameters.len() == signature.parameters.len()
                    && candidate
                        .parameters
                        .iter()
                        .zip(&signature.parameters)
                        .all(|(left, right)| left.type_id == *right)
                    && candidate.result == signature.result
            })
            .copied()
            .collect::<Vec<_>>();
        let [function] = matches.as_slice() else {
            return Err(Diagnostic::new(
                "E000206",
                format!("`{name}` has no unique declaration matching the callable type"),
                Some(ast.span),
            ));
        };
        Ok(ResolvedCallable {
            value: CallableValue::Direct(*function),
            signature: signature.clone(),
        })
    }

    fn callable_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        let AstExpressionKind::Name(name) = &callee.kind else {
            return Ok(None);
        };
        let callable = self.callable_substitutions.get(name).cloned();
        if callable.is_none() {
            let lambda = self
                .names
                .get(name)
                .and_then(|(_, variable, _)| self.callable_bindings.get(variable))
                .cloned();
            if let Some(CallableValue::Lambda {
                parameters,
                body,
                closure,
                closure_type,
                captures,
            }) = lambda
            {
                if parameters.len() != arguments.len()
                    || arguments.iter().any(|argument| argument.name.is_some())
                {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("callable `{name}` received the wrong arguments"),
                        Some(span),
                    ));
                }
                let values = arguments
                    .iter()
                    .map(|argument| self.expression(&argument.value, None))
                    .collect::<Result<Vec<_>, _>>()?;
                let previous_values = self.value_substitutions.clone();
                let closure = Expression {
                    id: self.next_id(),
                    type_id: closure_type,
                    kind: ExpressionKind::Binding(closure),
                    span,
                };
                for (index, (capture, ty)) in captures.into_iter().enumerate() {
                    let id = self.next_id();
                    self.value_substitutions.insert(
                        capture,
                        Expression {
                            id,
                            type_id: ty,
                            kind: ExpressionKind::Field {
                                object: Box::new(closure.clone()),
                                index: index as u32,
                            },
                            span,
                        },
                    );
                }
                for (parameter, value) in parameters.into_iter().zip(values) {
                    self.value_substitutions.insert(parameter, value);
                }
                let result = self.expression(&body, expected);
                self.value_substitutions = previous_values;
                return result.map(Some);
            }
            return Ok(None);
        }
        let callable = callable.expect("callable presence was checked");
        if arguments.len() != callable.signature.parameters.len()
            || arguments.iter().any(|argument| argument.name.is_some())
        {
            return Err(Diagnostic::new(
                "E000206",
                format!("callable `{name}` received the wrong arguments"),
                Some(span),
            ));
        }
        if expected
            .is_some_and(|expected| !self.types.assignable(callable.signature.result, expected))
        {
            return Err(semantic_error(
                "callable result does not satisfy the expected type".into(),
                span,
            ));
        }
        let values = arguments
            .iter()
            .zip(&callable.signature.parameters)
            .map(|(argument, ty)| self.expression(&argument.value, Some(*ty)))
            .collect::<Result<Vec<_>, _>>()?;
        let result = match callable.value {
            CallableValue::Direct(function) => Expression {
                id: self.next_id(),
                type_id: callable.signature.result,
                kind: ExpressionKind::Call {
                    callee: severian_hir::Callee::Direct {
                        function: self.function_definitions[&function],
                        substitution: self.function_substitutions[&function].clone(),
                    },
                    arguments: values,
                },
                span,
            },
            CallableValue::Lambda {
                parameters,
                body,
                closure,
                closure_type,
                captures,
            } => {
                let previous_values = self.value_substitutions.clone();
                let previous_callables = self.callable_substitutions.clone();
                let closure = Expression {
                    id: self.next_id(),
                    type_id: closure_type,
                    kind: ExpressionKind::Binding(closure),
                    span,
                };
                for (index, (name, ty)) in captures.into_iter().enumerate() {
                    let id = self.next_id();
                    self.value_substitutions.insert(
                        name,
                        Expression {
                            id,
                            type_id: ty,
                            kind: ExpressionKind::Field {
                                object: Box::new(closure.clone()),
                                index: index as u32,
                            },
                            span,
                        },
                    );
                }
                for (parameter, value) in parameters.into_iter().zip(values) {
                    self.value_substitutions.insert(parameter, value);
                }
                let result = self.expression(&body, Some(callable.signature.result));
                self.value_substitutions = previous_values;
                self.callable_substitutions = previous_callables;
                result?
            }
        };
        Ok(Some(result))
    }

    fn inline_source_call(
        &mut self,
        ast: &AstExpression,
        expected: Option<TypeId>,
    ) -> Result<Option<Expression>, Diagnostic> {
        let AstExpressionKind::Call { callee, arguments } = &ast.kind else {
            return Ok(None);
        };
        let Some(name) = callable_path(callee) else {
            return Ok(None);
        };
        if self.mock_inline_stack.contains(&name) {
            return Ok(None);
        }
        let Some(function) = self.source_functions.get(&name).and_then(|functions| {
            let matches = functions
                .iter()
                .filter(|function| function.parameters.len() == arguments.len())
                .collect::<Vec<_>>();
            (matches.len() == 1).then(|| matches[0].clone())
        }) else {
            return Ok(None);
        };
        let Some(body) = function.body.as_ref() else {
            return Ok(None);
        };
        let [AstStatement::Return {
            value: Some(returned),
            ..
        }] = body.as_slice()
        else {
            return Ok(None);
        };
        if arguments.iter().any(|argument| argument.name.is_some()) {
            return Ok(None);
        }
        let result = self.resolve_source_type(&function.result)?;
        if expected.is_some_and(|expected| !self.types.assignable(result, expected)) {
            return Ok(None);
        }
        let mut resolved = Vec::with_capacity(arguments.len());
        let mut resolved_callables = Vec::new();
        for (argument, parameter) in arguments.iter().zip(&function.parameters) {
            if let Some(signature) = self.function_annotation(&parameter.annotation)? {
                resolved_callables.push((
                    parameter.name.clone(),
                    self.resolve_callable_value(&argument.value, &signature)?,
                ));
            } else {
                let ty = self.resolve_source_type(&parameter.annotation)?;
                resolved.push((
                    parameter.name.clone(),
                    self.expression(&argument.value, Some(ty))?,
                ));
            }
        }
        let previous = self.value_substitutions.clone();
        let previous_callables = self.callable_substitutions.clone();
        for (parameter, value) in resolved {
            self.value_substitutions.insert(parameter, value);
        }
        for (parameter, callable) in resolved_callables {
            self.callable_substitutions.insert(parameter, callable);
        }
        self.mock_inline_stack.insert(name.clone());
        let lowered = self.expression(returned, Some(result));
        self.mock_inline_stack.remove(&name);
        self.value_substitutions = previous;
        self.callable_substitutions = previous_callables;
        lowered.map(Some)
    }

    fn mocked_call(
        &mut self,
        ast: &AstExpression,
        expected: Option<TypeId>,
    ) -> Result<Option<Expression>, Diagnostic> {
        let AstExpressionKind::Call { callee, arguments } = &ast.kind else {
            return Ok(None);
        };
        let Some(name) = callable_path(callee) else {
            return Ok(None);
        };
        let Some(mock) = self.mocks.get(&name).cloned() else {
            return Ok(None);
        };
        let Some(function) = self.source_functions.get(&name).and_then(|functions| {
            let matches = functions
                .iter()
                .filter(|function| function.parameters.len() == arguments.len())
                .collect::<Vec<_>>();
            (matches.len() == 1).then(|| matches[0].clone())
        }) else {
            return Err(Diagnostic::new(
                "E000217",
                format!("mock target `{name}` has no unique declaration"),
                Some(ast.span),
            ));
        };
        let result_type = self.resolve_source_type(&function.result)?;
        if expected.is_some_and(|expected| !self.types.assignable(result_type, expected)) {
            return Ok(None);
        }
        let mut parameter_types = Vec::with_capacity(function.parameters.len());
        for parameter in &function.parameters {
            parameter_types.push(self.resolve_source_type(&parameter.annotation)?);
        }
        let actual = arguments
            .iter()
            .zip(&parameter_types)
            .map(|(argument, ty)| self.expression(&argument.value, Some(*ty)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut selected = self.expression(&mock.fallback, Some(result_type))?;
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        for case in mock.cases.into_iter().rev() {
            let AstExpressionKind::Call {
                arguments: patterns,
                ..
            } = &case.call.kind
            else {
                unreachable!("mock installation validates call cases")
            };
            if patterns.len() != actual.len() {
                return Err(Diagnostic::new(
                    "E000217",
                    format!("mock case for `{name}` has the wrong argument count"),
                    Some(case.span),
                ));
            }
            let mut comparisons = Vec::with_capacity(patterns.len());
            for ((value, pattern), ty) in actual.iter().zip(patterns).zip(&parameter_types) {
                let pattern = self.expression(&pattern.value, Some(*ty))?;
                comparisons.push(Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Binary {
                        operator: BinaryOperator::Equal,
                        left: Box::new(value.clone()),
                        right: Box::new(pattern),
                    },
                    span: case.span,
                });
            }
            let condition = comparisons
                .into_iter()
                .reduce(|left, right| Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Binary {
                    operator: BinaryOperator::And,
                    left: Box::new(left),
                        right: Box::new(right),
                    },
                    span: case.span,
                })
                .unwrap_or_else(|| Expression {
                    id: self.next_id(),
                    type_id: boolean,
                    kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
                span: case.span,
            });
            let value = self.expression(&case.result, Some(result_type))?;
            selected = Expression {
                id: self.next_id(),
                type_id: result_type,
                kind: ExpressionKind::Fallback {
                    condition: Box::new(condition),
                    value: Box::new(value),
                    fallback: Box::new(selected),
                },
                span: ast.span,
            };
        }
        Ok(Some(selected))
    }

    #[allow(clippy::too_many_arguments)]
    fn object_set_entry(
        &mut self,
        binding: BindingId,
        instance: &ClassInstance,
        key_ast: &AstExpression,
        value_ast: &AstExpression,
        commit: bool,
        enforce_constraint: bool,
        span: severian_source::Span,
    ) -> Result<Statement, Diagnostic> {
        let writable = instance
            .fields
            .iter()
            .enumerate()
            .filter(|(_, field)| !field.name.starts_with('_'))
            .collect::<Vec<_>>();
        let lower_action = |analyzer: &mut Self,
                            field: usize,
                            declaration: &HirClassFieldDeclaration|
         -> Result<Statement, Diagnostic> {
            let value = analyzer.expression(value_ast, Some(declaration.ty))?;
            let value = if enforce_constraint {
                analyzer.validate_field_value(instance, field, value, span)?
            } else {
                value
            };
            Ok(if commit {
                Statement::FieldSet {
                    binding,
                    field: field as u32,
                    value,
                }
            } else {
                Statement::Expression(value)
            })
        };

        if let Some(field_name) = string_literal(key_ast) {
            let Some((field, declaration)) = writable
                .iter()
                .copied()
                .find(|(_, field)| field.name == field_name)
            else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!(
                        "class `{}` has no writable field `{field_name}`",
                        instance.name
                    ),
                    Some(key_ast.span),
                )
                .with_help("use the name of a public writable field declared by this class"));
            };
            return lower_action(self, field, declaration);
        }

        let Some((_, first)) = writable.first().copied() else {
            return Err(Diagnostic::new(
                "E000211",
                format!(
                    "class `{}` has no dynamically writable fields",
                    instance.name
                ),
                Some(span),
            ));
        };
        if writable.iter().any(|(_, field)| field.ty != first.ty) {
            return Err(Diagnostic::new(
                "E000204",
                format!(
                    "dynamic `set` on class `{}` is ambiguous because its public fields have different types",
                    instance.name
                ),
                Some(span),
            )
            .with_help("use a literal field name or fields with one common value type"));
        }
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        let key = self.expression(key_ast, Some(string))?;
        let unit = self
            .types
            .resolve_name("unit")
            .expect("bootstrap defines unit");
        let mut selected = Statement::Expression(self.throw_expression(
            format!("unknown field on `{}`", instance.name),
            unit,
            span,
        ));
        for (field, declaration) in writable.into_iter().rev() {
            let condition = self.field_condition(key.clone(), &declaration.name, span);
            let action = lower_action(self, field, declaration)?;
            selected = Statement::If {
                condition,
                then_block: Block {
                    statements: vec![action],
                },
                else_block: Block {
                    statements: vec![selected],
                },
            };
        }
        Ok(selected)
    }

    fn object_set(
        &mut self,
        binding: BindingId,
        instance: &ClassInstance,
        arguments: &[severian_ast::CallArgument],
        span: severian_source::Span,
    ) -> Result<Statement, Diagnostic> {
        match arguments {
            [key, value] if key.name.is_none() && value.name.is_none() => self.object_set_entry(
                binding,
                instance,
                &key.value,
                &value.value,
                true,
                true,
                span,
            ),
            [updates] if updates.name.is_none() => {
                let AstExpressionKind::Map(entries) = &updates.value.kind else {
                    return Err(Diagnostic::new(
                        "E000206",
                        "object `set` expects `(field, value)` or one map literal",
                        Some(span),
                    ));
                };
                let mut statements = Vec::with_capacity(entries.len() * 2);
                for entry in entries {
                    statements.push(self.object_set_entry(
                        binding,
                        instance,
                        &entry.key,
                        &entry.value,
                        false,
                        true,
                        entry.span,
                    )?);
                }
                for entry in entries {
                    statements.push(self.object_set_entry(
                        binding,
                        instance,
                        &entry.key,
                        &entry.value,
                        true,
                        false,
                        entry.span,
                    )?);
                }
                Ok(Statement::Sequence(Block { statements }))
            }
            _ => Err(Diagnostic::new(
                "E000206",
                "object `set` expects `(field, value)` or one map literal",
                Some(span),
            )),
        }
    }

    fn ensure_runtime_function(
        &mut self,
        symbol: &str,
        parameter_types: &[TypeId],
        result_type: TypeId,
    ) -> DefId {
        if let Some(definition) = self.runtime_definitions.get(symbol) {
            return *definition;
        }
        let definition = synthetic_runtime_definition(symbol);
        let id = FunctionId(definition.declaration.0);
        let parameters = parameter_types
            .iter()
            .enumerate()
            .map(|(index, ty)| FunctionParameter {
                binding: self.new_binding_id(),
                name: format!("argument{index}"),
                contract: universal_boundary(*ty),
            })
            .collect();
        self.runtime_functions.push(FunctionDeclaration {
            id,
            definition,
            substitution: severian_universal::Substitution::default(),
            name: symbol.into(),
            type_parameters: Vec::new(),
            parameters,
            result: universal_boundary(result_type),
            compile_route: severian_universal::CompileRoute::Standard,
            call_type: severian_hir::CallType::External(severian_hir::ExternalCall {
                interface: severian_hir::InterfaceId("native-runtime".into()),
                symbol: severian_hir::SymbolId(symbol.into()),
                provider: None,
                ffi: severian_hir::FfiId("c".into()),
                abi: severian_hir::AbiId("native".into()),
            }),
            body: None,
        });
        self.runtime_definitions.insert(symbol.into(), definition);
        definition
    }

    fn trait_namespace_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        let Some(path) = callable_path(callee) else {
            return Ok(None);
        };
        if self.functions.contains_key(&path) {
            return Ok(None);
        }
        let AstExpressionKind::Member { object, .. } = &callee.kind else {
            return Ok(None);
        };
        if matches!(&object.kind, AstExpressionKind::Name(name) if self.names.contains_key(name)) {
            return Ok(None);
        }
        let Some(namespace_method) = self.namespace_methods.get(&path).cloned() else {
            return Ok(None);
        };

        let declaration = &namespace_method.declaration;
        let result = self.resolve_source_type(&declaration.result)?;
        if expected.is_some_and(|expected| !self.types.assignable(result, expected)) {
            return Err(semantic_error(
                "namespace method result does not satisfy the expected type".into(),
                span,
            ));
        }

        let mut ordered = vec![None; declaration.parameters.len()];
        let mut positional = 0usize;
        let mut named = false;
        for argument in arguments {
            let index = if let Some(argument_name) = &argument.name {
                named = true;
                declaration
                    .parameters
                    .iter()
                    .position(|parameter| parameter.name == *argument_name)
            } else if named {
                None
            } else {
                let index = positional;
                positional += 1;
                (index < ordered.len()).then_some(index)
            };
            let Some(index) = index else {
                return Err(Diagnostic::new(
                    "E000206",
                    format!("call to `{path}` has incompatible arguments"),
                    Some(span),
                ));
            };
            if ordered[index].replace(argument.value.clone()).is_some() {
                return Err(Diagnostic::new(
                    "E000206",
                    format!("call to `{path}` supplies an argument more than once"),
                    Some(argument.span),
                ));
            }
        }
        let ordered = ordered
            .into_iter()
            .zip(&declaration.parameters)
            .map(|(argument, parameter)| argument.or_else(|| parameter.default.clone()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Diagnostic::new(
                    "E000206",
                    format!("call to `{path}` is missing a required argument"),
                    Some(span),
                )
            })?;
        let mut resolved_arguments = Vec::with_capacity(ordered.len());
        for (argument, parameter) in ordered.iter().zip(&declaration.parameters) {
            let parameter_type = self.resolve_source_type(&parameter.annotation)?;
            resolved_arguments.push(self.expression(argument, Some(parameter_type))?);
        }

        if namespace_method.implementations.is_empty() {
            return Err(Diagnostic::new(
                "E000206",
                format!(
                    "namespace method `{path}` has no implementation of trait `{}`",
                    namespace_method.trait_name
                ),
                Some(callee.span),
            ));
        }

        let mut selected = self.throw_expression(
            format!("no applicable implementation for `{path}`"),
            result,
            span,
        );
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        for (class_name, implementation) in namespace_method.implementations.iter().rev() {
            let previous = self.value_substitutions.clone();
            let candidate = (|| {
                for (parameter, value) in implementation.parameters.iter().zip(&resolved_arguments)
                {
                    self.value_substitutions
                        .insert(parameter.name.clone(), value.clone());
                }
                let mut conditions = implementation
                    .contracts
                    .iter()
                    .filter(|contract| !contract.deferred)
                    .map(|contract| self.expression(&contract.condition, Some(boolean)))
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                let condition = if conditions.is_empty() {
                    Expression {
                        id: self.next_id(),
                        type_id: boolean,
                        kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
                        span: implementation.span,
                    }
                } else {
                    let mut condition = conditions.remove(0);
                    for next in conditions {
                        condition = Expression {
                            id: self.next_id(),
                            type_id: boolean,
                            kind: ExpressionKind::Binary {
                                operator: BinaryOperator::And,
                                left: Box::new(condition),
                                right: Box::new(next),
                            },
                            span: implementation.span,
                        };
                    }
                    condition
                };
                let body = implementation.body.as_ref().ok_or_else(|| {
                    Diagnostic::new(
                        "E000211",
                        format!(
                            "implementation `{class_name}.{}` has no body",
                            implementation.name
                        ),
                        Some(implementation.span),
                    )
                })?;
                let [AstStatement::Return {
                    value: Some(value), ..
                }] = body.as_slice()
                else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!(
                            "namespace implementation `{class_name}.{}` must currently be a single return expression",
                            implementation.name
                        ),
                        Some(implementation.span),
                    ));
                };
                let value = self.expression(value, Some(result))?;
                Ok((condition, value))
            })();
            self.value_substitutions = previous;
            let (condition, value) = candidate?;
            selected = Expression {
                id: self.next_id(),
                type_id: result,
                kind: ExpressionKind::Fallback {
                    condition: Box::new(condition),
                    value: Box::new(value),
                    fallback: Box::new(selected),
                },
                span,
            };
        }
        Ok(Some(selected))
    }

    fn extension_namespace_operator(
        &mut self,
        path: &str,
        extension: NamespaceExtensionOperator,
        left: &AstExpression,
        right: &AstExpression,
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let [right_parameter] = extension.implementation.parameters.as_slice() else {
            return Err(Diagnostic::new(
                "E000206",
                format!("extension operator `{path}` must declare one right-hand parameter"),
                Some(extension.implementation.span),
            ));
        };

        let left = self.expression(left, None)?;
        let aliases = self.extension_target_aliases(&extension.target, left.type_id)?;
        let previous_aliases = std::mem::replace(&mut self.active_type_aliases, aliases);
        let resolved = (|| {
            let target = self.resolve_source_type(&extension.target)?;
            if target != left.type_id {
                return Err(Diagnostic::new(
                    "E000202",
                    format!("operator `{path}` does not apply to the left operand type"),
                    Some(span),
                ));
            }
            let right_type = self.resolve_source_type(&right_parameter.annotation)?;
            let right = self.expression(right, Some(right_type))?;
            let result = self.resolve_source_type(&extension.implementation.result)?;
            if expected.is_some_and(|expected| !self.types.assignable(result, expected)) {
                return Err(semantic_error(
                    "extension operator result does not satisfy the expected type".into(),
                    span,
                ));
            }
            self.lower_extension_operator(path, &extension, left, right, result, span)
        })();
        self.active_type_aliases = previous_aliases;
        resolved
    }

    fn extension_target_aliases(
        &self,
        target: &TypeAnnotation,
        actual: TypeId,
    ) -> Result<BTreeMap<String, TypeId>, Diagnostic> {
        let Some((name, arguments)) = target.named_parts() else {
            return Err(Diagnostic::new(
                "E000204",
                "an extension target must be a named type",
                Some(target.span),
            ));
        };
        if name == "set" {
            if self.set_type != Some(actual) {
                return Ok(BTreeMap::new());
            }
            let element = self.set_element.ok_or_else(|| {
                Diagnostic::new("E000211", "set element type is unknown", Some(target.span))
            })?;
            return Ok(arguments
                .first()
                .and_then(TypeAnnotation::simple_name)
                .map(|parameter| BTreeMap::from([(parameter.to_owned(), element)]))
                .unwrap_or_default());
        }
        for ((class_name, concrete), instance) in &self.class_instances {
            if class_name == name && instance.ty == actual && concrete.len() == arguments.len() {
                return Ok(arguments
                    .iter()
                    .zip(concrete)
                    .filter_map(|(argument, concrete)| {
                        argument
                            .simple_name()
                            .map(|parameter| (parameter.to_owned(), *concrete))
                    })
                    .collect());
            }
        }
        Ok(BTreeMap::new())
    }

    fn lower_extension_operator(
        &mut self,
        path: &str,
        extension: &NamespaceExtensionOperator,
        left: Expression,
        right: Expression,
        result: TypeId,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let definition = synthetic_extension_definition(
            path,
            extension.implementation.span,
            &[left.type_id, right.type_id, result],
        );
        let function_id = FunctionId(definition.declaration.0);
        if !self
            .runtime_functions
            .iter()
            .any(|function| function.id == function_id)
        {
            let self_binding = self.new_binding_id();
            let right_binding = self.new_binding_id();
            let parameters = vec![
                FunctionParameter {
                    binding: self_binding,
                    name: "self".into(),
                    contract: universal_boundary(left.type_id),
                },
                FunctionParameter {
                    binding: right_binding,
                    name: extension.implementation.parameters[0].name.clone(),
                    contract: universal_boundary(right.type_id),
                },
            ];

            let previous_names = self.names.clone();
            let previous_declarations = self.declarations.clone();
            let previous_mutable = self.mutable_variables.clone();
            let previous_values = self.value_substitutions.clone();
            let previous_function = self.active_function_name.clone();
            self.names.clear();
            self.declarations.clear();
            self.value_substitutions.clear();
            self.active_function_name = Some(format!("extension {path}"));
            for parameter in &parameters {
                let variable = severian_hir::VariableId(parameter.binding.0);
                self.mutable_variables.insert(variable);
                self.names.insert(
                    parameter.name.clone(),
                    (parameter.binding, variable, parameter.contract.ty),
                );
                self.declarations.insert(parameter.name.clone());
            }
            let mut bindings = std::mem::take(&mut self.helper_bindings);
            let body = self.block(&extension.implementation.body, &mut bindings, result);
            self.helper_bindings = bindings;
            self.names = previous_names;
            self.declarations = previous_declarations;
            self.mutable_variables = previous_mutable;
            self.value_substitutions = previous_values;
            self.active_function_name = previous_function;
            let body = body?;
            if block_flow(&extension.implementation.body) == ControlFlow::FallsThrough {
                return Err(Diagnostic::new(
                    "E000209",
                    format!("not every path in extension operator `{path}` returns a result"),
                    Some(extension.implementation.span),
                ));
            }
            self.runtime_functions.push(FunctionDeclaration {
                id: function_id,
                definition,
                substitution: severian_universal::Substitution::default(),
                name: format!(
                    "__sev_extension_{}_{}",
                    extension.namespace,
                    internal_name_part(path)
                ),
                type_parameters: Vec::new(),
                parameters,
                result: universal_boundary(result),
                compile_route: severian_universal::CompileRoute::Standard,
                call_type: CallType::Severian,
                body: Some(body),
            });
        }
        Ok(Expression {
            id: self.next_id(),
            type_id: result,
            kind: ExpressionKind::Call {
                callee: severian_hir::Callee::Direct {
                    function: definition,
                    substitution: severian_universal::Substitution::default(),
                },
                arguments: vec![left, right],
            },
            span,
        })
    }

    fn trait_namespace_operator(
        &mut self,
        operator: &str,
        left: &AstExpression,
        right: &AstExpression,
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Expression, Diagnostic> {
        let namespaces = self
            .active_operator_namespaces
            .get(operator)
            .into_iter()
            .flatten()
            .cloned()
            .collect::<BTreeSet<_>>();
        let namespace = match namespaces.iter().collect::<Vec<_>>().as_slice() {
            [] => {
                return Err(Diagnostic::new(
                    "E000202",
                    format!("operator `{operator}` requires an active trait namespace"),
                    Some(span),
                )
                .with_help(format!(
                    "decorate the containing function with `@namespace({operator})`"
                )));
            }
            [namespace] => (*namespace).clone(),
            _ => {
                return Err(Diagnostic::new(
                    "E000210",
                    format!(
                        "operator `{operator}` is ambiguous between namespaces: {}",
                        namespaces.into_iter().collect::<Vec<_>>().join(", ")
                    ),
                    Some(span),
                )
                .with_help("activate exactly one trait namespace for this operator"));
            }
        };
        let path = format!("{namespace}.{operator}");
        if let Some(extension) = self.namespace_extension_operators.get(&path).cloned() {
            return self.extension_namespace_operator(
                &path,
                extension,
                left,
                right,
                expected,
                span,
            );
        }
        let namespace_operator = self
            .namespace_operators
            .get(&path)
            .cloned()
            .ok_or_else(|| {
                Diagnostic::new(
                    "E000202",
                    format!("active trait namespace does not declare operator `{operator}`"),
                    Some(span),
                )
            })?;
        let [left_parameter, right_parameter] =
            namespace_operator.declaration.parameters.as_slice()
        else {
            return Err(Diagnostic::new(
                "E000206",
                format!("namespace operator `{path}` must declare two parameters"),
                Some(namespace_operator.declaration.span),
            ));
        };
        let result = self.resolve_source_type(&namespace_operator.declaration.result)?;
        if expected.is_some_and(|expected| !self.types.assignable(result, expected)) {
            return Err(semantic_error(
                "namespace operator result does not satisfy the expected type".into(),
                span,
            ));
        }
        let left_type = self.resolve_source_type(&left_parameter.annotation)?;
        let right_type = self.resolve_source_type(&right_parameter.annotation)?;
        let arguments = vec![
            self.expression(left, Some(left_type))?,
            self.expression(right, Some(right_type))?,
        ];
        if namespace_operator.implementations.is_empty() {
            return Err(Diagnostic::new(
                "E000206",
                format!(
                    "namespace operator `{path}` has no implementation of trait `{}`",
                    namespace_operator.trait_name
                ),
                Some(namespace_operator.declaration.span),
            ));
        }
        let boolean = self
            .types
            .resolve_name("bool")
            .expect("bootstrap defines bool");
        let mut selected = self.throw_expression(
            format!("no applicable implementation for `{path}`"),
            result,
            span,
        );
        for (class_name, implementation) in namespace_operator.implementations.iter().rev() {
            let previous = self.value_substitutions.clone();
            let candidate = (|| {
                for (parameter, value) in implementation.parameters.iter().zip(&arguments) {
                    self.value_substitutions
                        .insert(parameter.name.clone(), value.clone());
                }
                let mut conditions = implementation
                    .contracts
                    .iter()
                    .filter(|contract| !contract.deferred)
                    .map(|contract| self.expression(&contract.condition, Some(boolean)))
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                let condition = if conditions.is_empty() {
                    Expression {
                        id: self.next_id(),
                        type_id: boolean,
                        kind: ExpressionKind::Literal(LiteralValue::Boolean(true)),
                        span: implementation.span,
                    }
                } else {
                    let mut condition = conditions.remove(0);
                    for next in conditions {
                        condition = Expression {
                            id: self.next_id(),
                            type_id: boolean,
                            kind: ExpressionKind::Binary {
                                operator: BinaryOperator::And,
                                left: Box::new(condition),
                                right: Box::new(next),
                            },
                            span: implementation.span,
                        };
                    }
                    condition
                };
                let [AstStatement::Return {
                    value: Some(value), ..
                }] = implementation.body.as_slice()
                else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!(
                            "operator implementation `{class_name}.{operator}` must currently be a single return expression"
                        ),
                        Some(implementation.span),
                    ));
                };
                Ok((condition, self.expression(value, Some(result))?))
            })();
            self.value_substitutions = previous;
            let (condition, value) = candidate?;
            selected = Expression {
                id: self.next_id(),
                type_id: result,
                kind: ExpressionKind::Fallback {
                    condition: Box::new(condition),
                    value: Box::new(value),
                    fallback: Box::new(selected),
                },
                span,
            };
        }
        Ok(selected)
    }

    fn class_method_call(
        &mut self,
        callee: &AstExpression,
        arguments: &[severian_ast::CallArgument],
        expected: Option<TypeId>,
        span: severian_source::Span,
    ) -> Result<Option<Expression>, Diagnostic> {
        let AstExpressionKind::Member { object, name } = &callee.kind else {
            return Ok(None);
        };
        if callable_path(callee).is_some_and(|path| self.functions.contains_key(&path)) {
            return Ok(None);
        }
        if matches!(name.as_str(), "map" | "filter" | "reduce") {
            let AstExpressionKind::List(values) = &object.kind else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("list method `{name}` currently requires a list literal"),
                    Some(span),
                ));
            };
            let Some(AstExpressionKind::Lambda { parameters, body }) =
                arguments.first().map(|argument| &argument.value.kind)
            else {
                return Err(Diagnostic::new(
                    "E000206",
                    format!("list method `{name}` expects a lambda"),
                    Some(span),
                ));
            };
            match name.as_str() {
                "map" if parameters.len() == 1 && arguments.len() == 1 => {
                    let mapped = values
                        .iter()
                        .map(|value| {
                            substituted_expression(
                                body,
                                &BTreeMap::from([(parameters[0].clone(), value.clone())]),
                            )
                        })
                        .collect();
                    return self
                        .expression(
                            &AstExpression {
                                kind: AstExpressionKind::List(mapped),
                                span,
                            },
                            expected,
                        )
                        .map(Some);
                }
                "filter" if parameters.len() == 1 && arguments.len() == 1 => {
                    let mut filtered = Vec::new();
                    for value in values {
                        let condition = substituted_expression(
                            body,
                            &BTreeMap::from([(parameters[0].clone(), value.clone())]),
                        );
                        if constant_boolean_expression(&condition).ok_or_else(|| {
                            Diagnostic::new(
                                "E000211",
                                "literal-list filtering requires a compile-time predicate",
                                Some(condition.span),
                            )
                        })? {
                            filtered.push(value.clone());
                        }
                    }
                    return self
                        .expression(
                            &AstExpression {
                                kind: AstExpressionKind::List(filtered),
                                span,
                            },
                            expected,
                        )
                        .map(Some);
                }
                "reduce" if parameters.len() == 2 && (1..=2).contains(&arguments.len()) => {
                    let (mut total, remaining) = if let Some(initial) = arguments.get(1) {
                        (initial.value.clone(), values.as_slice())
                    } else {
                        let Some((first, remaining)) = values.split_first() else {
                            return Err(Diagnostic::new(
                                "E000211",
                                "reducing an empty list requires an initial value",
                                Some(span),
                            ));
                        };
                        (first.clone(), remaining)
                    };
                    for value in remaining {
                        total = substituted_expression(
                            body,
                            &BTreeMap::from([
                                (parameters[0].clone(), total),
                                (parameters[1].clone(), value.clone()),
                            ]),
                        );
                    }
                    return self.expression(&total, expected).map(Some);
                }
                _ => {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("list method `{name}` received incompatible arguments"),
                        Some(span),
                    ));
                }
            }
        }
        if name == "sorted"
            && matches!(arguments.first().map(|argument| &argument.value.kind), Some(AstExpressionKind::Lambda { .. }))
        {
            let AstExpressionKind::List(values) = &object.kind else {
                return Err(Diagnostic::new(
                    "E000211",
                    "key-based sorting currently requires a list literal",
                    Some(span),
                ));
            };
            let mut values = values.clone();
            if values.iter().all(|value| string_literal(value).is_some()) {
                values.sort_by_key(|value| string_literal(value).map(str::chars).map(Iterator::count));
                let descending = arguments.get(1).is_some_and(|argument| {
                    matches!(argument.value.kind, AstExpressionKind::Literal(AstLiteral::Boolean(true)))
                });
                if descending {
                    values.reverse();
                }
                let literal = AstExpression {
                    kind: AstExpressionKind::List(values),
                    span,
                };
                return self.expression(&literal, expected).map(Some);
            }
        }
        let object = self.expression(object, None)?;
        let string = self
            .types
            .resolve_name("string")
            .expect("bootstrap defines string");
        if object.type_id == string {
            let usize_type = self
                .types
                .resolve_name("usize")
                .expect("bootstrap defines usize");
            let bool_type = self
                .types
                .resolve_name("bool")
                .expect("bootstrap defines bool");
            let int_type = self
                .types
                .resolve_name("int")
                .expect("bootstrap defines int");
            let (symbol, parameters, result, resolved_arguments) = match name.as_str() {
                "length" if arguments.is_empty() => (
                    "__sev_string_length",
                    vec![string],
                    usize_type,
                    vec![object],
                ),
                "upper" if arguments.is_empty() => {
                    ("__sev_string_upper", vec![string], string, vec![object])
                }
                "contains" if arguments.len() == 1 && arguments[0].name.is_none() => {
                    let needle = self.expression(&arguments[0].value, Some(string))?;
                    (
                        "__sev_string_contains",
                        vec![string, string],
                        bool_type,
                        vec![object, needle],
                    )
                }
                "split" if arguments.len() == 1 && arguments[0].name.is_none() => {
                    let separator = self.expression(&arguments[0].value, Some(string))?;
                    let storage_type = string;
                    let storage = self.runtime_call(
                        "__sev_string_split",
                        &[string, string],
                        storage_type,
                        vec![object, separator],
                        span,
                    );
                    let list_type = self.instantiate_list_type(string);
                    return Ok(Some(Expression {
                        id: self.next_id(),
                        type_id: list_type,
                        kind: ExpressionKind::Aggregate {
                            class: list_type,
                            fields: vec![storage],
                        },
                        span,
                    }));
                }
                "strip" if arguments.is_empty() => {
                    ("__sev_string_strip", vec![string], string, vec![object])
                }
                "lower" if arguments.is_empty() => {
                    ("__sev_string_lower", vec![string], string, vec![object])
                }
                "replace" if arguments.len() == 2 => {
                    let needle = self.expression(&arguments[0].value, Some(string))?;
                    let replacement = self.expression(&arguments[1].value, Some(string))?;
                    (
                        "__sev_string_replace",
                        vec![string, string, string],
                        string,
                        vec![object, needle, replacement],
                    )
                }
                "starts_with" if arguments.len() == 1 => {
                    let needle = self.expression(&arguments[0].value, Some(string))?;
                    (
                        "__sev_string_starts_with",
                        vec![string, string],
                        bool_type,
                        vec![object, needle],
                    )
                }
                "ends_with" if arguments.len() == 1 => {
                    let needle = self.expression(&arguments[0].value, Some(string))?;
                    (
                        "__sev_string_ends_with",
                        vec![string, string],
                        bool_type,
                        vec![object, needle],
                    )
                }
                "find" if arguments.len() == 1 => {
                    let needle = self.expression(&arguments[0].value, Some(string))?;
                    (
                        "__sev_string_find",
                        vec![string, string],
                        int_type,
                        vec![object, needle],
                    )
                }
                "count" if arguments.len() == 1 => {
                    let needle = self.expression(&arguments[0].value, Some(string))?;
                    (
                        "__sev_string_count",
                        vec![string, string],
                        int_type,
                        vec![object, needle],
                    )
                }
                "characters" if arguments.is_empty() => {
                    let storage = self.runtime_call(
                        "__sev_string_characters",
                        &[string],
                        string,
                        vec![object],
                        span,
                    );
                    let list_type = self.instantiate_list_type(string);
                    return Ok(Some(Expression {
                        id: self.next_id(),
                        type_id: list_type,
                        kind: ExpressionKind::Aggregate {
                            class: list_type,
                            fields: vec![storage],
                        },
                        span,
                    }));
                }
                "bytes" if arguments.is_empty() => {
                    let byte = self.types.resolve_name("u8").expect("bootstrap defines u8");
                    let storage = self.runtime_call(
                        "__sev_string_bytes",
                        &[string],
                        string,
                        vec![object],
                        span,
                    );
                    let list_type = self.instantiate_list_type(byte);
                    return Ok(Some(Expression {
                        id: self.next_id(),
                        type_id: list_type,
                        kind: ExpressionKind::Aggregate {
                            class: list_type,
                            fields: vec![storage],
                        },
                        span,
                    }));
                }
                "frequencies" if arguments.is_empty() => {
                    let integer = self
                        .types
                        .resolve_name("int")
                        .expect("bootstrap defines int");
                    let characters = self.runtime_call(
                        "__sev_string_characters",
                        &[string],
                        string,
                        vec![object],
                        span,
                    );
                    let keys = self.runtime_call(
                        "__sev_list_frequency_keys_ptr",
                        &[string],
                        string,
                        vec![characters.clone()],
                        span,
                    );
                    let values = self.runtime_call(
                        "__sev_list_frequency_values_ptr",
                        &[string],
                        string,
                        vec![characters],
                        span,
                    );
                    let map_type = self.instantiate_map_type(string, integer);
                    return Ok(Some(Expression {
                        id: self.next_id(),
                        type_id: map_type,
                        kind: ExpressionKind::Aggregate {
                            class: map_type,
                            fields: vec![keys, values],
                        },
                        span,
                    }));
                }
                "length" | "upper" | "contains" | "split" | "strip" | "lower"
                | "replace" | "starts_with" | "ends_with" | "find" | "count"
                | "characters" | "bytes" | "frequencies" => {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("string method `{name}` received incompatible arguments"),
                        Some(span),
                    ));
                }
                _ => return Ok(None),
            };
            if expected.is_some_and(|expected| !self.types.assignable(result, expected)) {
                return Err(semantic_error(
                    "method result does not satisfy the expected type".into(),
                    span,
                ));
            }
            return Ok(Some(self.runtime_call(
                symbol,
                &parameters,
                result,
                resolved_arguments,
                span,
            )));
        }
        if let Some(element) = self.list_elements.get(&object.type_id).copied() {
            let usize_type = self
                .types
                .resolve_name("usize")
                .expect("bootstrap defines usize");
            if name == "length" && arguments.is_empty() {
                if expected.is_some_and(|expected| !self.types.assignable(usize_type, expected)) {
                    return Err(semantic_error(
                        "method result does not satisfy the expected type".into(),
                        span,
                    ));
                }
                let storage = self.list_storage_expression(object, span);
                let storage_type = storage.type_id;
                return Ok(Some(self.runtime_call(
                    "__sev_list_len",
                    &[storage_type],
                    usize_type,
                    vec![storage],
                    span,
                )));
            }
            if name == "join" && arguments.len() == 1 {
                let string = self
                    .types
                    .resolve_name("string")
                    .expect("bootstrap defines string");
                if element != string {
                    return Err(Diagnostic::new(
                        "E000211",
                        "list method `join` requires string elements",
                        Some(span),
                    ));
                }
                let separator = self.expression(&arguments[0].value, Some(string))?;
                let storage = self.list_storage_expression(object, span);
                let storage_type = storage.type_id;
                return Ok(Some(self.runtime_call(
                    "__sev_list_join",
                    &[storage_type, string],
                    string,
                    vec![storage, separator],
                    span,
                )));
            }
            if matches!(name.as_str(), "minimum" | "maximum" | "sum" | "last")
                && arguments.is_empty()
            {
                let suffix = self.list_runtime_suffix(element, span)?;
                if name != "last" && suffix != "i64" {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!("list method `{name}` requires numeric elements"),
                        Some(span),
                    ));
                }
                let storage = self.list_storage_expression(object, span);
                let storage_type = storage.type_id;
                return Ok(Some(self.runtime_call(
                    &format!("__sev_list_{name}_{suffix}"),
                    &[storage_type],
                    element,
                    vec![storage],
                    span,
                )));
            }
            if name == "frequencies" && arguments.is_empty() {
                let integer = self
                    .types
                    .resolve_name("int")
                    .expect("bootstrap defines int");
                let map_type = self.instantiate_map_type(element, integer);
                let suffix = self.list_runtime_suffix(element, span)?;
                let storage = self.list_storage_expression(object, span);
                let storage_type = storage.type_id;
                let keys = self.runtime_call(
                    &format!("__sev_list_frequency_keys_{suffix}"),
                    &[storage_type],
                    storage_type,
                    vec![storage.clone()],
                    span,
                );
                let values = self.runtime_call(
                    &format!("__sev_list_frequency_values_{suffix}"),
                    &[storage_type],
                    storage_type,
                    vec![storage],
                    span,
                );
                return Ok(Some(Expression {
                    id: self.next_id(),
                    type_id: map_type,
                    kind: ExpressionKind::Aggregate {
                        class: map_type,
                        fields: vec![keys, values],
                    },
                    span,
                }));
            }
            if name == "sorted" {
                if arguments.len() > 1
                    || arguments
                        .first()
                        .is_some_and(|argument| !matches!(argument.value.kind, AstExpressionKind::Literal(AstLiteral::Boolean(_))))
                {
                    return Err(Diagnostic::new(
                        "E000206",
                        "list method `sorted` expects no arguments",
                        Some(span),
                    ));
                }
                let list_type = object.type_id;
                if expected.is_some_and(|expected| !self.types.assignable(list_type, expected)) {
                    return Err(semantic_error(
                        "method result does not satisfy the expected type".into(),
                        span,
                    ));
                }
                let suffix = self.list_runtime_suffix(element, span)?;
                if !matches!(suffix, "i64" | "ptr") {
                    return Err(Diagnostic::new(
                        "E000211",
                        "list sorting currently supports numeric and string elements",
                        Some(span),
                    ));
                }
                let storage = self.list_storage_expression(object, span);
                let storage_type = storage.type_id;
                let sorted = if let Some(argument) = arguments.first() {
                    let boolean = self
                        .types
                        .resolve_name("bool")
                        .expect("bootstrap defines bool");
                    let descending = self.expression(&argument.value, Some(boolean))?;
                    self.runtime_call(
                        &format!("__sev_list_sorted_order_{suffix}"),
                        &[storage_type, boolean],
                        storage_type,
                        vec![storage, descending],
                        span,
                    )
                } else {
                    self.runtime_call(
                        &format!("__sev_list_sorted_{suffix}"),
                        &[storage_type],
                        storage_type,
                        vec![storage],
                        span,
                    )
                };
                return Ok(Some(Expression {
                    id: self.next_id(),
                    type_id: list_type,
                    kind: ExpressionKind::Aggregate {
                        class: list_type,
                        fields: vec![sorted],
                    },
                    span,
                }));
            }
        }
        if let Some((key_type, value_type)) = self.map_elements.get(&object.type_id).copied() {
            if matches!(name.as_str(), "get" | "set_default") && arguments.len() == 2 {
                let key = self.expression(&arguments[0].value, Some(key_type))?;
                let fallback = self.expression(&arguments[1].value, Some(value_type))?;
                let keys = self.collection_storage_expression(object.clone(), 0, span);
                let values = self.collection_storage_expression(object, 1, span);
                let storage_type = keys.type_id;
                let key_suffix = self.list_runtime_suffix(key_type, span)?;
                let value_suffix = self.list_runtime_suffix(value_type, span)?;
                let operation = if name == "get" { "get_default" } else { "set_default" };
                return Ok(Some(self.runtime_call(
                    &format!("__sev_map_{operation}_{key_suffix}_{value_suffix}"),
                    &[storage_type, storage_type, key_type, value_type],
                    value_type,
                    vec![keys, values, key, fallback],
                    span,
                )));
            }
        }
        if self.set_type == Some(object.type_id)
            && matches!(name.as_str(), "union" | "intersection" | "symmetric_difference")
            && arguments.len() == 1
        {
            let other = self.expression(&arguments[0].value, Some(object.type_id))?;
            let element = self.set_element.ok_or_else(|| {
                Diagnostic::new("E000211", "set element type is unknown", Some(span))
            })?;
            let suffix = self.list_runtime_suffix(element, span)?;
            let left = self.collection_storage_expression(object, 0, span);
            let right = self.collection_storage_expression(other, 0, span);
            let storage_type = left.type_id;
            let storage = self.runtime_call(
                &format!("__sev_set_{name}_{suffix}"),
                &[storage_type, storage_type],
                storage_type,
                vec![left, right],
                span,
            );
            let set_type = self.set_type.expect("set type was checked");
            return Ok(Some(Expression {
                id: self.next_id(),
                type_id: set_type,
                kind: ExpressionKind::Aggregate {
                    class: set_type,
                    fields: vec![storage],
                },
                span,
            }));
        }
        let Some(instance) = self.class_instances_by_type.get(&object.type_id).cloned() else {
            return Ok(None);
        };
        if name.starts_with("__") {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` is private to class `{}`", instance.name),
                Some(callee.span),
            )
            .with_help("call a public class method instead"));
        }
        if name == "get" && !instance.methods.iter().any(|method| method.name == *name) {
            let [argument] = arguments else {
                return Err(Diagnostic::new(
                    "E000206",
                    "object `get` expects exactly one field name",
                    Some(span),
                ));
            };
            if let Some(field_name) = string_literal(&argument.value) {
                let Some((field, declaration)) =
                    instance.fields.iter().enumerate().find(|(_, field)| {
                        field.name == field_name && !field.name.starts_with("__")
                    })
                else {
                    return Err(Diagnostic::new(
                        "E000211",
                        format!(
                            "class `{}` has no readable field `{field_name}`",
                            instance.name
                        ),
                        Some(argument.value.span),
                    )
                    .with_help("use the name of a public field declared by this class"));
                };
                if expected.is_some_and(|expected| !self.types.assignable(declaration.ty, expected))
                {
                    return Err(semantic_error(
                        "dynamic field result does not satisfy the expected type".into(),
                        span,
                    ));
                }
                return Ok(Some(Expression {
                    id: self.next_id(),
                    type_id: declaration.ty,
                    kind: ExpressionKind::Field {
                        object: Box::new(object),
                        index: field as u32,
                    },
                    span,
                }));
            }
            let readable = instance
                .fields
                .iter()
                .enumerate()
                .filter(|(_, field)| !field.name.starts_with("__"))
                .collect::<Vec<_>>();
            let Some((_, first)) = readable.first().copied() else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!(
                        "class `{}` has no dynamically readable fields",
                        instance.name
                    ),
                    Some(span),
                ));
            };
            if readable.iter().any(|(_, field)| field.ty != first.ty) {
                return Err(Diagnostic::new(
                    "E000204",
                    format!(
                        "dynamic `get` on class `{}` is ambiguous because its public fields have different types",
                        instance.name
                    ),
                    Some(span),
                )
                .with_help("use a literal field name or fields with one common result type"));
            }
            if expected.is_some_and(|expected| !self.types.assignable(first.ty, expected)) {
                return Err(semantic_error(
                    "dynamic field result does not satisfy the expected type".into(),
                    span,
                ));
            }
            let string = self
                .types
                .resolve_name("string")
                .expect("bootstrap defines string");
            let key = self.expression(&argument.value, Some(string))?;
            let mut selected = self.throw_expression(
                format!("unknown field on `{}`", instance.name),
                first.ty,
                span,
            );
            for (field, declaration) in readable.into_iter().rev() {
                let condition = self.field_condition(key.clone(), &declaration.name, span);
                let mut branch_object = object.clone();
                branch_object.id = self.next_id();
                let value = Expression {
                    id: self.next_id(),
                    type_id: declaration.ty,
                    kind: ExpressionKind::Field {
                        object: Box::new(branch_object),
                        index: field as u32,
                    },
                    span,
                };
                selected = Expression {
                    id: self.next_id(),
                    type_id: first.ty,
                    kind: ExpressionKind::Fallback {
                        condition: Box::new(condition),
                        value: Box::new(value),
                        fallback: Box::new(selected),
                    },
                    span,
                };
            }
            return Ok(Some(selected));
        }
        let Some(method) = instance.methods.iter().find(|method| method.name == *name) else {
            return Err(Diagnostic::new(
                "E000211",
                format!("class `{}` has no method `{name}`", instance.name),
                Some(callee.span),
            ));
        };
        if method.parameters.len() != arguments.len() {
            return Err(Diagnostic::new(
                "E000206",
                format!(
                    "method `{name}` expects {} argument(s), received {}",
                    method.parameters.len(),
                    arguments.len()
                ),
                Some(span),
            ));
        }
        let Some(body) = &method.body else {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` has no implementation"),
                Some(method.span),
            ));
        };
        if let Some(field_name) = body.iter().find_map(|statement| {
            let AstStatement::Return {
                value: Some(value), ..
            } = statement
            else {
                return None;
            };
            let AstExpressionKind::Call { callee, arguments } = &value.kind else {
                return None;
            };
            let AstExpressionKind::Member {
                object: field,
                name: operation,
            } = &callee.kind
            else {
                return None;
            };
            let AstExpressionKind::Name(field) = &field.kind else {
                return None;
            };
            (operation == "pop" && arguments.is_empty()).then_some(field.as_str())
        }) {
            let Some((field, declaration)) = instance
                .fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == field_name)
            else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("class `{}` has no field `{field_name}`", instance.name),
                    Some(method.span),
                ));
            };
            let Some(element) = self.list_elements.get(&declaration.ty).copied() else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("field `{field_name}` is not a list"),
                    Some(method.span),
                ));
            };
            if expected.is_some_and(|expected| !self.types.assignable(element, expected)) {
                return Err(semantic_error(
                    "method result does not satisfy the expected type".into(),
                    span,
                ));
            }
            let list = Expression {
                id: self.next_id(),
                type_id: declaration.ty,
                kind: ExpressionKind::Field {
                    object: Box::new(object),
                    index: field as u32,
                },
                span,
            };
            let storage = self.list_storage_expression(list, span);
            let storage_type = storage.type_id;
            let suffix = self.list_runtime_suffix(element, span)?;
            let symbol = format!("__sev_list_pop_{suffix}");
            return Ok(Some(self.runtime_call(
                &symbol,
                &[storage_type],
                element,
                vec![storage],
                span,
            )));
        }
        if let Some(return_value) = body.iter().find_map(|statement| match statement {
            AstStatement::Return {
                value: Some(value), ..
            } if !matches!(value.kind, AstExpressionKind::Name(_)) => Some(value),
            _ => None,
        }) {
            let result_type = self.resolve_source_type(&method.result)?;
            if expected.is_some_and(|expected| !self.types.assignable(result_type, expected)) {
                return Err(semantic_error(
                    "method result does not satisfy the expected type".into(),
                    span,
                ));
            }
            let previous = self.value_substitutions.clone();
            for (field, declaration) in instance.fields.iter().enumerate() {
                let id = self.next_id();
                self.value_substitutions.insert(
                    declaration.name.clone(),
                    Expression {
                        id,
                        type_id: declaration.ty,
                        kind: ExpressionKind::Field {
                            object: Box::new(object.clone()),
                            index: field as u32,
                        },
                        span,
                    },
                );
            }
            let resolved = (|| {
                for (parameter, argument) in method.parameters.iter().zip(arguments) {
                    let parameter_type = self.resolve_source_type(&parameter.annotation)?;
                    let value = self.expression(&argument.value, Some(parameter_type))?;
                    self.value_substitutions
                        .insert(parameter.name.clone(), value);
                }
                self.expression(return_value, Some(result_type))
            })();
            self.value_substitutions = previous;
            return resolved.map(Some);
        }
        let field_name = body.iter().find_map(|statement| {
            let AstStatement::Return {
                value: Some(value), ..
            } = statement
            else {
                return None;
            };
            let AstExpressionKind::Name(field) = &value.kind else {
                return None;
            };
            Some(field.as_str())
        });
        let Some((field, declaration)) = field_name.and_then(|field_name| {
            instance
                .fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == field_name)
        }) else {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` is not a field-returning method"),
                Some(method.span),
            ));
        };
        if expected.is_some_and(|expected| !self.types.assignable(declaration.ty, expected)) {
            return Err(semantic_error(
                "method result does not satisfy the expected type".into(),
                span,
            ));
        }
        Ok(Some(Expression {
            id: self.next_id(),
            type_id: declaration.ty,
            kind: ExpressionKind::Field {
                object: Box::new(object),
                index: field as u32,
            },
            span,
        }))
    }

    fn class_method_update(
        &mut self,
        expression: &AstExpression,
    ) -> Result<Option<Statement>, Diagnostic> {
        let AstExpressionKind::Call { callee, arguments } = &expression.kind else {
            return Ok(None);
        };
        let AstExpressionKind::Member { object, name } = &callee.kind else {
            return Ok(None);
        };
        let AstExpressionKind::Name(receiver) = &object.kind else {
            return Ok(None);
        };
        let Some((binding, _, ty)) = self.names.get(receiver).copied() else {
            return Ok(None);
        };
        let Some(instance) = self.class_instances_by_type.get(&ty).cloned() else {
            return Ok(None);
        };
        if name.starts_with("__") {
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` is private to class `{}`", instance.name),
                Some(callee.span),
            )
            .with_help("call a public class method instead"));
        }
        if name == "set" {
            return self
                .object_set(binding, &instance, arguments, expression.span)
                .map(Some);
        }
        if name == "get" && !instance.methods.iter().any(|method| method.name == *name) {
            return Ok(None);
        }
        let Some(mut method) = instance
            .methods
            .iter()
            .find(|method| method.name == *name)
            .cloned()
        else {
            return Err(Diagnostic::new(
                "E000211",
                format!("class `{}` has no method `{name}`", instance.name),
                Some(callee.span),
            ));
        };
        if method.parameters.len() != arguments.len() {
            return Err(Diagnostic::new(
                "E000206",
                format!(
                    "method `{name}` expects {} argument(s), received {}",
                    method.parameters.len(),
                    arguments.len()
                ),
                Some(expression.span),
            ));
        }
        let mut delegated = BTreeSet::from([method.name.clone()]);
        loop {
            let Some(body) = &method.body else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("method `{}` has no implementation", method.name),
                    Some(method.span),
                ));
            };
            let delegate = match body.as_slice() {
                [AstStatement::Expression(AstExpression {
                    kind: AstExpressionKind::Call { callee, arguments },
                    ..
                })] if arguments.is_empty() => match &callee.kind {
                    AstExpressionKind::Name(delegate) => Some(delegate.as_str()),
                    _ => None,
                },
                _ => None,
            };
            let Some(delegate) = delegate else {
                break;
            };
            if !delegated.insert(delegate.to_owned()) {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("method delegation through `{delegate}` is recursive"),
                    Some(method.span),
                ));
            }
            let Some(next) = instance
                .methods
                .iter()
                .find(|candidate| candidate.name == delegate && candidate.parameters.is_empty())
                .cloned()
            else {
                break;
            };
            method = next;
        }
        let body = method
            .body
            .as_ref()
            .expect("delegated methods retain bodies");
        if !body.is_empty()
            && body.iter().all(|statement| {
                matches!(statement, AstStatement::Binding(assignment)
                    if !assignment.update
                        && instance.fields.iter().any(|field| field.name == assignment.name))
            })
        {
            let previous = self.value_substitutions.clone();
            let resolved = (|| {
                for (parameter, argument) in method.parameters.iter().zip(arguments) {
                    let parameter_type = self.resolve_source_type(&parameter.annotation)?;
                    let value = self.expression(&argument.value, Some(parameter_type))?;
                    self.value_substitutions
                        .insert(parameter.name.clone(), value);
                }
                body.iter()
                    .map(|statement| {
                        let AstStatement::Binding(assignment) = statement else {
                            unreachable!("direct field assignment body was checked above")
                        };
                        let (field, declaration) = instance
                            .fields
                            .iter()
                            .enumerate()
                            .find(|(_, field)| field.name == assignment.name)
                            .expect("direct field assignment target was checked above");
                        Ok(Statement::FieldSet {
                            binding,
                            field: field as u32,
                            value: self.expression(&assignment.value, Some(declaration.ty))?,
                        })
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()
            })();
            self.value_substitutions = previous;
            return resolved.map(|statements| Some(Statement::Sequence(Block { statements })));
        }
        if let Some(field_name) = body.iter().find_map(|statement| {
            let AstStatement::Expression(expression) = statement else {
                return None;
            };
            let AstExpressionKind::Call {
                callee,
                arguments: method_arguments,
            } = &expression.kind
            else {
                return None;
            };
            let AstExpressionKind::Member {
                object: field,
                name: operation,
            } = &callee.kind
            else {
                return None;
            };
            let AstExpressionKind::Name(field) = &field.kind else {
                return None;
            };
            let [argument] = method_arguments.as_slice() else {
                return None;
            };
            let AstExpressionKind::Name(argument_name) = &argument.value.kind else {
                return None;
            };
            (operation == "append"
                && method
                    .parameters
                    .first()
                    .is_some_and(|parameter| argument_name == &parameter.name))
            .then_some(field.as_str())
        }) {
            let Some((field, declaration)) = instance
                .fields
                .iter()
                .enumerate()
                .find(|(_, field)| field.name == field_name)
            else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("class `{}` has no field `{field_name}`", instance.name),
                    Some(method.span),
                ));
            };
            let Some(element) = self.list_elements.get(&declaration.ty).copied() else {
                return Err(Diagnostic::new(
                    "E000211",
                    format!("field `{field_name}` is not a list"),
                    Some(method.span),
                ));
            };
            let receiver = Expression {
                id: self.next_id(),
                type_id: ty,
                kind: ExpressionKind::Binding(binding),
                span: object.span,
            };
            let list = Expression {
                id: self.next_id(),
                type_id: declaration.ty,
                kind: ExpressionKind::Field {
                    object: Box::new(receiver),
                    index: field as u32,
                },
                span: object.span,
            };
            let storage = self.list_storage_expression(list, expression.span);
            let storage_type = storage.type_id;
            let value = self.expression(&arguments[0].value, Some(element))?;
            let suffix = self.list_runtime_suffix(element, expression.span)?;
            let symbol = format!("__sev_list_push_{suffix}");
            let unit = self
                .types
                .resolve_name("unit")
                .expect("bootstrap defines unit");
            return Ok(Some(Statement::Expression(self.runtime_call(
                &symbol,
                &[storage_type, element],
                unit,
                vec![storage, value],
                expression.span,
            ))));
        }
        let update = body.iter().find_map(|statement| {
            let severian_ast::Statement::Binding(update) = statement else {
                return None;
            };
            if !update.update {
                return None;
            }
            let AstExpressionKind::Binary {
                operator,
                left,
                right,
            } = &update.value.kind
            else {
                return None;
            };
            let AstExpressionKind::Name(left) = &left.kind else {
                return None;
            };
            (left == &update.name).then(|| (update.name.as_str(), *operator, right.as_ref()))
        });
        let Some((field_name, operator, update_value)) = update else {
            let unit = self
                .types
                .resolve_name("unit")
                .expect("bootstrap defines unit");
            let none = self
                .types
                .resolve_name("None")
                .expect("bootstrap defines None");
            let result = self.resolve_source_type(&method.result)?;
            if result != unit && result != none {
                // A value-returning method used as a statement is still an
                // ordinary expression call. Let `class_method_call` lower it.
                return Ok(None);
            }
            if body
                .iter()
                .all(|statement| matches!(statement, AstStatement::Expression(_)))
            {
                let previous = self.value_substitutions.clone();
                let receiver = Expression {
                    id: self.next_id(),
                    type_id: ty,
                    kind: ExpressionKind::Binding(binding),
                    span: object.span,
                };
                for (field, declaration) in instance.fields.iter().enumerate() {
                    let id = self.next_id();
                    self.value_substitutions.insert(
                        declaration.name.clone(),
                        Expression {
                            id,
                            type_id: declaration.ty,
                            kind: ExpressionKind::Field {
                                object: Box::new(receiver.clone()),
                                index: field as u32,
                            },
                            span: object.span,
                        },
                    );
                }
                let resolved = (|| {
                    for (parameter, argument) in method.parameters.iter().zip(arguments) {
                        let parameter_type = self.resolve_source_type(&parameter.annotation)?;
                        let value = self.expression(&argument.value, Some(parameter_type))?;
                        self.value_substitutions
                            .insert(parameter.name.clone(), value);
                    }
                    body.iter()
                        .map(|statement| {
                            let AstStatement::Expression(expression) = statement else {
                                unreachable!("expression-only method body was checked above")
                            };
                            Ok(Statement::Expression(self.expression(expression, None)?))
                        })
                        .collect::<Result<Vec<_>, Diagnostic>>()
                })();
                self.value_substitutions = previous;
                return resolved.map(|statements| Some(Statement::Sequence(Block { statements })));
            }
            return Err(Diagnostic::new(
                "E000211",
                format!("method `{name}` cannot be lowered as a unit method"),
                Some(method.span),
            ));
        };
        let Some((field, declaration)) = instance
            .fields
            .iter()
            .enumerate()
            .find(|(_, field)| field.name == field_name)
        else {
            return Err(Diagnostic::new(
                "E000211",
                format!("class `{}` has no field `{field_name}`", instance.name),
                Some(method.span),
            ));
        };
        let previous = self.value_substitutions.clone();
        for (parameter, argument) in method.parameters.iter().zip(arguments) {
            let value = self.expression(&argument.value, Some(declaration.ty))?;
            self.value_substitutions
                .insert(parameter.name.clone(), value);
        }
        let value = self.expression(update_value, Some(declaration.ty));
        self.value_substitutions = previous;
        let value = value?;
        let operator = universal_binary(operator);
        if !self.types.supports_binary(operator, declaration.ty) {
            return Err(Diagnostic::new(
                "E000202",
                format!("field `{field_name}` does not support this update operator"),
                Some(method.span),
            ));
        }
        Ok(Some(Statement::FieldUpdate {
            binding,
            field: field as u32,
            operator,
            value,
        }))
    }
}

pub(crate) fn tuple_type_id(elements: &[TypeId]) -> TypeId {
    // Reserve the upper quarter of synthetic IDs for structural tuples. The
    // stable fold makes package signature collection and body analysis agree
    // without introducing tuple identities into the universal type catalog.
    let hash = elements.iter().fold(0x811c_9dc5u32, |hash, element| {
        (hash ^ element.0).wrapping_mul(0x0100_0193)
    });
    TypeId(0xc000_0000 | (hash & 0x3fff_ffff))
}

pub(crate) fn list_type_id(element: TypeId) -> TypeId {
    let hash = (0x811c_9dc5u32 ^ element.0).wrapping_mul(0x0100_0193);
    TypeId(0x0c00_0000 | (hash & 0x03ff_ffff))
}

pub(crate) const fn set_type_id() -> TypeId {
    TypeId(0x07ff_fffe)
}

pub(crate) fn pointer_type_id(element: TypeId) -> TypeId {
    severian_universal::raw_pointer_type_id(element)
}

pub(crate) fn map_type_id(key: TypeId, value: TypeId) -> TypeId {
    // Structural maps need the same identity during package signature
    // collection and body analysis. Keep their stable IDs in a range separate
    // from structural tuples.
    let hash = [key, value].iter().fold(0x811c_9dc5u32, |hash, element| {
        (hash ^ element.0).wrapping_mul(0x0100_0193)
    });
    TypeId(0x8000_0000 | (hash & 0x3fff_ffff))
}

pub(crate) fn fallible_type_id(success: TypeId, error: TypeId) -> TypeId {
    let hash = [success, error]
        .iter()
        .fold(0x811c_9dc5u32, |hash, element| {
            (hash ^ element.0).wrapping_mul(0x0100_0193)
        });
    TypeId(0x4000_0000 | (hash & 0x3fff_ffff))
}

pub(crate) fn union_type_id(members: &[TypeId]) -> TypeId {
    let mut members = members.to_vec();
    members.sort();
    members.dedup();
    let hash = members.iter().fold(0x811c_9dc5u32, |hash, member| {
        (hash ^ member.0).wrapping_mul(0x0100_0193)
    });
    TypeId(0x2000_0000 | (hash & 0x1fff_ffff))
}

pub(crate) fn function_type_id(parameters: &[TypeId], result: TypeId) -> TypeId {
    let hash = parameters
        .iter()
        .chain(std::iter::once(&result))
        .fold(0x811c_9dc5u32, |hash, ty| {
            (hash ^ ty.0).wrapping_mul(0x0100_0193)
        });
    TypeId(0x1000_0000 | (hash & 0x0fff_ffff))
}

fn contract_failure_expression(failure: Option<&AstExpression>) -> Option<&AstExpression> {
    let failure = failure?;
    if let AstExpressionKind::Call { callee, arguments } = &failure.kind {
        if callable_path(callee).as_deref() == Some("Error") {
            return arguments.first().map(|argument| &argument.value);
        }
    }
    Some(failure)
}

fn insert_before_returns(block: &mut Block, assertions: &[Statement]) {
    let mut lowered = Vec::with_capacity(block.statements.len() + assertions.len());
    for mut statement in std::mem::take(&mut block.statements) {
        match &mut statement {
            Statement::Sequence(nested) => insert_before_returns(nested, assertions),
            Statement::If {
                then_block,
                else_block,
                ..
            } => {
                insert_before_returns(then_block, assertions);
                insert_before_returns(else_block, assertions);
            }
            Statement::While { body, .. } => insert_before_returns(body, assertions),
            Statement::Try {
                body, catch_body, ..
            } => {
                insert_before_returns(body, assertions);
                insert_before_returns(catch_body, assertions);
            }
            Statement::Match { arms, .. } => {
                for arm in arms {
                    insert_before_returns(&mut arm.body, assertions);
                }
            }
            Statement::Return(_) => lowered.extend_from_slice(assertions),
            _ => {}
        }
        lowered.push(statement);
    }
    block.statements = lowered;
}

fn insert_hook_exits(block: &mut Block, hooks: &[LoweredHook]) {
    let mut lowered = Vec::new();
    for mut statement in std::mem::take(&mut block.statements) {
        match &mut statement {
            Statement::Sequence(nested) => insert_hook_exits(nested, hooks),
            Statement::If {
                then_block,
                else_block,
                ..
            } => {
                insert_hook_exits(then_block, hooks);
                insert_hook_exits(else_block, hooks);
            }
            Statement::While { body, .. } => insert_hook_exits(body, hooks),
            Statement::Try {
                body, catch_body, ..
            } => {
                insert_hook_exits(body, hooks);
                insert_hook_exits(catch_body, hooks);
            }
            Statement::Match { arms, .. } => {
                for arm in arms {
                    insert_hook_exits(&mut arm.body, hooks);
                }
            }
            Statement::Return(Some(value)) => {
                for hook in hooks {
                    if let Some(field) = hook.result_field {
                        lowered.push(Statement::FieldSet {
                            binding: hook.context,
                            field,
                            value: value.clone(),
                        });
                    }
                }
                for hook in hooks.iter().rev() {
                    if let Some((field, duration)) = &hook.duration {
                        lowered.push(Statement::FieldSet {
                            binding: hook.context,
                            field: *field,
                            value: duration.clone(),
                        });
                    }
                    lowered.extend(hook.without_phase.statements.iter().cloned());
                }
            }
            Statement::Return(None) => {
                for hook in hooks.iter().rev() {
                    if let Some((field, duration)) = &hook.duration {
                        lowered.push(Statement::FieldSet {
                            binding: hook.context,
                            field: *field,
                            value: duration.clone(),
                        });
                    }
                    lowered.extend(hook.without_phase.statements.iter().cloned());
                }
            }
            Statement::Expression(Expression {
                kind: ExpressionKind::Throw(error),
                ..
            }) => {
                for hook in hooks {
                    if let Some(field) = hook.error_field {
                        lowered.push(Statement::FieldSet {
                            binding: hook.context,
                            field,
                            value: error.as_ref().clone(),
                        });
                    }
                }
                for hook in hooks.iter().rev() {
                    if let Some((field, duration)) = &hook.duration {
                        lowered.push(Statement::FieldSet {
                            binding: hook.context,
                            field: *field,
                            value: duration.clone(),
                        });
                    }
                    lowered.extend(hook.without_phase.statements.iter().cloned());
                }
            }
            _ => {}
        }
        lowered.push(statement);
    }
    block.statements = lowered;
}

fn synthetic_definition(
    module: &str,
    function: &severian_ast::FunctionDeclaration,
    overload_ordinal: usize,
) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path(module).0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.function.{}.{overload_ordinal}",
            function.name
        )),
    }
}

fn synthetic_test_definition(module: &str, ordinal: usize) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path(module).0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.test.{ordinal}"
        )),
    }
}

fn synthetic_runtime_definition(symbol: &str) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path("severian.runtime").0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "severian.runtime.{symbol}"
        )),
    }
}

fn synthetic_extension_definition(
    path: &str,
    span: severian_source::Span,
    types: &[TypeId],
) -> DefId {
    let concrete = types
        .iter()
        .map(|ty| ty.0.to_string())
        .collect::<Vec<_>>()
        .join(".");
    let identity = format!(
        "severian.extension.{path}.{}.{}.{}",
        span.source.0, span.start, concrete
    );
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path("severian.extensions").0,
        declaration: severian_universal::DeclarationId::from_path(&identity),
    }
}

fn synthetic_trait_definition(module: &str, name: &str) -> DefId {
    DefId {
        package: 0,
        module: severian_universal::DeclarationId::from_path(module).0,
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.trait.{name}"
        )),
    }
}

fn collect_trait_namespace_methods(
    ast: &severian_ast::Module,
) -> Result<BTreeMap<String, NamespaceTraitMethod>, Diagnostic> {
    let classes = ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Class(declaration) => Some(declaration),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut methods = BTreeMap::new();
    for declaration in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Trait(declaration) => Some(declaration),
        _ => None,
    }) {
        for method in &declaration.methods {
            if method.hook.is_some() {
                continue;
            }
            for decorator in declaration.namespaces.iter().chain(&method.decorators) {
                if !decorator.arguments.is_empty() {
                    continue;
                }
                let path = format!("{}.{}", decorator.name, method.name);
                let implementations = classes
                    .iter()
                    .filter(|class| {
                        class.traits.iter().any(|implemented| {
                            implemented.simple_name() == Some(declaration.name.as_str())
                        })
                    })
                    .filter_map(|class| {
                        class
                            .methods
                            .iter()
                            .find(|candidate| candidate.name == method.name)
                            .cloned()
                            .map(|implementation| (class.name.clone(), implementation))
                    })
                    .collect();
                let namespace_method = NamespaceTraitMethod {
                    trait_name: declaration.name.clone(),
                    declaration: method.clone(),
                    implementations,
                };
                if methods.insert(path.clone(), namespace_method).is_some() {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("namespace member `{path}` is declared more than once"),
                        Some(decorator.span),
                    ));
                }
            }
        }
    }
    Ok(methods)
}

fn collect_trait_namespace_hooks(
    ast: &severian_ast::Module,
) -> Result<BTreeMap<String, NamespaceTraitHook>, Diagnostic> {
    let classes = ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Class(declaration) => Some(declaration),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut hooks = BTreeMap::new();
    for declaration in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Trait(declaration) => Some(declaration),
        _ => None,
    }) {
        let members = declaration
            .methods
            .iter()
            .filter(|method| method.hook.is_some())
            .map(|method| {
                let implementations = classes
                    .iter()
                    .filter(|class| {
                        class.traits.iter().any(|implemented| {
                            implemented.simple_name() == Some(declaration.name.as_str())
                        })
                    })
                    .filter_map(|class| {
                        class
                            .methods
                            .iter()
                            .find(|candidate| {
                                candidate.name == method.name && candidate.hook.is_some()
                            })
                            .cloned()
                            .map(|implementation| (class.name.clone(), implementation))
                    })
                    .collect();
                NamespaceTraitHookMember {
                    method_name: method.name.clone(),
                    selectors: method
                        .decorators
                        .iter()
                        .filter(|decorator| decorator.arguments.is_empty())
                        .map(|decorator| decorator.name.clone())
                        .collect(),
                    implementations,
                }
            })
            .collect::<Vec<_>>();

        for member in &members {
            for selector in &member.selectors {
                let hook = NamespaceTraitHook {
                    trait_name: declaration.name.clone(),
                    members: vec![member.clone()],
                };
                if hooks.insert(selector.clone(), hook).is_some() {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("hook decorator `@{selector}` is declared more than once"),
                        declaration
                            .methods
                            .iter()
                            .flat_map(|method| &method.decorators)
                            .find(|decorator| decorator.name == *selector)
                            .map(|decorator| decorator.span),
                    ));
                }
            }
        }

        for namespace in &declaration.namespaces {
            if members.is_empty() {
                continue;
            }
            if !namespace.arguments.is_empty() {
                return Err(Diagnostic::new(
                    "E000218",
                    "composed hook namespace declarations do not take arguments",
                    Some(namespace.span),
                ));
            }
            let hook = NamespaceTraitHook {
                trait_name: declaration.name.clone(),
                members: members.clone(),
            };
            if hooks.insert(namespace.name.clone(), hook).is_some() {
                return Err(Diagnostic::new(
                    "E000203",
                    format!(
                        "hook decorator `@{}` is declared more than once",
                        namespace.name
                    ),
                    Some(namespace.span),
                ));
            }
        }
    }
    Ok(hooks)
}

fn collect_trait_namespace_operators(
    ast: &severian_ast::Module,
) -> Result<BTreeMap<String, NamespaceTraitOperator>, Diagnostic> {
    let classes = ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Class(declaration) => Some(declaration),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut operators = BTreeMap::new();
    for declaration in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Trait(declaration) => Some(declaration),
        _ => None,
    }) {
        for operator in &declaration.operators {
            for decorator in declaration.namespaces.iter().chain(&operator.decorators) {
                if !decorator.arguments.is_empty() {
                    continue;
                }
                let spelling = ast_operator_spelling(operator.operator);
                let path = format!("{}.{spelling}", decorator.name);
                let implementations = classes
                    .iter()
                    .filter(|class| {
                        class.traits.iter().any(|implemented| {
                            implemented.simple_name() == Some(declaration.name.as_str())
                        })
                    })
                    .filter_map(|class| {
                        class
                            .operators
                            .iter()
                            .find(|candidate| candidate.operator == operator.operator)
                            .cloned()
                            .map(|implementation| (class.name.clone(), implementation))
                    })
                    .collect();
                let namespace_operator = NamespaceTraitOperator {
                    trait_name: declaration.name.clone(),
                    declaration: operator.clone(),
                    implementations,
                };
                if operators.insert(path.clone(), namespace_operator).is_some() {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("namespace operator `{path}` is declared more than once"),
                        Some(decorator.span),
                    ));
                }
            }
        }
    }
    Ok(operators)
}

fn collect_extension_namespace_operators(
    ast: &severian_ast::Module,
) -> Result<BTreeMap<String, NamespaceExtensionOperator>, Diagnostic> {
    let mut operators = BTreeMap::new();
    for extension in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Extension(extension) => Some(extension),
        _ => None,
    }) {
        for operator in &extension.operators {
            for decorator in extension.decorators.iter().chain(&operator.decorators) {
                if !decorator.arguments.is_empty() {
                    continue;
                }
                let spelling = ast_operator_spelling(operator.operator);
                let path = format!("{}.{spelling}", decorator.name);
                let entry = NamespaceExtensionOperator {
                    namespace: decorator.name.clone(),
                    target: extension.target.clone(),
                    implementation: operator.clone(),
                };
                if operators.insert(path.clone(), entry).is_some() {
                    return Err(Diagnostic::new(
                        "E000203",
                        format!("namespace operator `{path}` is declared more than once"),
                        Some(decorator.span),
                    ));
                }
            }
        }
    }
    Ok(operators)
}

fn validate_trait_implementations(ast: &severian_ast::Module) -> Result<(), Diagnostic> {
    let traits = ast
        .items
        .iter()
        .filter_map(|item| match item {
            severian_ast::Item::Trait(declaration) => {
                Some((declaration.name.as_str(), declaration))
            }
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();

    for class in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Class(declaration) => Some(declaration),
        _ => None,
    }) {
        for implemented in &class.traits {
            let Some((trait_name, _)) = implemented.named_parts() else {
                continue;
            };
            let Some(contract) = traits.get(trait_name) else {
                continue;
            };
            for required in &contract.methods {
                let Some(provided) = class
                    .methods
                    .iter()
                    .find(|method| method.name == required.name)
                else {
                    return Err(Diagnostic::new(
                        "E000218",
                        format!(
                            "class `{}` does not implement required method `{}.{}`",
                            class.name, trait_name, required.name
                        ),
                        Some(class.span),
                    ));
                };
                let same_parameters = required.parameters.len() == provided.parameters.len()
                    && required.parameters.iter().zip(&provided.parameters).all(
                        |(required, provided)| {
                            same_type_annotation(&required.annotation, &provided.annotation)
                        },
                    );
                if !same_parameters || !same_type_annotation(&required.result, &provided.result) {
                    return Err(Diagnostic::new(
                        "E000218",
                        format!(
                            "method `{}.{}` does not match trait `{}`",
                            class.name, provided.name, trait_name
                        ),
                        Some(provided.span),
                    ));
                }
                if required.hook.is_some() != provided.hook.is_some() {
                    return Err(Diagnostic::new(
                        "E000218",
                        format!(
                            "method `{}.{}` does not match hook requirement from trait `{}`",
                            class.name, provided.name, trait_name
                        ),
                        Some(provided.span),
                    ));
                }
            }
            for required in &contract.operators {
                let Some(provided) = class
                    .operators
                    .iter()
                    .find(|operator| operator.operator == required.operator)
                else {
                    return Err(Diagnostic::new(
                        "E000218",
                        format!(
                            "class `{}` does not implement required operator `{}.{}`",
                            class.name,
                            trait_name,
                            ast_operator_spelling(required.operator)
                        ),
                        Some(class.span),
                    ));
                };
                let same_parameters = required.parameters.len() == provided.parameters.len()
                    && required.parameters.iter().zip(&provided.parameters).all(
                        |(required, provided)| {
                            same_type_annotation(&required.annotation, &provided.annotation)
                        },
                    );
                if !same_parameters || !same_type_annotation(&required.result, &provided.result) {
                    return Err(Diagnostic::new(
                        "E000218",
                        format!(
                            "operator `{}.{}` does not match trait `{}`",
                            class.name,
                            ast_operator_spelling(provided.operator),
                            trait_name
                        ),
                        Some(provided.span),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_class_declarations(ast: &severian_ast::Module) -> Result<(), Diagnostic> {
    for function in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Function(declaration) => Some(declaration),
        _ => None,
    }) {
        validate_hook_context(function)?;
    }
    for method in ast.items.iter().flat_map(|item| match item {
        severian_ast::Item::Trait(declaration) => declaration.methods.iter(),
        _ => [].iter(),
    }) {
        validate_hook_context(method)?;
    }
    for class in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Class(declaration) => Some(declaration),
        _ => None,
    }) {
        let fields = class
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<BTreeSet<_>>();
        for method in class.constructors.iter().chain(&class.methods) {
            validate_hook_context(method)?;
            if let Some(parameter) = method
                .parameters
                .iter()
                .find(|parameter| fields.contains(parameter.name.as_str()))
            {
                return Err(Diagnostic::new(
                    "E000220",
                    format!(
                        "parameter `{}` shadows a field of class `{}`",
                        parameter.name, class.name
                    ),
                    Some(parameter.span),
                ));
            }
        }
        for operator in &class.operators {
            if let Some(parameter) = operator
                .parameters
                .iter()
                .find(|parameter| fields.contains(parameter.name.as_str()))
            {
                return Err(Diagnostic::new(
                    "E000220",
                    format!(
                        "parameter `{}` shadows a field of class `{}`",
                        parameter.name, class.name
                    ),
                    Some(parameter.span),
                ));
            }
        }
    }
    Ok(())
}

fn validate_hook_context(function: &severian_ast::FunctionDeclaration) -> Result<(), Diagnostic> {
    let Some(hook) = &function.hook else {
        return Ok(());
    };
    if function
        .parameters
        .iter()
        .any(|parameter| parameter.name == hook.context)
    {
        return Ok(());
    }
    Err(Diagnostic::new(
        "E000218",
        format!(
            "hook context `{}` is not a parameter of `{}`",
            hook.context, function.name
        ),
        Some(hook.span),
    ))
}

fn same_type_annotation(left: &TypeAnnotation, right: &TypeAnnotation) -> bool {
    match (&left.kind, &right.kind) {
        (
            severian_ast::TypeAnnotationKind::Named {
                name: left_name,
                arguments: left_arguments,
            },
            severian_ast::TypeAnnotationKind::Named {
                name: right_name,
                arguments: right_arguments,
            },
        ) => {
            left_name == right_name
                && left_arguments.len() == right_arguments.len()
                && left_arguments
                    .iter()
                    .zip(right_arguments)
                    .all(|(left, right)| same_type_annotation(left, right))
        }
        (
            severian_ast::TypeAnnotationKind::Union(left),
            severian_ast::TypeAnnotationKind::Union(right),
        ) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same_type_annotation(left, right))
        }
        (
            severian_ast::TypeAnnotationKind::Function {
                parameters: left_parameters,
                result: left_result,
            },
            severian_ast::TypeAnnotationKind::Function {
                parameters: right_parameters,
                result: right_result,
            },
        ) => {
            left_parameters.len() == right_parameters.len()
                && left_parameters
                    .iter()
                    .zip(right_parameters)
                    .all(|(left, right)| same_type_annotation(left, right))
                && same_type_annotation(left_result, right_result)
        }
        _ => false,
    }
}

fn callable_path(expression: &AstExpression) -> Option<String> {
    match &expression.kind {
        AstExpressionKind::Name(name) => Some(name.clone()),
        AstExpressionKind::Member { object, name } => {
            Some(format!("{}.{}", callable_path(object)?, name))
        }
        _ => None,
    }
}

fn substituted_expression(
    expression: &AstExpression,
    substitutions: &BTreeMap<String, AstExpression>,
) -> AstExpression {
    let mut result = expression.clone();
    match &mut result.kind {
        AstExpressionKind::Name(name) => {
            if let Some(value) = substitutions.get(name) {
                return value.clone();
            }
        }
        AstExpressionKind::List(values)
        | AstExpressionKind::Set(values)
        | AstExpressionKind::Tuple(values) => {
            for value in values {
                *value = substituted_expression(value, substitutions);
            }
        }
        AstExpressionKind::Map(entries) => {
            for entry in entries {
                entry.key = substituted_expression(&entry.key, substitutions);
                entry.value = substituted_expression(&entry.value, substitutions);
            }
        }
        AstExpressionKind::Member { object, .. }
        | AstExpressionKind::TypeApplication { callee: object, .. }
        | AstExpressionKind::Async {
            expression: object, ..
        }
        | AstExpressionKind::Await { expression: object }
        | AstExpressionKind::Throw { error: object }
        | AstExpressionKind::Unary {
            operand: object, ..
        } => **object = substituted_expression(object, substitutions),
        AstExpressionKind::Index { object, index } => {
            **object = substituted_expression(object, substitutions);
            **index = substituted_expression(index, substitutions);
        }
        AstExpressionKind::Slice {
            object,
            start,
            end,
            step,
            ..
        } => {
            **object = substituted_expression(object, substitutions);
            for bound in [start, end, step].into_iter().flatten() {
                **bound = substituted_expression(bound, substitutions);
            }
        }
        AstExpressionKind::Call { callee, arguments } => {
            **callee = substituted_expression(callee, substitutions);
            for argument in arguments {
                argument.value = substituted_expression(&argument.value, substitutions);
            }
        }
        AstExpressionKind::Conditional {
            value,
            condition,
            fallback,
        } => {
            **value = substituted_expression(value, substitutions);
            **condition = substituted_expression(condition, substitutions);
            **fallback = substituted_expression(fallback, substitutions);
        }
        AstExpressionKind::Fallback { value, fallback }
        | AstExpressionKind::Binary {
            left: value,
            right: fallback,
            ..
        } => {
            **value = substituted_expression(value, substitutions);
            **fallback = substituted_expression(fallback, substitutions);
        }
        AstExpressionKind::Literal(_)
        | AstExpressionKind::Lambda { .. }
        | AstExpressionKind::Mock { .. }
        | AstExpressionKind::ListComprehension { .. }
        | AstExpressionKind::SetComprehension { .. }
        | AstExpressionKind::MapComprehension { .. } => {}
    }
    result
}

fn constant_integer_expression(expression: &AstExpression) -> Option<i64> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::Integer(value)) => value.parse().ok(),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Negative,
            operand,
        } => constant_integer_expression(operand)?.checked_neg(),
        AstExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = constant_integer_expression(left)?;
            let right = constant_integer_expression(right)?;
            match operator {
                AstBinaryOperator::Pipe => Some(left | right),
                AstBinaryOperator::BitwiseAnd => Some(left & right),
                AstBinaryOperator::BitwiseXor => Some(left ^ right),
                AstBinaryOperator::Add => left.checked_add(right),
                AstBinaryOperator::Subtract => left.checked_sub(right),
                AstBinaryOperator::Multiply => left.checked_mul(right),
                AstBinaryOperator::Divide => left.checked_div(right),
                AstBinaryOperator::Remainder => left.checked_rem(right),
                _ => None,
            }
        }
        _ => None,
    }
}

fn constant_boolean_expression(expression: &AstExpression) -> Option<bool> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::Boolean(value)) => Some(*value),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Not,
            operand,
        } => constant_boolean_expression(operand).map(|value| !value),
        AstExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = constant_integer_expression(left)?;
            let right = constant_integer_expression(right)?;
            Some(match operator {
                AstBinaryOperator::Equal => left == right,
                AstBinaryOperator::NotEqual => left != right,
                AstBinaryOperator::Less => left < right,
                AstBinaryOperator::LessEqual => left <= right,
                AstBinaryOperator::Greater => left > right,
                AstBinaryOperator::GreaterEqual => left >= right,
                _ => return None,
            })
        }
        _ => None,
    }
}

fn collect_thrown_error_names(statements: &[AstStatement], names: &mut BTreeSet<String>) {
    for statement in statements {
        match statement {
            AstStatement::Expression(expression) => {
                collect_thrown_error_name(expression, names);
            }
            AstStatement::If {
                then_block,
                else_block,
                ..
            } => {
                collect_thrown_error_names(then_block, names);
                collect_thrown_error_names(else_block, names);
            }
            AstStatement::While { body, .. }
            | AstStatement::For { body, .. }
            | AstStatement::Unsafe { body, .. }
            | AstStatement::FallibleElse { body, .. } => {
                collect_thrown_error_names(body, names);
            }
            AstStatement::Try {
                body, catch_body, ..
            } => {
                collect_thrown_error_names(body, names);
                collect_thrown_error_names(catch_body, names);
            }
            AstStatement::Match { cases, .. } => {
                for case in cases {
                    collect_thrown_error_names(&case.body, names);
                }
            }
            AstStatement::Select {
                cases, error_body, ..
            } => {
                for case in cases {
                    collect_thrown_error_names(&case.body, names);
                }
                collect_thrown_error_names(error_body, names);
            }
            AstStatement::Binding(_)
            | AstStatement::Destructure { .. }
            | AstStatement::FieldAssignment { .. }
            | AstStatement::IndexAssignment { .. }
            | AstStatement::Defer { .. }
            | AstStatement::Return { .. }
            | AstStatement::Assert { .. }
            | AstStatement::Break { .. }
            | AstStatement::Continue { .. } => {}
        }
    }
}

fn collect_expected_throw_functions(statements: &[AstStatement], names: &mut BTreeSet<String>) {
    for statement in statements {
        match statement {
            AstStatement::Expression(AstExpression {
                kind: AstExpressionKind::Call { callee, arguments },
                ..
            }) if callable_path(callee).as_deref() == Some("throws") => {
                if let [argument] = arguments.as_slice() {
                    if let AstExpressionKind::Call { callee, .. } = &argument.value.kind {
                        if let Some(name) = callable_path(callee) {
                            names.insert(name);
                        }
                    }
                }
            }
            AstStatement::If {
                then_block,
                else_block,
                ..
            } => {
                collect_expected_throw_functions(then_block, names);
                collect_expected_throw_functions(else_block, names);
            }
            AstStatement::While { body, .. }
            | AstStatement::For { body, .. }
            | AstStatement::Unsafe { body, .. }
            | AstStatement::FallibleElse { body, .. } => {
                collect_expected_throw_functions(body, names);
            }
            AstStatement::Try {
                body, catch_body, ..
            } => {
                collect_expected_throw_functions(body, names);
                collect_expected_throw_functions(catch_body, names);
            }
            AstStatement::Match { cases, .. } => {
                for case in cases {
                    collect_expected_throw_functions(&case.body, names);
                }
            }
            AstStatement::Select {
                cases, error_body, ..
            } => {
                for case in cases {
                    collect_expected_throw_functions(&case.body, names);
                }
                collect_expected_throw_functions(error_body, names);
            }
            _ => {}
        }
    }
}

fn collect_thrown_error_name(expression: &AstExpression, names: &mut BTreeSet<String>) {
    let AstExpressionKind::Throw { error } = &expression.kind else {
        return;
    };
    let name = match &error.kind {
        AstExpressionKind::Call { callee, .. } => callable_path(callee),
        AstExpressionKind::Name(name) => Some(name.clone()),
        _ => None,
    };
    if let Some(name) = name {
        names.insert(name);
    }
}

fn collect_expression_names(expression: &AstExpression, names: &mut BTreeSet<String>) {
    match &expression.kind {
        AstExpressionKind::Name(name) => {
            names.insert(name.clone());
        }
        AstExpressionKind::Lambda { parameters, body } => {
            let mut nested = BTreeSet::new();
            collect_expression_names(body, &mut nested);
            for parameter in parameters {
                nested.remove(parameter);
            }
            names.extend(nested);
        }
        AstExpressionKind::List(values)
        | AstExpressionKind::Set(values)
        | AstExpressionKind::Tuple(values) => {
            for value in values {
                collect_expression_names(value, names);
            }
        }
        AstExpressionKind::Map(entries) => {
            for entry in entries {
                collect_expression_names(&entry.key, names);
                collect_expression_names(&entry.value, names);
            }
        }
        AstExpressionKind::ListComprehension { value, clauses }
        | AstExpressionKind::SetComprehension { value, clauses } => {
            collect_expression_names(value, names);
            for clause in clauses {
                collect_expression_names(&clause.iterable, names);
                if let Some(condition) = &clause.condition {
                    collect_expression_names(condition, names);
                }
                for binding in &clause.bindings {
                    names.remove(binding);
                }
            }
        }
        AstExpressionKind::MapComprehension { key, value, clauses } => {
            collect_expression_names(key, names);
            collect_expression_names(value, names);
            for clause in clauses {
                collect_expression_names(&clause.iterable, names);
                if let Some(condition) = &clause.condition {
                    collect_expression_names(condition, names);
                }
                for binding in &clause.bindings {
                    names.remove(binding);
                }
            }
        }
        AstExpressionKind::Mock { cases, fallback } => {
            for case in cases {
                collect_expression_names(&case.call, names);
                collect_expression_names(&case.result, names);
            }
            collect_expression_names(fallback, names);
        }
        AstExpressionKind::Member { object, .. }
        | AstExpressionKind::TypeApplication { callee: object, .. } => {
            collect_expression_names(object, names);
        }
        AstExpressionKind::Index { object, index } => {
            collect_expression_names(object, names);
            collect_expression_names(index, names);
        }
        AstExpressionKind::Slice {
            object,
            start,
            end,
            step,
            ..
        } => {
            collect_expression_names(object, names);
            for value in [start, end, step].into_iter().flatten() {
                collect_expression_names(value, names);
            }
        }
        AstExpressionKind::Call { callee, arguments } => {
            collect_expression_names(callee, names);
            for argument in arguments {
                collect_expression_names(&argument.value, names);
            }
        }
        AstExpressionKind::Async { expression, .. }
        | AstExpressionKind::Await { expression }
        | AstExpressionKind::Unary {
            operand: expression,
            ..
        } => collect_expression_names(expression, names),
        AstExpressionKind::Conditional {
            value,
            condition,
            fallback,
        } => {
            collect_expression_names(value, names);
            collect_expression_names(condition, names);
            collect_expression_names(fallback, names);
        }
        AstExpressionKind::Fallback { value, fallback }
        | AstExpressionKind::Binary {
            left: value,
            right: fallback,
            ..
        } => {
            collect_expression_names(value, names);
            collect_expression_names(fallback, names);
        }
        AstExpressionKind::Throw { error } => collect_expression_names(error, names),
        AstExpressionKind::Literal(_) => {}
    }
}

fn operator_namespaces(decorators: &[severian_ast::Decorator]) -> BTreeMap<String, Vec<String>> {
    let mut namespaces = BTreeMap::<String, Vec<String>>::new();
    for decorator in decorators {
        for argument in &decorator.arguments {
            let severian_ast::DecoratorValue::Name(operator) = &argument.value else {
                continue;
            };
            if argument.name.is_none() && is_binary_operator_spelling(operator) {
                namespaces
                    .entry(operator.clone())
                    .or_default()
                    .push(decorator.name.clone());
            }
        }
    }
    namespaces
}

fn is_binary_operator_spelling(operator: &str) -> bool {
    matches!(
        operator,
        "|" | "&"
            | "^"
            | "+"
            | "-"
            | "*"
            | "/"
            | "%"
            | "**"
            | "=="
            | "!="
            | "<"
            | "<="
            | ">"
            | ">="
            | "in"
            | "and"
            | "or"
    )
}

fn ast_binary_spelling(operator: AstBinaryOperator) -> &'static str {
    match operator {
        AstBinaryOperator::Pipe => "|",
        AstBinaryOperator::BitwiseAnd => "&",
        AstBinaryOperator::BitwiseXor => "^",
        AstBinaryOperator::Add => "+",
        AstBinaryOperator::Subtract => "-",
        AstBinaryOperator::Multiply => "*",
        AstBinaryOperator::Divide => "/",
        AstBinaryOperator::Remainder => "%",
        AstBinaryOperator::Power => "**",
        AstBinaryOperator::Equal | AstBinaryOperator::Identity => "==",
        AstBinaryOperator::NotEqual => "!=",
        AstBinaryOperator::Less => "<",
        AstBinaryOperator::LessEqual => "<=",
        AstBinaryOperator::Greater => ">",
        AstBinaryOperator::GreaterEqual => ">=",
        AstBinaryOperator::Contains => "in",
        AstBinaryOperator::And => "and",
        AstBinaryOperator::Or => "or",
    }
}

fn ast_operator_spelling(operator: severian_ast::OperatorSyntax) -> &'static str {
    use severian_ast::OperatorSyntax as Operator;
    match operator {
        Operator::Pipe => "|",
        Operator::BitwiseAnd => "&",
        Operator::BitwiseXor => "^",
        Operator::Plus => "+",
        Operator::Minus => "-",
        Operator::Multiply => "*",
        Operator::Divide => "/",
        Operator::Remainder => "%",
        Operator::Power => "**",
        Operator::Equal => "==",
        Operator::NotEqual => "!=",
        Operator::Less => "<",
        Operator::LessEqual => "<=",
        Operator::Greater => ">",
        Operator::GreaterEqual => ">=",
        Operator::Contains => "in",
        Operator::And => "and",
        Operator::Or => "or",
        Operator::Not => "not",
    }
}

fn class_application(expression: &AstExpression) -> Option<(&str, &[TypeAnnotation])> {
    let AstExpressionKind::TypeApplication { callee, arguments } = &expression.kind else {
        return None;
    };
    let AstExpressionKind::Name(name) = &callee.kind else {
        return None;
    };
    Some((name, arguments))
}

fn static_integer(expression: &AstExpression) -> Option<i64> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::Integer(value)) => value.parse().ok(),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Negative,
            operand,
        } => static_integer(operand)?.checked_neg(),
        _ => None,
    }
}

fn static_integer_in(
    expression: &AstExpression,
    environment: &BTreeMap<String, i64>,
) -> Option<i64> {
    match &expression.kind {
        AstExpressionKind::Name(name) => environment.get(name).copied(),
        AstExpressionKind::Literal(AstLiteral::Integer(value)) => value.parse().ok(),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Negative,
            operand,
        } => static_integer_in(operand, environment)?.checked_neg(),
        AstExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = static_integer_in(left, environment)?;
            let right = static_integer_in(right, environment)?;
            match operator {
                AstBinaryOperator::Pipe => Some(left | right),
                AstBinaryOperator::BitwiseAnd => Some(left & right),
                AstBinaryOperator::BitwiseXor => Some(left ^ right),
                AstBinaryOperator::Add => left.checked_add(right),
                AstBinaryOperator::Subtract => left.checked_sub(right),
                AstBinaryOperator::Multiply => left.checked_mul(right),
                AstBinaryOperator::Divide => (right != 0).then(|| left / right),
                AstBinaryOperator::Remainder => (right != 0).then(|| left % right),
                _ => None,
            }
        }
        _ => None,
    }
}

fn static_boolean(expression: &AstExpression, environment: &BTreeMap<String, i64>) -> Option<bool> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::Boolean(value)) => Some(*value),
        AstExpressionKind::Unary {
            operator: AstUnaryOperator::Not,
            operand,
        } => Some(!static_boolean(operand, environment)?),
        AstExpressionKind::Binary {
            operator: AstBinaryOperator::And,
            left,
            right,
        } => Some(static_boolean(left, environment)? && static_boolean(right, environment)?),
        AstExpressionKind::Binary {
            operator: AstBinaryOperator::Or,
            left,
            right,
        } => Some(static_boolean(left, environment)? || static_boolean(right, environment)?),
        AstExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = static_integer_in(left, environment)?;
            let right = static_integer_in(right, environment)?;
            Some(match operator {
                AstBinaryOperator::Equal | AstBinaryOperator::Identity => left == right,
                AstBinaryOperator::NotEqual => left != right,
                AstBinaryOperator::Less => left < right,
                AstBinaryOperator::LessEqual => left <= right,
                AstBinaryOperator::Greater => left > right,
                AstBinaryOperator::GreaterEqual => left >= right,
                _ => return None,
            })
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StaticLoopControl {
    Continue,
    Break,
    Fallthrough,
}

fn specialize_loop_body(
    statements: &[AstStatement],
    environment: &mut BTreeMap<String, i64>,
) -> (Vec<AstStatement>, StaticLoopControl) {
    let mut output = Vec::new();
    for statement in statements {
        match statement {
            AstStatement::Break { .. } => return (output, StaticLoopControl::Break),
            AstStatement::Continue { .. } => return (output, StaticLoopControl::Continue),
            AstStatement::Binding(binding) => {
                if let Some(value) = static_integer_in(&binding.value, environment) {
                    environment.insert(binding.name.clone(), value);
                }
                output.push(statement.clone());
            }
            AstStatement::If {
                condition,
                then_block,
                else_block,
                ..
            } if static_boolean(condition, environment).is_some() => {
                let selected = if static_boolean(condition, environment) == Some(true) {
                    then_block
                } else {
                    else_block
                };
                let (specialized, control) = specialize_loop_body(selected, environment);
                output.extend(specialized);
                if control != StaticLoopControl::Fallthrough {
                    return (output, control);
                }
            }
            _ => output.push(statement.clone()),
        }
    }
    (output, StaticLoopControl::Fallthrough)
}

fn static_range_values(iterable: &AstExpression) -> Option<Vec<i64>> {
    let AstExpressionKind::Call { callee, arguments } = &iterable.kind else {
        return None;
    };
    if callable_path(callee).as_deref() != Some("range") {
        return None;
    }
    let (start, end) = match arguments.as_slice() {
        [end] => (0, static_integer(&end.value)?),
        [start, end] => (static_integer(&start.value)?, static_integer(&end.value)?),
        _ => return None,
    };
    let count = end.saturating_sub(start).max(0);
    (count <= 10_000).then(|| (start..end).collect())
}

fn increment_before_continue(block: &mut Block, increment: &Statement) {
    let statements = std::mem::take(&mut block.statements);
    for mut statement in statements {
        match &mut statement {
            Statement::Continue { .. } => block.statements.push(increment.clone()),
            Statement::Sequence(sequence) => increment_before_continue(sequence, increment),
            Statement::If {
                then_block,
                else_block,
                ..
            } => {
                increment_before_continue(then_block, increment);
                increment_before_continue(else_block, increment);
            }
            Statement::Match { arms, .. } => {
                for arm in arms {
                    increment_before_continue(&mut arm.body, increment);
                }
            }
            Statement::Try {
                body, catch_body, ..
            } => {
                increment_before_continue(body, increment);
                increment_before_continue(catch_body, increment);
            }
            _ => {}
        }
        block.statements.push(statement);
    }
}

fn resolve_type_annotation(
    types: &TypeContext,
    annotation: &TypeAnnotation,
) -> Result<TypeId, Diagnostic> {
    if matches!(
        annotation.kind,
        severian_ast::TypeAnnotationKind::Function { .. }
    ) {
        return Err(Diagnostic::new(
            "E000204",
            "function types require semantic structural resolution",
            Some(annotation.span),
        ));
    }
    if let severian_ast::TypeAnnotationKind::Union(members) = &annotation.kind {
        let mut concrete = members
            .iter()
            .filter_map(|member| {
                let name = member.simple_name()?;
                (!matches!(name, "None" | "absent")).then_some(member)
            })
            .map(|member| resolve_type_annotation(types, member))
            .collect::<Result<Vec<_>, _>>()?;
        concrete.sort();
        concrete.dedup();
        if let [only] = concrete.as_slice() {
            return Ok(*only);
        }
        if concrete.is_empty() {
            return types.resolve_name("None").ok_or_else(|| {
                Diagnostic::new("E000204", "unknown type `None`", Some(annotation.span))
            });
        }
        return Err(Diagnostic::new(
            "E000204",
            "unions with multiple concrete representations are not implemented",
            Some(annotation.span),
        ));
    }
    let Some(name) = annotation.simple_name() else {
        return Err(Diagnostic::new(
            "E000204",
            "this source type form is not yet supported by universal resolution",
            Some(annotation.span),
        ));
    };
    let name = if name == "absent" { "None" } else { name };
    types.resolve_name(name).ok_or_else(|| {
        Diagnostic::new(
            "E000204",
            format!("unknown type `{name}`"),
            Some(annotation.span),
        )
    })
}

fn universal_boundary(type_id: TypeId) -> BoundaryType {
    BoundaryType {
        ty: type_id,
        modifiers: Vec::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ConversionRank {
    Exact,
    Widening(u16),
    General,
}

fn conversion_rank(
    types: &TypeContext,
    actual: TypeId,
    expected: TypeId,
) -> Option<ConversionRank> {
    if actual == expected {
        return Some(ConversionRank::Exact);
    }
    if !types.assignable(actual, expected) {
        return None;
    }
    let actual = types.primitive(actual)?.representation;
    let expected = types.primitive(expected)?.representation;
    match (actual, expected) {
        (
            severian_universal::PrimitiveRepresentation::Integer {
                bits: severian_universal::IntegerWidth::Fixed(actual),
                ..
            },
            severian_universal::PrimitiveRepresentation::Integer {
                bits: severian_universal::IntegerWidth::Fixed(expected),
                ..
            },
        )
        | (
            severian_universal::PrimitiveRepresentation::Float {
                format: severian_universal::FloatFormat::Ieee(actual),
            },
            severian_universal::PrimitiveRepresentation::Float {
                format: severian_universal::FloatFormat::Ieee(expected),
            },
        ) => Some(ConversionRank::Widening(expected - actual)),
        _ => Some(ConversionRank::General),
    }
}

fn expression_conversion_rank(
    types: &TypeContext,
    expression: &Expression,
    expected: TypeId,
) -> Option<ConversionRank> {
    match &expression.kind {
        ExpressionKind::Convert { conversion, .. } if conversion.to == expected => {
            conversion_rank(types, conversion.from, expected)
        }
        _ => conversion_rank(types, expression.type_id, expected),
    }
}

fn dominates(left: &[ConversionRank], right: &[ConversionRank]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| left <= right)
        && left.iter().zip(right).any(|(left, right)| left < right)
}

fn test_mode(
    name: &str,
    span: severian_source::Span,
) -> Result<severian_hir::TestMode, Diagnostic> {
    match name {
        "property" => Ok(severian_hir::TestMode::Property),
        "cases" => Ok(severian_hir::TestMode::Cases),
        "fuzz" => Ok(severian_hir::TestMode::Fuzz),
        "model" => Ok(severian_hir::TestMode::Model),
        "differential" => Ok(severian_hir::TestMode::Differential),
        "bench" | "benchmark" => Ok(severian_hir::TestMode::Benchmark),
        "chaos" => Ok(severian_hir::TestMode::Chaos),
        "profile" => Ok(severian_hir::TestMode::Profile),
        "compiler" => Ok(severian_hir::TestMode::Compiler),
        "integ" | "integration" => Ok(severian_hir::TestMode::Integration),
        name if name.starts_with("timeout:") => {
            let value = name.trim_start_matches("timeout:");
            let split = value
                .find(|character: char| character.is_ascii_alphabetic())
                .ok_or_else(|| {
                    Diagnostic::new("E000217", "invalid test timeout", Some(span))
                })?;
            let (magnitude, suffix) = value.split_at(split);
            Ok(severian_hir::TestMode::Timeout(duration_nanos(
                magnitude, suffix, span,
            )?))
        }
        _ => Err(Diagnostic::new(
            "E000213",
            format!("unknown test runner `{name}`"),
            Some(span),
        )),
    }
}

fn duration_nanos(
    magnitude: &str,
    suffix: &str,
    span: severian_source::Span,
) -> Result<u128, Diagnostic> {
    let multiplier = match suffix {
        "ns" => 1.0,
        "us" => 1_000.0,
        "ms" => 1_000_000.0,
        "s" => 1_000_000_000.0,
        "min" => 60_000_000_000.0,
        "hr" => 3_600_000_000_000.0,
        "day" => 86_400_000_000_000.0,
        _ => {
            return Err(Diagnostic::new(
                "E000217",
                "a timeout must use a time unit",
                Some(span),
            ))
        }
    };
    magnitude
        .parse::<f64>()
        .ok()
        .map(|value| (value * multiplier).round())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| value as u128)
        .ok_or_else(|| Diagnostic::new("E000217", "invalid timeout duration", Some(span)))
}

fn profile_duration_expectation(
    contract: &severian_ast::FunctionContract,
) -> Result<severian_hir::TestExpectation, Diagnostic> {
    let AstExpressionKind::Binary {
        operator,
        left,
        right,
    } = &contract.condition.kind
    else {
        return Err(Diagnostic::new(
            "E000217",
            "a profile contract must compare `time` with a duration literal",
            Some(contract.condition.span),
        ));
    };
    let left_is_time = matches!(&left.kind, AstExpressionKind::Name(name) if name == "time");
    let right_is_time = matches!(&right.kind, AstExpressionKind::Name(name) if name == "time");
    let (literal, comparison) = match (left_is_time, right_is_time) {
        (true, false) => (right.as_ref(), profile_comparison(*operator)),
        (false, true) => (
            left.as_ref(),
            profile_comparison(*operator).map(reverse_comparison),
        ),
        _ => {
            return Err(Diagnostic::new(
                "E000217",
                "a profile contract must compare exactly one `time` value",
                Some(contract.condition.span),
            ))
        }
    };
    let comparison = comparison.ok_or_else(|| {
        Diagnostic::new(
            "E000217",
            "profile timing contracts support `<`, `<=`, `>`, and `>=`",
            Some(contract.condition.span),
        )
    })?;
    let AstExpressionKind::Literal(AstLiteral::Measured { magnitude, suffix }) = &literal.kind
    else {
        return Err(Diagnostic::new(
            "E000217",
            "a profile timing threshold must be a duration literal",
            Some(literal.span),
        ));
    };
    let multiplier = match suffix.as_str() {
        "ns" => 1.0,
        "us" => 1_000.0,
        "ms" => 1_000_000.0,
        "s" => 1_000_000_000.0,
        "min" => 60_000_000_000.0,
        "hr" => 3_600_000_000_000.0,
        "day" => 86_400_000_000_000.0,
        _ => {
            return Err(Diagnostic::new(
                "E000217",
                "a profile timing threshold must use a time unit",
                Some(literal.span),
            ))
        }
    };
    let threshold_nanos = magnitude
        .parse::<f64>()
        .ok()
        .map(|value| (value * multiplier).round())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| value as u128)
        .ok_or_else(|| {
            Diagnostic::new(
                "E000217",
                "invalid profile duration threshold",
                Some(literal.span),
            )
        })?;
    let message = contract_failure_expression(contract.failure.as_ref())
        .and_then(string_literal)
        .unwrap_or("profile timing contract failed")
        .to_owned();
    Ok(severian_hir::TestExpectation::ProfileDuration {
        comparison,
        threshold_nanos,
        message,
    })
}

fn profile_statement_expectation(
    statement: &AstStatement,
) -> Result<Option<severian_hir::TestExpectation>, Diagnostic> {
    let AstStatement::Expression(AstExpression {
        kind: AstExpressionKind::Call { callee, arguments },
        ..
    }) = statement
    else {
        return Ok(None);
    };
    if callable_path(callee).as_deref() != Some("expect") {
        return Ok(None);
    }
    let [argument] = arguments.as_slice() else {
        return Ok(None);
    };
    let AstExpressionKind::Binary {
        operator,
        left,
        right,
    } = &argument.value.kind
    else {
        return Ok(None);
    };
    let profile_field = |expression: &AstExpression| match &expression.kind {
        AstExpressionKind::Member { object, name }
            if matches!(&object.kind, AstExpressionKind::Name(root) if root == "profile") =>
        {
            Some(name.clone())
        }
        _ => None,
    };
    let (field, literal, comparison) = if let Some(field) = profile_field(left) {
        (field, right.as_ref(), profile_comparison(*operator))
    } else if let Some(field) = profile_field(right) {
        (
            field,
            left.as_ref(),
            profile_comparison(reverse_ast_comparison(*operator)),
        )
    } else {
        return Ok(None);
    };
    let comparison = comparison.ok_or_else(|| {
        Diagnostic::new(
            "E000217",
            "profile expectations support `<`, `<=`, `>`, and `>=`",
            Some(argument.value.span),
        )
    })?;
    let AstExpressionKind::Literal(AstLiteral::Measured { magnitude, suffix }) = &literal.kind
    else {
        return Err(Diagnostic::new(
            "E000217",
            "profile expectations require a measured literal",
            Some(literal.span),
        ));
    };
    match field.as_str() {
        "time" => Ok(Some(severian_hir::TestExpectation::ProfileDuration {
            comparison,
            threshold_nanos: duration_nanos(magnitude, suffix, literal.span)?,
            message: "profile time expectation failed".into(),
        })),
        "memory" => Ok(Some(severian_hir::TestExpectation::ProfileMemory {
            comparison,
            threshold_bytes: data_bytes(magnitude, suffix, literal.span)?,
            message: "profile memory expectation failed".into(),
        })),
        _ => Err(Diagnostic::new(
            "E000217",
            format!("unknown profile measurement `{field}`"),
            Some(argument.value.span),
        )),
    }
}

fn reverse_ast_comparison(operator: AstBinaryOperator) -> AstBinaryOperator {
    match operator {
        AstBinaryOperator::Less => AstBinaryOperator::Greater,
        AstBinaryOperator::LessEqual => AstBinaryOperator::GreaterEqual,
        AstBinaryOperator::Greater => AstBinaryOperator::Less,
        AstBinaryOperator::GreaterEqual => AstBinaryOperator::LessEqual,
        other => other,
    }
}

fn data_bytes(
    magnitude: &str,
    suffix: &str,
    span: severian_source::Span,
) -> Result<u128, Diagnostic> {
    let multiplier = match suffix {
        "B" => 1.0,
        "KB" => 1_000.0,
        "MB" => 1_000_000.0,
        "GB" => 1_000_000_000.0,
        "KiB" => 1_024.0,
        "MiB" => 1_048_576.0,
        "GiB" => 1_073_741_824.0,
        _ => {
            return Err(Diagnostic::new(
                "E000217",
                "profile memory thresholds require a data unit",
                Some(span),
            ))
        }
    };
    magnitude
        .parse::<f64>()
        .ok()
        .map(|value| (value * multiplier).round())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| value as u128)
        .ok_or_else(|| Diagnostic::new("E000217", "invalid memory threshold", Some(span)))
}

fn profile_comparison(operator: AstBinaryOperator) -> Option<severian_hir::DurationComparison> {
    Some(match operator {
        AstBinaryOperator::Less => severian_hir::DurationComparison::Less,
        AstBinaryOperator::LessEqual => severian_hir::DurationComparison::LessEqual,
        AstBinaryOperator::Greater => severian_hir::DurationComparison::Greater,
        AstBinaryOperator::GreaterEqual => severian_hir::DurationComparison::GreaterEqual,
        _ => return None,
    })
}

fn reverse_comparison(
    comparison: severian_hir::DurationComparison,
) -> severian_hir::DurationComparison {
    match comparison {
        severian_hir::DurationComparison::Less => severian_hir::DurationComparison::Greater,
        severian_hir::DurationComparison::LessEqual => {
            severian_hir::DurationComparison::GreaterEqual
        }
        severian_hir::DurationComparison::Greater => severian_hir::DurationComparison::Less,
        severian_hir::DurationComparison::GreaterEqual => {
            severian_hir::DurationComparison::LessEqual
        }
    }
}

fn internal_name_part(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlFlow {
    FallsThrough,
    Returns,
}

fn block_flow(statements: &[AstStatement]) -> ControlFlow {
    for statement in statements {
        let flow = match statement {
            AstStatement::Return { .. }
            | AstStatement::Expression(AstExpression {
                kind: AstExpressionKind::Throw { .. },
                ..
            }) => ControlFlow::Returns,
            AstStatement::If {
                then_block,
                else_block,
                ..
            } if !else_block.is_empty()
                && block_flow(then_block) == ControlFlow::Returns
                && block_flow(else_block) == ControlFlow::Returns =>
            {
                ControlFlow::Returns
            }
            AstStatement::Match { cases, .. }
                if !cases.is_empty()
                    && cases
                        .iter()
                        .all(|case| block_flow(&case.body) == ControlFlow::Returns) =>
            {
                ControlFlow::Returns
            }
            AstStatement::Unsafe { body, .. } if block_flow(body) == ControlFlow::Returns => {
                ControlFlow::Returns
            }
            AstStatement::Try {
                body, catch_body, ..
            } if block_flow(body) == ControlFlow::Returns
                && block_flow(catch_body) == ControlFlow::Returns =>
            {
                ControlFlow::Returns
            }
            AstStatement::Binding(_)
            | AstStatement::Destructure { .. }
            | AstStatement::FieldAssignment { .. }
            | AstStatement::IndexAssignment { .. }
            | AstStatement::Expression(_)
            | AstStatement::Defer { .. }
            | AstStatement::Assert { .. }
            | AstStatement::Unsafe { .. }
            | AstStatement::Try { .. }
            | AstStatement::FallibleElse { .. }
            | AstStatement::If { .. }
            | AstStatement::While { .. }
            | AstStatement::For { .. }
            | AstStatement::Break { .. }
            | AstStatement::Continue { .. }
            | AstStatement::Match { .. }
            | AstStatement::Select { .. } => ControlFlow::FallsThrough,
        };
        if flow == ControlFlow::Returns {
            return flow;
        }
    }
    ControlFlow::FallsThrough
}

fn integration_panic_binding(statements: &[AstStatement]) -> Option<String> {
    statements
        .iter()
        .find_map(integration_panic_capture)
        .map(|(_, binding)| binding.to_owned())
}

fn integration_panic_capture(statement: &AstStatement) -> Option<(&str, &str)> {
    let AstStatement::Expression(AstExpression {
        kind: AstExpressionKind::Call { callee, arguments },
        ..
    }) = statement
    else {
        return None;
    };
    if callable_path(callee).as_deref() != Some("throws") {
        return None;
    }
    let [argument] = arguments.as_slice() else {
        return None;
    };
    let AstExpressionKind::Name(wrapper) = &argument.value.kind else {
        return None;
    };
    let function = wrapper.strip_suffix("_wrapper")?;
    let AstExpressionKind::Name(binding) = &argument.expected_error.as_ref()?.kind else {
        return None;
    };
    Some((function, binding))
}

fn integration_expectation(
    statement: &AstStatement,
    panic_binding: Option<&str>,
) -> Option<severian_hir::TestExpectation> {
    if let Some((function, binding)) = integration_panic_capture(statement) {
        return Some(severian_hir::TestExpectation::Panics {
            function: function.to_owned(),
            binding: binding.to_owned(),
        });
    }
    let AstStatement::Assert { condition, .. } = statement else {
        return None;
    };
    if let AstExpressionKind::Unary {
        operator: AstUnaryOperator::Not,
        operand,
    } = &condition.kind
    {
        let AstExpressionKind::Binary {
            operator: AstBinaryOperator::Contains,
            left,
            right,
        } = &operand.kind
        else {
            return None;
        };
        return Some(severian_hir::TestExpectation::Excludes {
            stream: test_stream(right)?,
            value: string_literal(left)?.to_owned(),
        });
    }
    let AstExpressionKind::Binary {
        operator,
        left,
        right,
    } = &condition.kind
    else {
        return None;
    };
    if *operator == AstBinaryOperator::Equal {
        if let AstExpressionKind::Member { object, name } = &left.kind {
            if name == "message" {
                if let (AstExpressionKind::Name(binding), Some(value)) =
                    (&object.kind, string_literal(right))
                {
                    if panic_binding == Some(binding) {
                        return Some(severian_hir::TestExpectation::PanicMessage {
                            binding: binding.clone(),
                            value: value.to_owned(),
                        });
                    }
                }
            }
        }
    }
    match operator {
        AstBinaryOperator::Contains => Some(severian_hir::TestExpectation::Contains {
            stream: test_stream(right)?,
            value: string_literal(left)?.to_owned(),
        }),
        AstBinaryOperator::Equal => {
            if let (Some(stream), Some(value)) = (test_stream(left), string_literal(right)) {
                Some(severian_hir::TestExpectation::Equals {
                    stream,
                    value: value.to_owned(),
                })
            } else {
                Some(severian_hir::TestExpectation::Equals {
                    stream: test_stream(right)?,
                    value: string_literal(left)?.to_owned(),
                })
            }
        }
        _ => None,
    }
}

fn test_stream(expression: &AstExpression) -> Option<severian_hir::TestStream> {
    match &expression.kind {
        AstExpressionKind::Name(name) if name == "stdout" => Some(severian_hir::TestStream::Stdout),
        AstExpressionKind::Name(name) if name == "stderr" => Some(severian_hir::TestStream::Stderr),
        _ => None,
    }
}

fn string_literal(expression: &AstExpression) -> Option<&str> {
    match &expression.kind {
        AstExpressionKind::Literal(AstLiteral::String(value)) => Some(value),
        _ => None,
    }
}

fn universal_literal(literal: &AstLiteral) -> LiteralValue {
    match literal {
        AstLiteral::Integer(value) => LiteralValue::Integer(value.clone()),
        AstLiteral::Float(value) => LiteralValue::Float(value.clone()),
        AstLiteral::Measured { .. } => {
            unreachable!("measured literals are resolved with their dimension")
        }
        AstLiteral::Boolean(value) => LiteralValue::Boolean(*value),
        AstLiteral::Character(value) => LiteralValue::Character(*value),
        AstLiteral::String(value) => LiteralValue::String(value.clone()),
        AstLiteral::Bytes(value) => LiteralValue::Bytes(value.clone()),
        AstLiteral::None => LiteralValue::None,
        AstLiteral::Unit => LiteralValue::Unit,
    }
}

fn explicit_drop_receiver(expression: &AstExpression) -> Option<&str> {
    let AstExpressionKind::Call { callee, arguments } = &expression.kind else {
        return None;
    };
    if !arguments.is_empty() {
        return None;
    }
    let AstExpressionKind::Member { object, name } = &callee.kind else {
        return None;
    };
    if name != "drop" {
        return None;
    }
    let AstExpressionKind::Name(receiver) = &object.kind else {
        return None;
    };
    Some(receiver)
}

fn expression_is_binding(expression: &Expression, binding: BindingId) -> bool {
    matches!(expression.kind, ExpressionKind::Binding(candidate) if candidate == binding)
}

fn expression_contains_binding(expression: &Expression, binding: BindingId) -> bool {
    match &expression.kind {
        ExpressionKind::Binding(candidate) => *candidate == binding,
        ExpressionKind::Literal(_) | ExpressionKind::Function(_) => false,
        ExpressionKind::Aggregate { fields, .. } => fields
            .iter()
            .any(|field| expression_contains_binding(field, binding)),
        ExpressionKind::Field { object, .. }
        | ExpressionKind::Async {
            expression: object, ..
        }
        | ExpressionKind::Await(object)
        | ExpressionKind::Throw(object)
        | ExpressionKind::Convert {
            operand: object, ..
        }
        | ExpressionKind::Borrow {
            operand: object, ..
        }
        | ExpressionKind::Move(object)
        | ExpressionKind::Unary {
            operand: object, ..
        } => expression_contains_binding(object, binding),
        ExpressionKind::Call { arguments, .. } => arguments
            .iter()
            .any(|argument| expression_contains_binding(argument, binding)),
        ExpressionKind::AsyncFieldUpdate {
            binding: candidate,
            value,
            ..
        } => *candidate == binding || expression_contains_binding(value, binding),
        ExpressionKind::Fallback {
            condition,
            value,
            fallback,
        } => {
            expression_contains_binding(condition, binding)
                || expression_contains_binding(value, binding)
                || expression_contains_binding(fallback, binding)
        }
        ExpressionKind::Binary { left, right, .. } => {
            expression_contains_binding(left, binding)
                || expression_contains_binding(right, binding)
        }
    }
}

fn align_layout(value: u64, alignment: u64) -> u64 {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value.saturating_add(alignment - remainder)
    }
}

fn measured_literal(
    magnitude: &str,
    suffix: &str,
    span: severian_source::Span,
) -> Result<(&'static str, LiteralValue), Diagnostic> {
    let magnitude = magnitude.parse::<f64>().map_err(|_| {
        Diagnostic::new(
            "E000203",
            format!("invalid magnitude in `{magnitude}{suffix}`"),
            Some(span),
        )
    })?;
    let canonical = match suffix {
        "b" => magnitude / 8.0,
        "B" | "s" | "Hz" | "C" | "V" | "A" | "W" => magnitude,
        "KB" | "kHz" => magnitude * 1_000.0,
        "MB" | "MHz" => magnitude * 1_000_000.0,
        "GB" | "GHz" => magnitude * 1_000_000_000.0,
        "TB" => magnitude * 1_000_000_000_000.0,
        "KiB" => magnitude * 1_024.0,
        "MiB" => magnitude * 1_048_576.0,
        "GiB" => magnitude * 1_073_741_824.0,
        "TiB" => magnitude * 1_099_511_627_776.0,
        "pct" => magnitude / 100.0,
        "ns" => magnitude / 1_000_000_000.0,
        "us" => magnitude / 1_000_000.0,
        "ms" | "mV" | "mA" => magnitude / 1_000.0,
        "min" => magnitude * 60.0,
        "hr" => magnitude * 3_600.0,
        "day" => magnitude * 86_400.0,
        "F" => (magnitude - 32.0) * 5.0 / 9.0,
        "K" => magnitude - 273.15,
        _ => {
            return Err(Diagnostic::new(
                "E000203",
                format!("unknown numeric unit suffix `{suffix}`"),
                Some(span),
            )
            .with_help(
                "use a declared unit suffix or separate the number and identifier with whitespace",
            ));
        }
    };
    let type_name = measured_type_name(suffix).expect("every normalized suffix has a dimension");
    if !canonical.is_finite() {
        return Err(Diagnostic::new(
            "E000203",
            "measured literal is outside the supported numeric range",
            Some(span),
        ));
    }
    let mut spelling = canonical.to_string();
    if !spelling.contains(['.', 'e', 'E']) {
        spelling.push_str(".0");
    }
    Ok((type_name, LiteralValue::Float(spelling)))
}

fn measured_type_name(suffix: &str) -> Option<&'static str> {
    Some(match suffix {
        "b" | "B" | "KB" | "MB" | "GB" | "TB" | "KiB" | "MiB" | "GiB" | "TiB" => "data_size",
        "pct" => "percentage",
        "ns" | "us" | "ms" | "s" | "min" | "hr" | "day" => "duration",
        "Hz" | "kHz" | "MHz" | "GHz" => "frequency",
        "C" | "F" | "K" => "temperature",
        "mV" | "V" => "voltage",
        "mA" | "A" => "current",
        "W" => "power",
        _ => return None,
    })
}

fn universal_unary(operator: AstUnaryOperator) -> UnaryOperator {
    match operator {
        AstUnaryOperator::Positive => UnaryOperator::Positive,
        AstUnaryOperator::Negative => UnaryOperator::Negative,
        AstUnaryOperator::Not => UnaryOperator::Not,
        AstUnaryOperator::AddressOf => {
            unreachable!("raw address-of is lowered before universal resolution")
        }
        AstUnaryOperator::Borrow | AstUnaryOperator::BorrowMut => {
            unreachable!("borrows are lowered before universal resolution")
        }
        AstUnaryOperator::Copy => unreachable!("copy is lowered before universal resolution"),
        AstUnaryOperator::Move => unreachable!("moves are lowered before universal resolution"),
    }
}

fn universal_binary(operator: AstBinaryOperator) -> BinaryOperator {
    match operator {
        AstBinaryOperator::Pipe => BinaryOperator::BitwiseOr,
        AstBinaryOperator::BitwiseAnd => BinaryOperator::BitwiseAnd,
        AstBinaryOperator::BitwiseXor => BinaryOperator::BitwiseXor,
        AstBinaryOperator::Add => BinaryOperator::Add,
        AstBinaryOperator::Subtract => BinaryOperator::Subtract,
        AstBinaryOperator::Multiply => BinaryOperator::Multiply,
        AstBinaryOperator::Divide => BinaryOperator::Divide,
        AstBinaryOperator::Remainder => BinaryOperator::Remainder,
        AstBinaryOperator::Power => BinaryOperator::Power,
        AstBinaryOperator::Equal => BinaryOperator::Equal,
        AstBinaryOperator::Identity => {
            unreachable!("identity is lowered before universal resolution")
        }
        AstBinaryOperator::NotEqual => BinaryOperator::NotEqual,
        AstBinaryOperator::Less => BinaryOperator::Less,
        AstBinaryOperator::LessEqual => BinaryOperator::LessEqual,
        AstBinaryOperator::Greater => BinaryOperator::Greater,
        AstBinaryOperator::GreaterEqual => BinaryOperator::GreaterEqual,
        AstBinaryOperator::Contains => BinaryOperator::Contains,
        AstBinaryOperator::And => BinaryOperator::And,
        AstBinaryOperator::Or => BinaryOperator::Or,
    }
}

fn semantic_error(message: String, span: severian_source::Span) -> Diagnostic {
    Diagnostic::new("E000202", message, Some(span))
}

fn primitive_method_operator(name: &str) -> Option<BinaryOperator> {
    Some(match name {
        "add" => BinaryOperator::Add,
        "subtract" => BinaryOperator::Subtract,
        "multiply" => BinaryOperator::Multiply,
        "divide" => BinaryOperator::Divide,
        "remainder" => BinaryOperator::Remainder,
        "power" => BinaryOperator::Power,
        "equal" => BinaryOperator::Equal,
        "not_equal" => BinaryOperator::NotEqual,
        "less_than" => BinaryOperator::Less,
        "less_equal" => BinaryOperator::LessEqual,
        "greater_than" => BinaryOperator::Greater,
        "greater_equal" => BinaryOperator::GreaterEqual,
        "contains" => BinaryOperator::Contains,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_source::SourceFile;

    fn analyze_source(source: &str) -> (Program, severian_universal::UniversalContext) {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source("test.sev", source);
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let hir = analyze(&ast, &context.types).unwrap();
        (hir, context)
    }

    #[test]
    fn plain_extensions_add_methods_but_cannot_replace_them() {
        analyze_source(
            "class Counter:\n    value: int\n    def get() -> int:\n        return value\n\nextend Counter:\n    def reset() -> Counter:\n        return Counter(0)\n\ndef main():\n    counter := Counter(10)\n    counter = counter.reset()\n    assert(counter.get() == 0)\n",
        );

        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "invalid-extension.sev",
            "class Counter:\n    value: int\n    def get() -> int:\n        return value\n\nextend Counter:\n    def get() -> int:\n        return 0\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert!(error.message.contains("cannot replace behavior `Counter.get`"));
    }

    #[test]
    fn namespaced_set_extensions_lower_full_operator_bodies() {
        let (program, _) = analyze_source(
            "@combinatorics\nextend set[T]:\n    operator +(other: set[T]) -> set[T]:\n        result := self\n        for value in other:\n            result.add(value)\n        return result\n    operator -(other: set[T]) -> set[T]:\n        result := set[T]()\n        for value in self:\n            if value not in other:\n                result.add(value)\n        return result\n\n@combinatorics(+, -)\ndef combine(left: set[int], right: set[int]) -> set[int]:\n    return (left + right) - {2}\n\ndef main():\n    result := combine({1, 2, 3}, {3, 4})\n    assert(result == {1, 3, 4})\n",
        );
        let module = &program.modules[0];
        assert!(module
            .functions
            .iter()
            .any(|function| function.name.starts_with("__sev_extension_combinatorics")));
        severian_mir::build(&program).unwrap();

        let context = severian_bootstrap::load().unwrap();
        let outside = SourceFile::virtual_source(
            "inactive-extension.sev",
            "@combinatorics\nextend set[T]:\n    operator +(other: set[T]) -> set[T]:\n        return self\n\ndef main():\n    result := {1, 2} + {2, 3}\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&outside).unwrap()).unwrap();
        assert!(analyze(&ast, &context.types).is_err());

        let overwrite = SourceFile::virtual_source(
            "native-extension-overwrite.sev",
            "@combinatorics\nextend set[T]:\n    def contains(value: T) -> bool:\n        return false\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&overwrite).unwrap()).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert!(error.message.contains("cannot replace behavior `set.contains`"));
    }

    #[test]
    fn canonical_extension_examples_reach_mir() {
        for (name, text) in [
            (
                "basic-extend.sev",
                include_str!("../../../../docs/examples/01-types/07-extend/01-basic-extend.sev"),
            ),
            (
                "namespace-extensions.sev",
                include_str!("../../../../docs/examples/01-types/07-extend/02-namespace-extensions.sev"),
            ),
        ] {
            let context = severian_bootstrap::load().unwrap();
            let source = SourceFile::virtual_source(name, text);
            let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
            let program = analyze_with_context(
                &ast,
                &context.types,
                AnalysisContext {
                    mode: AnalysisMode::Test,
                    module_name: name,
                },
            )
            .unwrap();
            severian_mir::build(&program).unwrap();
        }
    }

    #[test]
    fn source_parameter_effects_wrap_call_arguments() {
        let (program, _) = analyze_source(
            "def read(values: list[int]) -> usize:\n    return values.length()\n\ndef clear(values: list[int]):\n    values.clear()\n\ndef store(values: list[int]) -> list[int]:\n    return values\n\ndef main():\n    values := [1, 2, 3]\n    read(values)\n    clear(values)\n    stored := store(values)\n",
        );
        let module = &program.modules[0];
        let definition = |name: &str| {
            module
                .functions
                .iter()
                .find(|function| function.name == name)
                .unwrap()
                .definition
        };
        let main = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .unwrap();
        let expressions = main
            .body
            .as_ref()
            .unwrap()
            .statements
            .iter()
            .filter_map(|statement| match statement {
                Statement::Expression(expression) => Some(expression),
                Statement::Binding(binding) => module
                    .bindings
                    .iter()
                    .find(|candidate| candidate.id == *binding)
                    .map(|binding| &binding.value),
                _ => None,
            })
            .collect::<Vec<_>>();
        let argument = |target| {
            expressions.iter().find_map(|expression| {
                let ExpressionKind::Call { callee, arguments } = &expression.kind else {
                    return None;
                };
                matches!(callee, severian_hir::Callee::Direct { function, .. } if *function == target)
                    .then(|| &arguments[0])
            })
        };

        assert!(matches!(
            argument(definition("read")).map(|argument| &argument.kind),
            Some(ExpressionKind::Borrow {
                exclusive: false,
                ..
            })
        ));
        assert!(matches!(
            argument(definition("clear")).map(|argument| &argument.kind),
            Some(ExpressionKind::Borrow {
                exclusive: true,
                ..
            })
        ));
        assert!(matches!(
            argument(definition("store")).map(|argument| &argument.kind),
            Some(ExpressionKind::Move(_))
        ));
    }

    #[test]
    fn declaration_operator_controls_storage_mutability() {
        let (program, _) = analyze_source("constant = 1\nvariable := 2\nvariable = 3\n");
        let module = &program.modules[0];
        assert!(!module.bindings[0].mutable);
        assert!(module.bindings[1].mutable);
        assert!(module.bindings[2].mutable);
        assert_eq!(module.bindings[1].variable, module.bindings[2].variable);

        let mir = severian_mir::build(&program).unwrap();
        assert_eq!(mir.globals.len(), 2);
        assert!(!mir.globals[0].mutable);
        assert!(mir.globals[1].mutable);
        assert!(mir
            .globals
            .iter()
            .all(|global| global.span.end > global.span.start));
        for block in &mir.initializer.blocks {
            assert_eq!(block.statements.len(), block.statement_spans.len());
            assert!(block.statement_spans.iter().all(Option::is_some));
        }
    }

    #[test]
    fn immutable_binding_cannot_be_reassigned() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source("test.sev", "constant = 1\nconstant = 2\n");
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000203");
        assert!(error.message.contains("immutable binding `constant`"));
    }

    #[test]
    fn transition_aware_enums_accept_declared_edges_and_reject_missing_edges() {
        analyze_source(
            "enum Direction:\n    Left\n    Right\ndef unrestricted():\n    direction := Left\n    direction = Right\n",
        );
        analyze_source(
            "enum Status:\n    Connecting -> Received | Failed\n    Received\n    Failed\ndef valid():\n    state := Connecting\n    state = Received\n",
        );

        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "invalid-transition.sev",
            "enum Status:\n    Connecting -> Received | Failed\n    Received\n    Failed\ndef invalid():\n    state := Received\n    state = Connecting\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000213");
        assert!(error
            .message
            .contains("Status.Received -> Status.Connecting"));
    }

    #[test]
    fn enum_accepted_values_lower_to_canonical_variants() {
        let (program, _) = analyze_source(
            "enum Symbol:\n    Add {\"+\", \"+=\"}\n    Multiply {\"*\", \"*=\", 0}\ndef main():\n    add := Symbol(\"+=\")\n    multiply := Symbol(0)\n    assert(add == Symbol.Add)\n    assert(multiply == Symbol.Multiply)\n",
        );
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn enum_accepted_values_must_identify_one_variant() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "ambiguous-enum.sev",
            "enum Bad:\n    First {\"+\"}\n    Second {\"+\"}\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000213");
        assert!(error.message.contains("Bad.First"));
        assert!(error.message.contains("Bad.Second"));
    }

    #[test]
    fn select_lowers_to_a_bounded_receive_loop() {
        let (program, _) = analyze_source(
            "def main():\n    commands = channel[string]\n    received := 0\n    select with limit=1:\n        case command from commands:\n            received += 1\n        case error:\n            assert(false)\n",
        );
        let module = &program.modules[0];
        let main = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .unwrap();
        let body = main.body.as_ref().unwrap();
        let Statement::Sequence(select) = &body.statements[2] else {
            panic!("select lowers to an ordered counter and loop")
        };
        assert!(matches!(select.statements[0], Statement::Binding(_)));
        assert!(matches!(select.statements[1], Statement::While { .. }));
        for symbol in [
            "__sev_channel_claim",
            "__sev_channel_recv_ptr",
            "__sev_channel_is_closed",
            "__sev_channel_yield",
        ] {
            assert!(module
                .functions
                .iter()
                .any(|function| function.name == symbol));
        }
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn while_survives_hir_and_updates_keep_variable_identity() {
        let (program, _) = analyze_source(
            "def main():\n    while count < 3 with count := 0:\n        count += 1\n",
        );
        let module = &program.modules[0];
        let body = module.functions[0].body.as_ref().unwrap();
        let Statement::Sequence(sequence) = &body.statements[0] else {
            panic!("while with an initializer lowers to a sequence");
        };
        let Statement::Binding(initial) = sequence.statements[0] else {
            panic!("initializer remains an ordered binding");
        };
        let Statement::While {
            condition, body, ..
        } = &sequence.statements[1]
        else {
            panic!("runtime while remains in HIR");
        };
        assert!(matches!(
            &condition.kind,
            severian_hir::ExpressionKind::Binary { left, .. }
                if matches!(left.kind, severian_hir::ExpressionKind::Binding(id) if id == initial)
        ));
        let Statement::Binding(update) = body.statements[0] else {
            panic!("loop update remains a binding version");
        };
        let initial = module
            .bindings
            .iter()
            .find(|binding| binding.id == initial)
            .unwrap();
        let update = module
            .bindings
            .iter()
            .find(|binding| binding.id == update)
            .unwrap();
        assert_eq!(initial.variable, update.variable);
        assert_ne!(initial.id, update.id);

        let mir = severian_mir::build(&program).unwrap();
        let cfg = mir.functions[0].body.as_ref().unwrap();
        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.locals.iter().filter(|local| local.mutable).count(), 1);
        assert!(matches!(
            cfg.blocks[0].terminator,
            severian_mir::Terminator::Goto(severian_mir::BlockId(1), ref arguments)
                if arguments.is_empty()
        ));
        assert!(matches!(
            cfg.blocks[1].terminator,
            severian_mir::Terminator::Branch {
                then_block: severian_mir::BlockId(2),
                else_block: severian_mir::BlockId(3),
                ..
            }
        ));
        assert!(matches!(
            cfg.blocks[2].terminator,
            severian_mir::Terminator::Goto(severian_mir::BlockId(1), ref arguments)
                if arguments.is_empty()
        ));
    }

    #[test]
    fn function_contracts_lower_messages_to_assertions_and_errors_to_throws() {
        let (program, _) = analyze_source(
            "def bounded(value: int) -> int with { value >= 0, defer value <= 10 -> Error(\"too large\") }:\n    return value\n",
        );
        let body = program.modules[0].functions[0].body.as_ref().unwrap();
        assert!(matches!(body.statements[0], Statement::Assert { .. }));
        assert!(matches!(body.statements[1], Statement::Expression(_)));
        assert!(matches!(body.statements[2], Statement::Return(_)));
        let mir = severian_mir::build(&program).unwrap();
        let statements = mir.functions[0]
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .flat_map(|block| &block.statements);
        assert_eq!(
            statements
                .filter(|statement| matches!(statement, severian_mir::CfgStatement::Assert { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn fallible_else_binds_the_contract_error() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "fallible.sev",
            "def divide(value: f64, divisor: f64) -> f64 | Error with { divisor != 0.0 -> Error(\"zero\") }:\n    return value / divisor\ntest:\n    divide(1, 0) else error:\n        assert(error == Error)\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "fallible",
            },
        )
        .unwrap();
        let test = program.modules[0]
            .functions
            .iter()
            .find(|function| function.name.contains("test"))
            .unwrap();
        assert!(matches!(
            test.body.as_ref().unwrap().statements.as_slice(),
            [Statement::Sequence(Block { statements })]
                if matches!(statements.as_slice(), [Statement::Binding(_), Statement::Binding(_), Statement::If { .. }])
        ));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn defer_actions_run_lifo_on_return() {
        let (program, _) = analyze_source(
            "def first():\n    pass\ndef second():\n    pass\ndef finish() -> int:\n    defer first()\n    defer second()\n    return 7\n",
        );
        let module = &program.modules[0];
        let first = module.functions[0].definition;
        let second = module.functions[1].definition;
        let body = module.functions[2].body.as_ref().unwrap();
        let called = body
            .statements
            .iter()
            .filter_map(|statement| {
                let Statement::Expression(Expression {
                    kind:
                        ExpressionKind::Call {
                            callee: severian_hir::Callee::Direct { function, .. },
                            ..
                        },
                    ..
                }) = statement
                else {
                    return None;
                };
                Some(*function)
            })
            .collect::<Vec<_>>();
        assert_eq!(called, [second, first]);
        assert!(matches!(body.statements.last(), Some(Statement::Return(_))));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn hook_context_must_name_a_function_parameter() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "invalid-hook.sev",
            "def trace(value: string) with context:\n    with context:\n        print(value)\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000218");
        assert!(error.message.contains("is not a parameter"));
    }

    #[test]
    fn loop_control_is_rejected_outside_a_loop() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source("test.sev", "def main():\n    break\n");
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000211");
        assert!(error.message.contains("inside a loop"));
    }

    #[test]
    fn break_and_continue_lower_to_cfg_edges() {
        let (program, _) = analyze_source(
            "def main():\n    while count < 5 with count := 0:\n        count += 1\n        if count == 2:\n            continue\n        if count == 4:\n            break\n",
        );
        let mir = severian_mir::build(&program).unwrap();
        let cfg = mir.functions[0].body.as_ref().unwrap();
        let gotos = cfg
            .blocks
            .iter()
            .filter_map(|block| match block.terminator {
                severian_mir::Terminator::Goto(target, _) => Some(target),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(gotos.iter().filter(|target| target.0 == 1).count() >= 2);
        assert!(gotos.iter().any(|target| target.0 == 3));
    }

    #[test]
    fn annotation_and_both_literal_orders_share_i32() {
        let (program, context) = analyze_source("x: i32 = 10\na = x + 1\nb = 1 + x\n");
        let i32 = context.types.resolve_name("i32").unwrap();
        assert!(program.modules[0]
            .bindings
            .iter()
            .all(|binding| binding.type_id == i32));
    }

    #[test]
    fn unconstrained_literals_default_only_after_operator_matching() {
        let (program, context) = analyze_source("a = 1 + 2\n");
        let int = context.types.resolve_name("int").unwrap();
        assert_eq!(program.modules[0].bindings[0].type_id, int);
    }

    #[test]
    fn single_argument_primitive_prints_use_display_string() {
        let (program, _) = analyze_source(
            "def main():\n    count: i32 = 10\n    large: i64 = 1_000_000\n    ratio: f64 = 0.5\n    print(count)\n    print(large)\n    print(ratio)\n",
        );
        let names = program.modules[0]
            .functions
            .iter()
            .map(|function| function.name.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("__sev_string_from_int"));
        assert!(names.contains("__sev_string_from_float"));
        assert!(names.contains("__sev_print_string"));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn question_equal_survives_semantic_analysis() {
        let (program, _) = analyze_source("result ?= 1\nordinary = 2\n");
        assert!(program.modules[0].bindings[0].preserve_error);
        assert!(!program.modules[0].bindings[1].preserve_error);
    }

    #[test]
    fn expected_binary_and_default_unary_constraints_are_ranked() {
        let (program, context) = analyze_source("a: i32 = 1 + 2\nb = -1\n");
        assert_eq!(
            program.modules[0].bindings[0].type_id,
            context.types.resolve_name("i32").unwrap()
        );
        assert_eq!(
            program.modules[0].bindings[1].type_id,
            context.types.resolve_name("int").unwrap()
        );
    }

    #[test]
    fn power_expressions_lower_to_numeric_runtime_calls() {
        let source = "def powers() -> float:\n    integer = 2 ** 2\n    root = integer ** .5\n    floating = 4.0 ** 2\n    return root + floating\n";
        let (program, _) = analyze_source(source);
        let names = program.modules[0]
            .functions
            .iter()
            .map(|function| function.name.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("__sev_pow_i64_i64"));
        assert!(names.contains("__sev_pow_f64_f64"));
        assert!(names.contains("__sev_pow_f64_i64"));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn generic_class_instances_keep_distinct_layouts_and_field_updates() {
        let source = "class Box[T]:\n    value: T\n    def addition(addition: T):\n        value += addition\ndef main():\n    ints := Box[int](10)\n    floats := Box[f64](2.5)\n    ints.addition(20)\n    floats.addition(4.5)\n    observed_int := ints.value\n    observed_float := floats.value\n";
        let (program, _) = analyze_source(source);
        let module = &program.modules[0];
        assert_eq!(module.classes.len(), 2);
        assert_ne!(module.classes[0].id, module.classes[1].id);
        let main = module
            .functions
            .iter()
            .find(|function| function.name == "main")
            .unwrap();
        assert_eq!(
            main.body
                .as_ref()
                .unwrap()
                .statements
                .iter()
                .filter(|statement| matches!(statement, Statement::FieldUpdate { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn no_value_class_methods_lower_expression_bodies() {
        let source = "trait Drawable:\n    def draw() -> None\nclass Point: Drawable\n    x: float\n    y: float\n    def draw() -> None:\n        print(\"point\", x, y)\ndef main():\n    point = Point(3.0, 4.0)\n    point.draw()\n";
        let (program, _) = analyze_source(source);
        let main = program.modules[0]
            .functions
            .iter()
            .find(|function| function.name == "main")
            .unwrap();
        assert!(matches!(
            main.body.as_ref().unwrap().statements.as_slice(),
            [Statement::Binding(_), Statement::Sequence(Block { statements })]
                if matches!(statements.as_slice(), [Statement::Expression(_)])
        ));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn trait_method_decorator_registers_a_constrained_namespace_call() {
        let source = "trait File:\n    @file\n    def read(path: string) -> string with { (path) -> bool }\nclass LuaFile: File\n    def read(path: string) -> string with { \".lua\" in path }:\n        return \"lua\"\nclass JsonFile: File\n    def read(path: string) -> string with { \".json\" in path }:\n        return \"json\"\ndef selected() -> string:\n    return file.read(\"test.json\")\n";
        let (program, _) = analyze_source(source);
        let selected = program.modules[0]
            .functions
            .iter()
            .find(|function| function.name == "selected")
            .unwrap();
        assert!(matches!(
            selected.body.as_ref().unwrap().statements.as_slice(),
            [Statement::Return(Some(Expression {
                kind: ExpressionKind::Fallback { .. },
                ..
            }))]
        ));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn function_decorator_selects_trait_owned_pipe_operator() {
        let source = "trait StringOperator:\n    @strings\n    operator |(left: string, right: string) -> string\nclass Strings:\n    trait StringOperator\n    operator |(left: string, right: string) -> string:\n        return \"selected:\" + left + right\n@strings(|)\ndef combine(left: string, right: string) -> string:\n    return left | right\n";
        let (program, _) = analyze_source(source);
        let combine = program.modules[0]
            .functions
            .iter()
            .find(|function| function.name == "combine")
            .unwrap();
        assert!(matches!(
            combine.body.as_ref().unwrap().statements.as_slice(),
            [Statement::Return(Some(Expression {
                kind: ExpressionKind::Fallback { .. },
                ..
            }))]
        ));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn class_parameters_cannot_shadow_fields() {
        let source = SourceFile::virtual_source(
            "shadow.sev",
            "class Box[T]:\n    value: T\n    def invalid(value: T):\n        pass\n",
        );
        let universal = severian_bootstrap::load().unwrap();
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let error = analyze(&ast, &universal.types).unwrap_err();
        assert_eq!(error.code, "E000220");
        assert!(error.message.contains("shadows a field"));
    }

    #[test]
    fn explicit_drop_consumes_a_resource() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "drop.sev",
            "class Resource:\n    name: string\n    def Resource(name_param: string):\n        name := name_param\n    def drop():\n        \"closed\"\ndef invalid():\n    resource := Resource(\"temporary\")\n    drop resource\n    resource.name\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000201");
        assert!(error.message.contains("resource"));
    }

    #[test]
    fn raw_allocation_requires_an_unsafe_scope() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "allocation.sev",
            "def invalid():\n    memory := allocate[int](4)\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000219");
        assert!(error.message.contains("raw allocation"));
    }

    #[test]
    fn untyped_case_binding_is_a_catch_all_not_a_magic_result_variant() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "match.sev",
            "def invalid(result: int) -> int:\n    match result:\n        case value:\n            return value\n        case error: int:\n            return error\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000214");
        assert!(error.message.contains("unreachable"));
    }

    #[test]
    fn build_excludes_tests_and_test_mode_uses_descriptive_internal_names() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "checks.sev",
            "def helper():\n    pass\n\ntest with profile and compiler \"frontend diagnostics\":\n    reject:\n        missing()\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let build = analyze(&ast, &context.types).unwrap();
        assert!(build.modules[0].tests.is_empty());
        assert_eq!(build.modules[0].functions.len(), 1);

        let tests = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "package_checks",
            },
        )
        .unwrap();
        assert_eq!(tests.modules[0].tests.len(), 1);
        assert_eq!(
            tests.modules[0].functions[1].name,
            "__sev_package_checks_test_with_profile-compiler_frontend_diagnostics_0"
        );
    }

    #[test]
    fn profile_test_contracts_become_duration_expectations() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "profile-contract.sev",
            "test with profile with\n{\n    0.1s < time -> Error(\"too fast\")\n}:\n    assert(true)\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "profile_contract",
            },
        )
        .unwrap();
        assert_eq!(
            program.modules[0].tests[0].expectations,
            [severian_hir::TestExpectation::ProfileDuration {
                comparison: severian_hir::DurationComparison::Greater,
                threshold_nanos: 100_000_000,
                message: "too fast".to_owned(),
            }]
        );
    }

    #[test]
    fn fallible_async_calls_propagate_errors_when_awaited() {
        let (program, _) = analyze_source(
            "def fetch() -> bool | Error:\n    return true\n\ndef run() -> bool:\n    task = async fetch() with self\n    return await task\n",
        );
        let module = &program.modules[0];
        let task = module
            .bindings
            .iter()
            .find(|binding| matches!(binding.value.kind, ExpressionKind::Async { .. }))
            .expect("task binding");
        let ExpressionKind::Async { expression, .. } = &task.value.kind else {
            unreachable!()
        };
        assert!(matches!(expression.kind, ExpressionKind::Call { .. }));
        let run = module
            .functions
            .iter()
            .find(|function| function.name == "run")
            .unwrap();
        assert!(matches!(
            run.body.as_ref().unwrap().statements.last(),
            Some(Statement::Return(Some(Expression {
                kind: ExpressionKind::Fallback { .. },
                ..
            })))
        ));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn async_unit_class_updates_lower_as_spawned_mutations() {
        let (program, _) = analyze_source(
            "class Counter:\n    value: int\n    def increment():\n        value += 1\ndef main():\n    counter := Counter(0)\n    task = async counter.increment() with self and lock\n    await task\n",
        );
        let module = &program.modules[0];
        module
            .bindings
            .iter()
            .find(|binding| {
                matches!(
                    binding.value.kind,
                    ExpressionKind::AsyncFieldUpdate { locked: true, .. }
                )
            })
            .expect("async method produces a task binding");
        let mir = severian_mir::build(&program).unwrap();
        let main = mir
            .functions
            .iter()
            .find(|function| function.name == "main")
            .unwrap();
        assert!(main.body.as_ref().unwrap().blocks.iter().any(|block| {
            matches!(
                block.terminator,
                severian_mir::Terminator::SpawnFieldUpdate { locked: true, .. }
            )
        }));
    }

    #[test]
    fn compiler_tests_allow_shared_setup_but_reject_named_diagnostics() {
        let context = severian_bootstrap::load().unwrap();
        let setup = SourceFile::virtual_source(
            "compiler-test.sev",
            "test with compiler:\n    value := 1\n    reject:\n        missing()\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&setup).unwrap()).unwrap();
        analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "package_compiler_test",
            },
        )
        .unwrap();

        let named = SourceFile::virtual_source(
            "compiler-test.sev",
            "test with compiler:\n    reject error:\n        missing()\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&named).unwrap()).unwrap();
        let error = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "package_compiler_test",
            },
        )
        .unwrap_err();
        assert_eq!(error.code, "E000217");
    }

    #[test]
    fn captured_stream_assertions_make_ordinary_tests_integration_tests() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "implicit-integration.sev",
            "test:\n    assert(\"captured\" in stdout)\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "implicit_integration",
            },
        )
        .unwrap();
        let test = &program.modules[0].tests[0];
        assert_eq!(test.modes, [severian_hir::TestMode::Integration]);
        assert_eq!(
            test.expectations,
            [severian_hir::TestExpectation::Contains {
                stream: severian_hir::TestStream::Stdout,
                value: "captured".to_owned(),
            }]
        );
    }

    #[test]
    fn function_scopes_can_read_and_shadow_globals() {
        let (program, _) = analyze_source(
            "value := 1\ndef use_global():\n    observed := value\ndef main():\n    value := 2\n    observed := value\n",
        );
        let module = &program.modules[0];
        let global = module.bindings[0].id;
        let first_body_binding = module.bindings[1].id;
        let shadow = module.bindings[2].id;
        let shadow_use = &module.bindings[3].value;
        assert_ne!(global, first_body_binding);
        assert_ne!(global, shadow);
        assert_eq!(shadow_use.kind, ExpressionKind::Binding(shadow));
    }

    #[test]
    fn function_parameters_can_be_reassigned() {
        let (program, _) = analyze_source(
            "def transfer(source: i64, destination: i64, amount: i64) -> (i64, i64) with\n{\n    defer source + destination == original_total,\n}:\n    original_total = source + destination\n    source -= amount\n    destination += amount\n    return (source, destination)\n",
        );
        let module = &program.modules[0];
        let function = &module.functions[0];
        for parameter in &function.parameters[..2] {
            let variable = severian_hir::VariableId(parameter.binding.0);
            let update = module
                .bindings
                .iter()
                .find(|binding| binding.variable == variable)
                .expect("parameter reassignment produces a binding update");
            assert!(update.mutable);
        }

        let mir = severian_mir::build(&program).unwrap();
        let body = mir.functions[0].body.as_ref().unwrap();
        assert!(body
            .locals
            .iter()
            .filter(|local| local.argument)
            .all(|local| local.mutable));
    }

    #[test]
    fn return_analysis_is_control_flow_based_not_last_statement_based() {
        let (program, _) =
            analyze_source("def answer() -> int:\n    return 42\n    unreachable := 0\n");
        assert_eq!(program.modules[0].functions.len(), 1);
    }

    #[test]
    fn overload_resolution_prefers_exact_parameters_over_widening() {
        let (program, context) = analyze_source(
            "def choose(value: i32) -> i32:\n    return value\ndef choose(value: i64) -> i64:\n    return value\nsource: i32 = 1\nselected = choose(source)\n",
        );
        let selected = program.modules[0].bindings.last().unwrap();
        assert_eq!(selected.type_id, context.types.resolve_name("i32").unwrap());
    }

    #[test]
    fn numeric_promotions_and_explicit_casts_are_preserved_in_hir() {
        let (program, context) = analyze_source(
            "count = 10\nratio = 0.5\nmixed = count + ratio\nnarrowed = int(ratio)\nwidened = float(count)\n",
        );
        let bindings = &program.modules[0].bindings;
        let float = context.types.resolve_name("float").unwrap();
        let int = context.types.resolve_name("int").unwrap();

        assert_eq!(bindings[2].type_id, float);
        let ExpressionKind::Binary { left, .. } = &bindings[2].value.kind else {
            panic!("mixed arithmetic must remain a binary expression");
        };
        assert!(matches!(
            left.kind,
            ExpressionKind::Convert {
                conversion: severian_hir::Conversion {
                    kind: severian_hir::ConversionKind::NumericWidening,
                    ..
                },
                ..
            }
        ));

        assert_eq!(bindings[3].type_id, int);
        assert_eq!(bindings[4].type_id, float);
        for binding in [&bindings[3], &bindings[4]] {
            assert!(matches!(
                binding.value.kind,
                ExpressionKind::Convert {
                    conversion: severian_hir::Conversion {
                        kind: severian_hir::ConversionKind::NumericCast,
                        ..
                    },
                    ..
                }
            ));
        }
    }

    #[test]
    fn ordinary_union_parameters_lower_to_tagged_values_and_dispatch_conversions() {
        let source = "def to_float(value: string | int | float) -> float:\n    return float(value)\ndef selected() -> float:\n    return to_float(\"4.5\") + to_float(4) + to_float(4.5)\n";
        let (program, context) = analyze_source(source);
        let module = &program.modules[0];
        let to_float = module
            .functions
            .iter()
            .find(|function| function.name == "to_float")
            .unwrap();
        let union = to_float.parameters[0].contract.ty;
        assert!(module.classes.iter().any(|class| {
            class.id == union && class.fields.len() == 4 && class.fields[0].name == "__tag"
        }));
        assert!(to_float
            .body
            .as_ref()
            .unwrap()
            .statements
            .iter()
            .any(|statement| {
                matches!(
                    statement,
                    Statement::Return(Some(Expression {
                    kind: ExpressionKind::Fallback { .. },
                    ..
                }))
            )
        }));
        let float = context.types.resolve_name("float").unwrap();
        assert_eq!(to_float.result.ty, float);
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn callable_parameters_specialize_functions_and_lambdas_with_snapshot_captures() {
        let source = "def add(a: int, b: int) -> int:\n    return a + b\ndef apply(op: (int, int) -> int, left: int, right: int) -> int:\n    return op(left, right)\ntest:\n    assert(apply(add, 20, 22) == 42)\n    offset := 3\n    operation = lambda value, unused: value + offset + unused\n    offset = 10\n    assert(apply(operation, 4, 0) == 7)\n";
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source("lambda.sev", source);
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "lambda",
            },
        )
        .unwrap();
        let module = &program.modules[0];
        let apply = module
            .functions
            .iter()
            .find(|function| function.name == "apply")
            .unwrap();
        assert!(
            apply.body.is_none(),
            "higher-order body should be specialized"
        );
        assert!(module
            .classes
            .iter()
            .any(|class| class.name.starts_with("lambda#") && class.fields.len() == 1));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn concrete_overload_ranks_ahead_of_union_injection() {
        let source = "def choose(value: int) -> int:\n    return 1\ndef choose(value: int | string) -> int:\n    return 2\ndef selected() -> int:\n    return choose(4)\n";
        let (program, _) = analyze_source(source);
        let module = &program.modules[0];
        let concrete = module
            .functions
            .iter()
            .find(|function| function.name == "choose")
            .unwrap()
            .definition;
        let selected = module
            .functions
            .iter()
            .find(|function| function.name == "selected")
            .unwrap();
        assert!(matches!(
            selected.body.as_ref().unwrap().statements.as_slice(),
            [Statement::Return(Some(Expression {
                kind: ExpressionKind::Call {
                    callee: severian_hir::Callee::Direct { function, .. },
                    ..
                },
                ..
            }))] if *function == concrete
        ));
    }

    #[test]
    fn crossed_widening_overloads_remain_ambiguous() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "ambiguous.sev",
            "def choose(left: i32, right: i128) -> i32:\n    return left\ndef choose(left: i64, right: i64) -> i64:\n    return left\nleft: i32 = 1\nright: i32 = 2\nselected = choose(left, right)\n",
        );
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze(&ast, &context.types).unwrap_err();
        assert_eq!(error.code, "E000206");
    }

    #[test]
    fn mixed_integer_comparisons_do_not_coerce_distinct_unit_dimensions() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source("units.sev", "test:\n    assert(10ms < 20MB)\n");
        let tokens = severian_lexer::scan(&source).unwrap();
        let ast = severian_parser::parse(&tokens).unwrap();
        let error = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "units",
            },
        )
        .unwrap_err();
        assert_eq!(error.code, "E000202");
    }

    #[test]
    fn conversion_categories_order_exact_before_widening_before_general() {
        let context = severian_bootstrap::load().unwrap();
        let resolve = |name| context.types.resolve_name(name).unwrap();
        let exact = conversion_rank(&context.types, resolve("i32"), resolve("i32")).unwrap();
        let widening = conversion_rank(&context.types, resolve("i32"), resolve("i64")).unwrap();
        let general = conversion_rank(&context.types, resolve("bf16"), resolve("f32")).unwrap();
        assert!(exact < widening);
        assert!(widening < general);
    }

    #[test]
    fn sorted_lists_lower_to_representation_specific_runtime_calls() {
        let (program, _) = analyze_source(
            "strings = [\"beta\", \"alpha\"].sorted()\nnumbers = [3, 1, 2].sorted()\n",
        );
        let symbols = program.modules[0]
            .functions
            .iter()
            .filter_map(|function| match &function.call_type {
                severian_hir::CallType::External(call) => Some(call.symbol.0.as_str()),
                severian_hir::CallType::Severian => None,
            })
            .collect::<Vec<_>>();
        assert!(symbols.contains(&"__sev_list_sorted_ptr"));
        assert!(symbols.contains(&"__sev_list_sorted_i64"));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn indexing_composite_lists_uses_runtime_symbols_for_each_representation() {
        let (program, _) = analyze_source(
            "nested = [[1, 2], [3, 4]]\nrow = nested[0]\npairs = [(1, 2), (3, 4)]\npair = pairs[0]\n",
        );
        let symbols = program.modules[0]
            .functions
            .iter()
            .filter_map(|function| match &function.call_type {
                severian_hir::CallType::External(call) => Some(call.symbol.0.as_str()),
                severian_hir::CallType::Severian => None,
            })
            .collect::<Vec<_>>();
        assert!(symbols.contains(&"__sev_list_index_list"));
        assert!(symbols.contains(&"__sev_list_index_pair_i64"));
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn testing_predicates_scopes_and_generated_bindings_lower_to_mir() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "testing.sev",
            "test with property:\n    value: int {-10 <= value <= 10}\n    when(value >= 0):\n        expect(approximate(float(value), 0.0, atol=10.0))\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "testing",
            },
        )
        .unwrap();
        assert_eq!(program.modules[0].tests[0].modes, [severian_hir::TestMode::Property]);
        severian_mir::build(&program).unwrap();
    }

    #[test]
    fn dynamic_field_operations_lower_with_constraints_and_catchable_failures() {
        let (program, _) = analyze_source(
            "class Point:\n    x: int {0 <= x <= 100}\n    y: int {0 <= y <= 100}\n\ndef main():\n    point := Point(3, 4)\n    axis := runtime_string(\"x\")\n    point.set(axis, 10)\n    point.set({\"x\": 20, \"y\": 30})\n    observed := point.get(axis)\n    throws(point.set(\"x\", runtime_int(200)) -> ConstraintError)\n",
        );
        severian_mir::build(&program).unwrap();
        let body = program.modules[0].functions[0].body.as_ref().unwrap();
        assert!(body.statements.iter().any(|statement| {
            matches!(statement, Statement::If { .. } | Statement::Sequence(_))
        }));
        assert!(body
            .statements
            .iter()
            .any(|statement| matches!(statement, Statement::ExpectThrow { .. })));
    }

    #[test]
    fn builders_ordered_constraints_and_mocks_lower_to_executable_control_flow() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "builders.sev",
            "def foo(value: int) -> int:\n    return value\ndef calculate(value: int) -> int:\n    return foo(value) * 2\nclass User:\n    age: int {\n        age < 0 -> Error(\"negative\"),\n        age > 130 -> Error(\"old\"),\n    }\ntest:\n    mock(\n        foo(0) -> 10\n        else throw Error(\"unexpected\")\n    )\n    user := User().set(age, 36)\n    expect(calculate(0) == 20)\n    throws(foo(1) -> Error)\n    throws(foo(2) -> Error(\"unexpected\"))\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "builders",
            },
        )
        .unwrap();
        severian_mir::build(&program).unwrap();
        let test_id = program.modules[0].tests[0].function;
        let test = program.modules[0]
            .functions
            .iter()
            .find(|function| function.id == test_id)
            .unwrap();
        assert!(test
            .body
            .as_ref()
            .unwrap()
            .statements
            .iter()
            .any(|statement| { matches!(statement, Statement::ExpectThrow { .. }) }));
        assert!(program.modules[0]
            .functions
            .iter()
            .any(|function| function.name == "__sev_expect"));
    }

    #[test]
    fn try_catch_propagates_one_error_value_across_fallible_calls() {
        let (program, _) = analyze_source(
            "def fail() -> Error:\n    throw Error(\"origin\")\n\ndef inner() -> int | Error:\n    fail()\n    return 1\n\ndef outer() -> int | Error:\n    value = inner()\n    return value + 1\n\ndef main():\n    try:\n        outer()\n    catch error:\n        message = error.message\n        stack = error.call_stack\n",
        );
        let mir = severian_mir::build(&program).unwrap();
        assert!(mir
            .functions
            .iter()
            .all(|function| function
                .body
                .as_ref()
                .is_none_or(|body| body.blocks.iter().all(|block| {
                !matches!(block.terminator, severian_mir::Terminator::Throw(_))
            }))));
    }

    #[test]
    fn integration_panic_wrappers_become_isolated_runner_expectations() {
        let context = severian_bootstrap::load().unwrap();
        let source = SourceFile::virtual_source(
            "panic.sev",
            "def crash():\n    return\n\ntest with integ:\n    throws(crash_wrapper -> error)\n    assert(error.message == \"boom\")\n",
        );
        let ast = severian_parser::parse(&severian_lexer::scan(&source).unwrap()).unwrap();
        let program = analyze_with_context(
            &ast,
            &context.types,
            AnalysisContext {
                mode: AnalysisMode::Test,
                module_name: "panic",
            },
        )
        .unwrap();
        assert_eq!(
            program.modules[0].tests[0].expectations,
            vec![
                severian_hir::TestExpectation::Panics {
                    function: "crash".into(),
                    binding: "error".into(),
                },
                severian_hir::TestExpectation::PanicMessage {
                    binding: "error".into(),
                    value: "boom".into(),
                },
            ]
        );
    }
}
