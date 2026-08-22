#![forbid(unsafe_code)]

mod package;
mod queries;

pub use package::{
    analyze_package, analyze_package_with_context, DefKind, Definition, ExportMap, FunctionDecl,
    ModuleScope, PackageAnalysisContext, ProgramIndex, Resolution, Scope, TypedProgram, Visibility,
};
pub use severian_universal::{DeclarationId, DefId};
pub use queries::{QueryError, ScopeId, SemanticQueries};

use severian_ast::{
    BinaryOperator as AstBinaryOperator, Expression as AstExpression,
    ExpressionKind as AstExpressionKind, Literal as AstLiteral, Statement as AstStatement,
    TypeAnnotation, UnaryOperator as AstUnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_hir::{
    Binding, BindingId, Block, BoundaryType, CallType, Expression, ExpressionKind,
    FunctionDeclaration, FunctionId, FunctionParameter, HirId, Module, Program, Statement, TypeId,
};
use severian_universal::{
    BinaryOperator, LiteralValue, TypeConstraint, TypeContext, UnaryOperator,
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
    analyze_with_package_functions(ast, types, context, &[], &[], &[])
}

#[derive(Debug, Clone)]
pub(crate) struct PackageFunction {
    pub lookup: String,
    pub id: FunctionId,
    pub definition: DefId,
    pub type_parameters: Vec<severian_universal::GenericParamId>,
    pub parameters: Vec<TypeId>,
    pub result: TypeId,
}

pub(crate) fn analyze_with_package_functions(
    ast: &severian_ast::Module,
    types: &TypeContext,
    context: AnalysisContext<'_>,
    visible_functions: &[PackageFunction],
    own_function_ids: &[FunctionId],
    test_function_ids: &[FunctionId],
) -> Result<Program, Diagnostic> {
    let mut analyzer = Analyzer {
        types,
        names: BTreeMap::new(),
        declarations: BTreeSet::new(),
        functions: BTreeMap::new(),
        function_definitions: BTreeMap::new(),
        signatures: BTreeMap::new(),
        next_hir: 0,
        next_binding: 0,
    };
    for function in visible_functions {
        analyzer
            .functions
            .entry(function.lookup.clone())
            .or_default()
            .push(function.id);
        analyzer.signatures.insert(
            function.id,
            FunctionSignature {
                parameters: function.parameters.clone(),
                result: function.result,
            },
        );
        analyzer
            .function_definitions
            .insert(function.id, function.definition);
    }
    let mut module = Module::default();

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
            .unwrap_or_else(|| FunctionId(module.functions.len() as u128));
        let package_function = visible_functions
            .iter()
            .find(|function| function.id == id);
        let definition = package_function.map_or_else(
            || synthetic_definition(context.module_name, ast_function),
            |function| function.definition,
        );
        let mut parameters = Vec::new();
        let mut parameter_types = Vec::new();
        for parameter in &ast_function.parameters {
            let type_id = resolve_type_annotation(types, &parameter.annotation)?;
            let binding = analyzer.new_binding_id();
            parameter_types.push(type_id);
            parameters.push(FunctionParameter {
                binding,
                name: parameter.name.clone(),
                contract: universal_boundary(type_id),
            });
        }
        let result = resolve_type_annotation(types, &ast_function.result)?;
        let compile_route = types
            .compile_route(result)
            .map_err(|error| semantic_error(error.to_string(), ast_function.result.span))?;
        if own_function_ids.is_empty() {
            analyzer
                .functions
                .entry(ast_function.name.clone())
                .or_default()
                .push(id);
            analyzer.signatures.insert(
                id,
                FunctionSignature {
                    parameters: parameter_types.clone(),
                    result,
                },
            );
            analyzer.function_definitions.insert(id, definition);
        }
        module.functions.push(FunctionDeclaration {
            id,
            definition,
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
                .unwrap_or_else(|| FunctionId(module.functions.len() as u128));
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
            if modes.is_empty()
                && test
                    .body
                    .iter()
                    .any(|statement| integration_expectation(statement).is_some())
            {
                modes.push(severian_hir::TestMode::Integration);
            }
            module.tests.push(severian_hir::TestDeclaration {
                name: test
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("test {}", index + 1)),
                modes,
                function: id,
                expectations: Vec::new(),
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
        analyzer.names = globals.clone();
        analyzer.declarations.clear();
        for parameter in &function.parameters {
            let type_id = parameter.contract.ty;
            if !analyzer.declarations.insert(parameter.name.clone()) {
                return Err(Diagnostic::new(
                    "E000203",
                    format!("parameter `{}` is declared more than once", parameter.name),
                    Some(ast_function.span),
                ));
            }
            analyzer
                .names
                .insert(parameter.name.clone(), (parameter.binding, type_id));
        }
        let mut body = Block::default();
        let result_type = function.result.ty;
        for statement in ast_body {
            body.statements.push(analyzer.statement(
                statement,
                &mut module.bindings,
                result_type,
            )?);
        }
        if result_type != types.resolve_name("unit").expect("bootstrap defines unit")
            && block_flow(ast_body) == ControlFlow::FallsThrough
        {
            return Err(Diagnostic::new(
                "E000209",
                "not every path in this function returns its declared result",
                Some(ast_function.span),
            ));
        }
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
            analyzer.declarations.clear();
            let unit = types.resolve_name("unit").expect("bootstrap defines unit");
            let mut body = Block::default();
            if module.tests[offset]
                .modes
                .contains(&severian_hir::TestMode::Compiler)
            {
                if !test.body.is_empty() {
                    return Err(Diagnostic::new(
                        "E000217",
                        "compiler tests currently allow only `accept:` and `reject:` cases; diagnostic assertions are not implemented",
                        Some(test.span),
                    ));
                }
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
            for statement in &test.body {
                if module.tests[offset]
                    .modes
                    .contains(&severian_hir::TestMode::Integration)
                {
                    if let Some(expectation) = integration_expectation(statement) {
                        module.tests[offset].expectations.push(expectation);
                        continue;
                    }
                }
                body.statements
                    .push(analyzer.statement(statement, &mut module.bindings, unit)?);
            }
            module.functions[source_function_count + offset].body = Some(body);
        }
    }
    Ok(Program {
        modules: vec![module],
    })
}

struct Analyzer<'a> {
    types: &'a TypeContext,
    names: BTreeMap<String, (BindingId, TypeId)>,
    /// Names declared in the current lexical scope. `names` also contains
    /// readable parent bindings, which may be shadowed by this set.
    declarations: BTreeSet<String>,
    next_hir: u32,
    next_binding: u32,
    functions: BTreeMap<String, Vec<FunctionId>>,
    function_definitions: BTreeMap<FunctionId, DefId>,
    signatures: BTreeMap<FunctionId, FunctionSignature>,
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    parameters: Vec<TypeId>,
    result: TypeId,
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
        let update_type = if ast_binding.update {
            Some(
                self.names
                    .get(&ast_binding.name)
                    .map(|(_, type_id)| *type_id)
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
        if !ast_binding.update && !self.declarations.insert(ast_binding.name.clone()) {
            return Err(Diagnostic::new(
                "E000203",
                format!("binding `{}` is already defined", ast_binding.name),
                Some(ast_binding.span),
            ));
        }
        let expected = ast_binding
            .annotation
            .as_ref()
            .map(|annotation| resolve_type_annotation(self.types, annotation))
            .transpose()?
            .or(update_type);
        let value = self.expression(&ast_binding.value, expected)?;
        let type_id = expected.unwrap_or(value.type_id);
        if !self.types.assignable(value.type_id, type_id) {
            return Err(Diagnostic::new(
                "E000205",
                "binding value is not assignable to its declared type",
                Some(ast_binding.value.span),
            ));
        }
        let id = self.new_binding_id();
        self.names.insert(ast_binding.name.clone(), (id, type_id));
        bindings.push(Binding {
            id,
            type_id,
            value,
            span: ast_binding.span,
        });
        Ok(id)
    }

    fn statement(
        &mut self,
        statement: &AstStatement,
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Statement, Diagnostic> {
        match statement {
            AstStatement::Binding(binding) => {
                Ok(Statement::Binding(self.binding(binding, bindings)?))
            }
            AstStatement::Expression(expression) => {
                Ok(Statement::Expression(self.expression(expression, None)?))
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
                    Some(value) => Some(self.expression(value, Some(result_type))?),
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
                condition,
                then_block,
                else_block,
                ..
            } => {
                let boolean = self
                    .types
                    .resolve_name("bool")
                    .expect("bootstrap defines bool");
                let condition = self.expression(condition, Some(boolean))?;
                let outer_names = self.names.clone();
                let outer_declarations = self.declarations.clone();
                let then_block = self.block(then_block, bindings, result_type)?;
                self.names.clone_from(&outer_names);
                self.declarations.clone_from(&outer_declarations);
                let else_block = self.block(else_block, bindings, result_type)?;
                self.names = outer_names;
                self.declarations = outer_declarations;
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
        }
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
                catch_all = true;
            } else {
                catch_all = true;
            }

            self.names.clone_from(&outer_names);
            self.declarations.clone_from(&outer_declarations);
            let binding = if let Some(name) = &case.binding {
                let id = self.new_binding_id();
                let binding_type = type_id.unwrap_or(subject_type);
                self.names.insert(name.clone(), (id, binding_type));
                self.declarations.insert(name.clone());
                bindings.push(Binding {
                    id,
                    type_id: binding_type,
                    value: subject.clone(),
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

    fn block(
        &mut self,
        statements: &[AstStatement],
        bindings: &mut Vec<Binding>,
        result_type: TypeId,
    ) -> Result<Block, Diagnostic> {
        let mut block = Block::default();
        for statement in statements {
            block
                .statements
                .push(self.statement(statement, bindings, result_type)?);
        }
        Ok(block)
    }

    fn next_id(&mut self) -> HirId {
        let id = HirId(self.next_hir);
        self.next_hir += 1;
        id
    }

    fn prepare(&mut self, ast: &AstExpression) -> Result<Prepared, Diagnostic> {
        match &ast.kind {
            AstExpressionKind::Literal(value) => {
                Ok(Prepared::Literal(universal_literal(value), ast.span))
            }
            _ => self.expression(ast, None).map(Prepared::Resolved),
        }
    }

    fn finish(&mut self, prepared: Prepared, expected: TypeId) -> Result<Expression, Diagnostic> {
        match prepared {
            Prepared::Literal(value, span) => {
                let type_id = self
                    .types
                    .resolve_literal(&value, Some(expected))
                    .map_err(|error| semantic_error(error.to_string(), span))?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Literal(value),
                    span,
                })
            }
            Prepared::Resolved(expression)
                if self.types.assignable(expression.type_id, expected) =>
            {
                Ok(expression)
            }
            Prepared::Resolved(expression) => Err(semantic_error(
                "operator operand does not satisfy the selected signature".into(),
                expression.span,
            )),
        }
    }

    fn expression(
        &mut self,
        ast: &AstExpression,
        expected: Option<TypeId>,
    ) -> Result<Expression, Diagnostic> {
        match &ast.kind {
            AstExpressionKind::Literal(value) => {
                let value = universal_literal(value);
                let type_id = self
                    .types
                    .resolve_literal(&value, expected)
                    .map_err(|error| semantic_error(error.to_string(), ast.span))?;
                Ok(Expression {
                    id: self.next_id(),
                    type_id,
                    kind: ExpressionKind::Literal(value),
                    span: ast.span,
                })
            }
            AstExpressionKind::Name(name) => {
                let Some((binding, type_id)) = self.names.get(name).copied() else {
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
            AstExpressionKind::Member { .. } => Err(Diagnostic::new(
                "E000211",
                "member access is not implemented yet",
                Some(ast.span),
            )),
            AstExpressionKind::Call { callee, arguments } => {
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
                    if signature.parameters.len() != arguments.len()
                        || expected.is_some_and(|expected| {
                            !self.types.assignable(signature.result, expected)
                        })
                    {
                        continue;
                    }
                    let resolved = arguments
                        .iter()
                        .zip(&signature.parameters)
                        .map(|(argument, parameter)| self.expression(argument, Some(*parameter)))
                        .collect::<Result<Vec<_>, _>>();
                    if let Ok(arguments) = resolved {
                        let conversions = arguments
                            .iter()
                            .zip(&signature.parameters)
                            .map(|(argument, parameter)| {
                                conversion_rank(self.types, argument.type_id, *parameter)
                                    .expect("resolved arguments are assignable")
                            })
                            .collect::<Vec<_>>();
                        matches.push((conversions, function, signature.result, arguments));
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
                let [(_, function, result, arguments)] = best.as_slice() else {
                    return Err(Diagnostic::new(
                        "E000206",
                        format!("call to `{name}` has no unique compatible declaration"),
                        Some(ast.span),
                    ));
                };
                Ok(Expression {
                    id: self.next_id(),
                    type_id: *result,
                    kind: ExpressionKind::Call {
                        callee: severian_hir::Callee::Direct {
                            function: self.function_definitions[function],
                            substitution: severian_universal::Substitution::default(),
                        },
                        arguments: (*arguments).clone(),
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Unary { operator, operand } => {
                if *operator == AstUnaryOperator::Move {
                    return Err(Diagnostic::new(
                        "E000302",
                        "move checking is not implemented yet",
                        Some(ast.span),
                    ));
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
                let operator = universal_binary(*operator);
                // Both operands remain constraints until a single signature is
                // selected; neither side gets an early default literal type.
                let left = self.prepare(left)?;
                let right = self.prepare(right)?;
                let resolved = self
                    .types
                    .resolve_binary(operator, left.constraint(), right.constraint(), expected)
                    .map_err(|error| semantic_error(error.to_string(), ast.span))?;
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
}

fn synthetic_definition(
    module: &str,
    function: &severian_ast::FunctionDeclaration,
) -> DefId {
    DefId {
        package: 0,
        module: u128::from(severian_universal::DeclarationId::from_path(module).0),
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.function.{}.{:?}",
            function.name, function.type_parameters
        )),
    }
}

fn synthetic_test_definition(module: &str, ordinal: usize) -> DefId {
    DefId {
        package: 0,
        module: u128::from(severian_universal::DeclarationId::from_path(module).0),
        declaration: severian_universal::DeclarationId::from_path(&format!(
            "{module}.test.{ordinal}"
        )),
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

fn resolve_type_annotation(
    types: &TypeContext,
    annotation: &TypeAnnotation,
) -> Result<TypeId, Diagnostic> {
    let Some(name) = annotation.simple_name() else {
        return Err(Diagnostic::new(
            "E000204",
            "this source type form is not yet supported by universal resolution",
            Some(annotation.span),
        ));
    };
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
        ) => Some(ConversionRank::Widening(expected - actual)),
        (
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
        "bench" | "benchmark" => Ok(severian_hir::TestMode::Benchmark),
        "chaos" => Ok(severian_hir::TestMode::Chaos),
        "profile" => Ok(severian_hir::TestMode::Profile),
        "compiler" => Ok(severian_hir::TestMode::Compiler),
        "integ" | "integration" => Ok(severian_hir::TestMode::Integration),
        _ => Err(Diagnostic::new(
            "E000213",
            format!("unknown test runner `{name}`"),
            Some(span),
        )),
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
            AstStatement::Return { .. } => ControlFlow::Returns,
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
            AstStatement::Binding(_)
            | AstStatement::Expression(_)
            | AstStatement::Assert { .. }
            | AstStatement::If { .. }
            | AstStatement::Match { .. } => ControlFlow::FallsThrough,
        };
        if flow == ControlFlow::Returns {
            return flow;
        }
    }
    ControlFlow::FallsThrough
}

fn integration_expectation(statement: &AstStatement) -> Option<severian_hir::TestExpectation> {
    let AstStatement::Assert { condition, .. } = statement else {
        return None;
    };
    let AstExpressionKind::Binary {
        operator,
        left,
        right,
    } = &condition.kind
    else {
        return None;
    };
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
        AstLiteral::Boolean(value) => LiteralValue::Boolean(*value),
        AstLiteral::Character(value) => LiteralValue::Character(*value),
        AstLiteral::String(value) => LiteralValue::String(value.clone()),
        AstLiteral::Bytes(value) => LiteralValue::Bytes(value.clone()),
        AstLiteral::None => LiteralValue::None,
        AstLiteral::Unit => LiteralValue::Unit,
    }
}

fn universal_unary(operator: AstUnaryOperator) -> UnaryOperator {
    match operator {
        AstUnaryOperator::Positive => UnaryOperator::Positive,
        AstUnaryOperator::Negative => UnaryOperator::Negative,
        AstUnaryOperator::Not => UnaryOperator::Not,
        AstUnaryOperator::Move => unreachable!("move is rejected before universal resolution"),
    }
}

fn universal_binary(operator: AstBinaryOperator) -> BinaryOperator {
    match operator {
        AstBinaryOperator::Add => BinaryOperator::Add,
        AstBinaryOperator::Subtract => BinaryOperator::Subtract,
        AstBinaryOperator::Multiply => BinaryOperator::Multiply,
        AstBinaryOperator::Divide => BinaryOperator::Divide,
        AstBinaryOperator::Remainder => BinaryOperator::Remainder,
        AstBinaryOperator::Power => BinaryOperator::Power,
        AstBinaryOperator::Equal => BinaryOperator::Equal,
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
    fn compiler_tests_reject_unimplemented_body_and_diagnostic_assertions() {
        let context = severian_bootstrap::load().unwrap();
        for source_text in [
            "test with compiler:\n    assert(false)\n    reject:\n        missing()\n",
            "test with compiler:\n    reject error:\n        missing()\n",
        ] {
            let source = SourceFile::virtual_source("compiler-test.sev", source_text);
            let tokens = severian_lexer::scan(&source).unwrap();
            let ast = severian_parser::parse(&tokens).unwrap();
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
    fn conversion_categories_order_exact_before_widening_before_general() {
        let context = severian_bootstrap::load().unwrap();
        let resolve = |name| context.types.resolve_name(name).unwrap();
        let exact = conversion_rank(&context.types, resolve("i32"), resolve("i32")).unwrap();
        let widening = conversion_rank(&context.types, resolve("i32"), resolve("i64")).unwrap();
        let general = conversion_rank(&context.types, resolve("bf16"), resolve("f32")).unwrap();
        assert!(exact < widening);
        assert!(widening < general);
    }
}
