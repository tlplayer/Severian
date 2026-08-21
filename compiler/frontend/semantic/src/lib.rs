#![forbid(unsafe_code)]

use severian_ast::{
    BinaryOperator as AstBinaryOperator, Expression as AstExpression,
    ExpressionKind as AstExpressionKind, Literal as AstLiteral, Statement as AstStatement,
    TypeAnnotation, UnaryOperator as AstUnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_hir::{
    Binding, BindingId, Block, BoundaryType, CallType, Expression, ExpressionKind,
    FunctionDeclaration, FunctionId, FunctionParameter, HirId, Module, Program, SemanticType,
    Statement, TypeId,
};
use severian_universal::{
    BinaryOperator, LiteralValue, TypeConstraint, TypeContext, UnaryOperator,
};
use std::collections::{BTreeMap, BTreeSet};

pub fn analyze(ast: &severian_ast::Module, types: &TypeContext) -> Result<Program, Diagnostic> {
    let mut analyzer = Analyzer {
        types,
        names: BTreeMap::new(),
        declarations: BTreeSet::new(),
        functions: BTreeMap::new(),
        signatures: Vec::new(),
        next_hir: 0,
        next_binding: 0,
    };
    let mut module = Module::default();

    // Function identities and signatures are registered before executable
    // statements are analyzed. Bodies remain ordinary analyzed blocks.
    for ast_function in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Function(function) => Some(function),
        _ => None,
    }) {
        let id = FunctionId(module.functions.len() as u32);
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
        analyzer
            .functions
            .entry(ast_function.name.clone())
            .or_default()
            .push(id);
        analyzer.signatures.push(FunctionSignature {
            parameters: parameter_types,
            result,
        });
        module.functions.push(FunctionDeclaration {
            id,
            name: ast_function.name.clone(),
            parameters,
            result: universal_boundary(result),
            compile_route,
            call_type: CallType::Severian,
            body: None,
        });
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
            let SemanticType::Universal(type_id) = parameter.contract.ty else {
                unreachable!("source parameters are universally resolved")
            };
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
        for statement in ast_body {
            body.statements
                .push(analyzer.statement(statement, &mut module.bindings)?);
        }
        let SemanticType::Universal(result_type) = function.result.ty else {
            unreachable!("source results are universally resolved")
        };
        if result_type != types.resolve_name("unit").expect("bootstrap defines unit") {
            return Err(Diagnostic::new(
                "E000209",
                "a function body with a non-unit result requires an explicit return",
                Some(ast_function.span),
            ));
        }
        function.body = Some(body);
        if function.name == "main" {
            let arguments_type = types.resolve_name("args").expect("bootstrap defines args");
            let valid_parameters = match function.parameters.as_slice() {
                [] => true,
                [parameter] => parameter.contract.ty == SemanticType::Universal(arguments_type),
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
    signatures: Vec<FunctionSignature>,
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
        if !self.declarations.insert(ast_binding.name.clone()) {
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
            .transpose()?;
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
    ) -> Result<Statement, Diagnostic> {
        match statement {
            AstStatement::Binding(binding) => {
                Ok(Statement::Binding(self.binding(binding, bindings)?))
            }
            AstStatement::Expression(expression) => {
                Ok(Statement::Expression(self.expression(expression, None)?))
            }
        }
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
            AstExpressionKind::Call { callee, arguments } => {
                let AstExpressionKind::Name(name) = &callee.kind else {
                    return Err(Diagnostic::new(
                        "E000206",
                        "call target must resolve to a function declaration",
                        Some(callee.span),
                    ));
                };
                let candidates = self.functions.get(name).cloned().unwrap_or_default();
                let mut matches = Vec::new();
                for function in candidates {
                    let signature = self.signatures[function.0 as usize].clone();
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
                        matches.push((function, signature.result, arguments));
                    }
                }
                let [(function, result, arguments)] = matches.as_slice() else {
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
                        function: *function,
                        arguments: arguments.clone(),
                    },
                    span: ast.span,
                })
            }
            AstExpressionKind::Unary { operator, operand } => {
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
        ty: SemanticType::Universal(type_id),
        modifiers: Vec::new(),
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
}
