#![forbid(unsafe_code)]

use severian_ast::{
    BinaryOperator as AstBinaryOperator, Expression as AstExpression,
    ExpressionKind as AstExpressionKind, Literal as AstLiteral, TypeAnnotation,
    UnaryOperator as AstUnaryOperator,
};
use severian_diagnostics::Diagnostic;
use severian_hir::{
    Binding, BindingId, Expression, ExpressionKind, HirId, Module, Program, TypeId,
};
use severian_universal::{
    BinaryOperator, LiteralValue, TypeConstraint, TypeContext, UnaryOperator,
};
use std::collections::BTreeMap;

pub fn analyze(ast: &severian_ast::Module, types: &TypeContext) -> Result<Program, Diagnostic> {
    let mut analyzer = Analyzer {
        types,
        names: BTreeMap::new(),
        next_hir: 0,
        next_binding: 0,
    };
    let mut bindings = Vec::new();
    for ast_binding in ast.items.iter().filter_map(|item| match item {
        severian_ast::Item::Binding(binding) => Some(binding),
        _ => None,
    }) {
        if analyzer.names.contains_key(&ast_binding.name) {
            return Err(Diagnostic::new(
                "E000203",
                format!("binding `{}` is already defined", ast_binding.name),
                Some(ast_binding.span),
            ));
        }
        let expected = ast_binding
            .annotation
            .as_ref()
            .map(|annotation| resolve_type_annotation(types, annotation))
            .transpose()?;
        let value = analyzer.expression(&ast_binding.value, expected)?;
        let type_id = expected.unwrap_or(value.type_id);
        if !types.assignable(value.type_id, type_id) {
            return Err(Diagnostic::new(
                "E000205",
                "binding value is not assignable to its declared type",
                Some(ast_binding.value.span),
            ));
        }
        let id = BindingId(analyzer.next_binding);
        analyzer.next_binding += 1;
        analyzer
            .names
            .insert(ast_binding.name.clone(), (id, type_id));
        bindings.push(Binding {
            id,
            type_id,
            value,
            span: ast_binding.span,
        });
    }
    Ok(Program {
        modules: vec![Module { bindings }],
    })
}

struct Analyzer<'a> {
    types: &'a TypeContext,
    names: BTreeMap<String, (BindingId, TypeId)>,
    next_hir: u32,
    next_binding: u32,
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

fn universal_literal(literal: &AstLiteral) -> LiteralValue {
    match literal {
        AstLiteral::Integer(value) => LiteralValue::Integer(value.clone()),
        AstLiteral::Float(value) => LiteralValue::Float(value.clone()),
        AstLiteral::Boolean(value) => LiteralValue::Boolean(*value),
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
    use severian_universal::TargetSpec;

    fn analyze_source(source: &str) -> (Program, severian_universal::UniversalContext) {
        let context = severian_bootstrap::load(TargetSpec::host()).unwrap();
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
}
