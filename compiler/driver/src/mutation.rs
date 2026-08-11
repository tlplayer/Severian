use crate::Compilation;
use severian_hir::{BinaryOp, Expression, HirId, SourceFile, SourceSpan};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Mutation {
    pub index: usize,
    pub description: String,
    pub file: Option<PathBuf>,
    pub line: Option<u32>,
}

pub fn count(compilation: &Compilation) -> usize {
    let mut program = production_program(compilation);
    let mut count = 0;
    program.visit_expressions_mut(&mut |expression| {
        count += mutations(expression).len();
    });
    count
}

pub fn apply(compilation: &Compilation, wanted: usize) -> Option<(Compilation, Mutation)> {
    let mut hir = production_program(compilation);
    let sources = hir.metadata.sources.clone();
    let mut current = 0;
    let mut selected = None;
    hir.visit_expressions_mut(&mut |expression| {
        let candidates = mutations(expression);
        if selected.is_some() || wanted < current || wanted >= current + candidates.len() {
            current += candidates.len();
            return;
        }
        let (id, replacement, description) = candidates[wanted - current].clone();
        let (file, line) = location(&sources, id);
        *expression = replacement;
        selected = Some(Mutation {
            index: wanted,
            description,
            file,
            line,
        });
        current += candidates.len();
    });
    let mutation = selected?;
    let mir = severian_mir::lower(&hir);
    let mlir = severian_lowering::lower(&mir);
    Some((
        Compilation {
            hir: compilation.hir.clone(),
            optimized_hir: hir,
            mir,
            mlir,
        },
        mutation,
    ))
}

fn production_program(compilation: &Compilation) -> severian_hir::Program {
    let mut program = compilation.optimized_hir.clone();
    for function in &mut program.functions {
        function.tests.clear();
    }
    for class in &mut program.classes {
        for function in class.methods.iter_mut().chain(&mut class.constructors) {
            function.tests.clear();
        }
    }
    program
}

fn mutations(expression: &Expression) -> Vec<(HirId, Expression, String)> {
    let Expression::Typed {
        id,
        ty,
        expression: inner,
    } = expression
    else {
        return Vec::new();
    };
    let candidates = match inner.as_ref() {
        Expression::Boolean(value) => vec![(
            Expression::Boolean(!value),
            format!("boolean {value} -> {}", !value),
        )],
        Expression::Integer(value) => vec![if *value == 0 {
            (Expression::Integer(1), "integer 0 -> 1".into())
        } else {
            (Expression::Integer(0), format!("integer {value} -> 0"))
        }],
        Expression::Float(value) => {
            let number = f64::from_bits(*value);
            let replacement: f64 = if number == 0.0 { 1.0 } else { 0.0 };
            vec![(
                Expression::Float(replacement.to_bits()),
                format!("float {number} -> {replacement}"),
            )]
        }
        Expression::Binary { left, op, right } => replacement_ops(*op)
            .into_iter()
            .map(|replacement| {
                (
                    Expression::Binary {
                        left: left.clone(),
                        op: replacement,
                        right: right.clone(),
                    },
                    format!("{} -> {}", op_name(*op), op_name(replacement)),
                )
            })
            .collect(),
        Expression::Unary { op, expression } => vec![(
            expression.as_ref().clone(),
            format!("remove unary {}", unary_name(*op)),
        )],
        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => vec![
            (
                Expression::Conditional {
                    condition: Box::new(Expression::Unary {
                        op: severian_hir::UnaryOp::Not,
                        expression: condition.clone(),
                    }),
                    then_expression: then_expression.clone(),
                    else_expression: else_expression.clone(),
                },
                "negate conditional condition".into(),
            ),
            (
                Expression::Conditional {
                    condition: condition.clone(),
                    then_expression: else_expression.clone(),
                    else_expression: then_expression.clone(),
                },
                "swap conditional branches".into(),
            ),
        ],
        Expression::Call { .. } | Expression::CallValue { .. } => default_value(*ty)
            .map(|replacement| vec![(replacement, "replace call result with default".into())])
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    candidates
        .into_iter()
        .map(|(replacement, description)| {
            (
                *id,
                Expression::Typed {
                    id: *id,
                    ty: *ty,
                    expression: Box::new(replacement),
                },
                description,
            )
        })
        .collect()
}

fn replacement_ops(op: BinaryOp) -> Vec<BinaryOp> {
    match op {
        BinaryOp::Add => vec![BinaryOp::Sub],
        BinaryOp::Sub => vec![BinaryOp::Add],
        BinaryOp::Mul => vec![BinaryOp::Div],
        BinaryOp::Div => vec![BinaryOp::Mul],
        BinaryOp::Mod => vec![BinaryOp::Div],
        BinaryOp::Power => vec![BinaryOp::Mul],
        BinaryOp::Equal => vec![BinaryOp::NotEqual],
        BinaryOp::NotEqual => vec![BinaryOp::Equal],
        BinaryOp::Less => vec![BinaryOp::LessEqual, BinaryOp::GreaterEqual],
        BinaryOp::LessEqual => vec![BinaryOp::Less, BinaryOp::Greater],
        BinaryOp::Greater => vec![BinaryOp::GreaterEqual, BinaryOp::LessEqual],
        BinaryOp::GreaterEqual => vec![BinaryOp::Greater, BinaryOp::Less],
        BinaryOp::And => vec![BinaryOp::Or],
        BinaryOp::Or => vec![BinaryOp::And],
        BinaryOp::In => Vec::new(),
    }
}

fn default_value(ty: severian_hir::ValueType) -> Option<Expression> {
    use severian_hir::ValueType;
    Some(match ty {
        ValueType::Bool => Expression::Boolean(false),
        ValueType::Int => Expression::Integer(0),
        ValueType::Float => Expression::Float(0.0f64.to_bits()),
        ValueType::String => Expression::String(String::new()),
        ValueType::List => Expression::List(Vec::new()),
        _ => return None,
    })
}

fn unary_name(op: severian_hir::UnaryOp) -> &'static str {
    match op {
        severian_hir::UnaryOp::Negate => "-",
        severian_hir::UnaryOp::Not => "not",
    }
}

fn op_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Power => "**",
        BinaryOp::Equal => "==",
        BinaryOp::NotEqual => "!=",
        BinaryOp::Less => "<",
        BinaryOp::LessEqual => "<=",
        BinaryOp::Greater => ">",
        BinaryOp::GreaterEqual => ">=",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::In => "in",
    }
}

fn location(sources: &severian_hir::SourceMap, id: HirId) -> (Option<PathBuf>, Option<u32>) {
    let Some(SourceSpan { file, range }) = sources.expression_span(id) else {
        return (None, None);
    };
    let Some(file) = sources.file(file) else {
        return (None, None);
    };
    (Some(file.path.clone()), line(file, range.start))
}

fn line(file: &SourceFile, byte: usize) -> Option<u32> {
    (byte <= file.source.len()).then(|| {
        file.source[..byte]
            .bytes()
            .filter(|value| *value == b'\n')
            .count() as u32
            + 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_mutations_cross_the_boundary() {
        assert_eq!(
            replacement_ops(BinaryOp::Greater),
            [BinaryOp::GreaterEqual, BinaryOp::LessEqual]
        );
        assert_eq!(
            replacement_ops(BinaryOp::LessEqual),
            [BinaryOp::Less, BinaryOp::Greater]
        );
    }

    #[test]
    fn arithmetic_mutations_cover_modulo_and_power() {
        assert_eq!(replacement_ops(BinaryOp::Mod), [BinaryOp::Div]);
        assert_eq!(replacement_ops(BinaryOp::Power), [BinaryOp::Mul]);
    }
}
