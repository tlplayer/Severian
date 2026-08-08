use super::constant_fold::{fold_constant_binary, fold_constant_unary};
use crate::{Pass, PassError};
use severian_hir::{BinaryOp, Expression, Program};

#[derive(Debug, Clone, Copy)]
pub struct AlgebraicSimplification {
    pub max_rounds: usize,
}

impl Default for AlgebraicSimplification {
    fn default() -> Self {
        Self { max_rounds: 8 }
    }
}

impl Pass for AlgebraicSimplification {
    fn name(&self) -> &'static str {
        "xla-algebraic-simplification"
    }

    fn run(&self, program: &mut Program) -> Result<(), PassError> {
        for _ in 0..self.max_rounds {
            let mut changed = false;
            program.visit_expressions_mut(&mut |expression| {
                changed |= simplify_expression(expression);
            });

            if !changed {
                break;
            }
        }

        Ok(())
    }
}

pub fn simplify_expression(expression: &mut Expression) -> bool {
    let replacement = match expression {
        Expression::Unary { op, expression } => fold_constant_unary(*op, expression),

        Expression::Binary { left, op, right } => {
            fold_constant_binary(left, *op, right)
                .or_else(|| simplify_binary(left, *op, right))
        }

        Expression::Conditional {
            condition,
            then_expression,
            else_expression,
        } => match condition.as_ref() {
            Expression::Boolean(true) => Some((**then_expression).clone()),
            Expression::Boolean(false) => Some((**else_expression).clone()),
            _ if then_expression == else_expression => Some((**then_expression).clone()),
            _ => None,
        },

        _ => None,
    };

    if let Some(replacement) = replacement {
        *expression = replacement;
        true
    } else {
        false
    }
}

fn simplify_binary(
    left: &Expression,
    op: BinaryOp,
    right: &Expression,
) -> Option<Expression> {
    match op {
        BinaryOp::Add => {
            if is_zero(right) {
                Some(left.clone())
            } else if is_zero(left) {
                Some(right.clone())
            } else {
                None
            }
        }

        BinaryOp::Sub if is_zero(right) => Some(left.clone()),

        BinaryOp::Mul => {
            if is_one(right) {
                Some(left.clone())
            } else if is_one(left) {
                Some(right.clone())
            } else if is_zero(right) && is_obviously_pure(left) {
                Some(right.clone())
            } else if is_zero(left) && is_obviously_pure(right) {
                Some(left.clone())
            } else {
                None
            }
        }

        BinaryOp::Div if is_one(right) => Some(left.clone()),

        BinaryOp::And => match (left, right) {
            (Expression::Boolean(true), rhs) => Some(rhs.clone()),
            (Expression::Boolean(false), _) => Some(Expression::Boolean(false)),
            (lhs, Expression::Boolean(true)) => Some(lhs.clone()),
            _ => None,
        },

        BinaryOp::Or => match (left, right) {
            (Expression::Boolean(false), rhs) => Some(rhs.clone()),
            (Expression::Boolean(true), _) => Some(Expression::Boolean(true)),
            (lhs, Expression::Boolean(false)) => Some(lhs.clone()),
            _ => None,
        },

        BinaryOp::Equal if left == right && is_obviously_pure(left) => {
            Some(Expression::Boolean(true))
        }

        BinaryOp::NotEqual if left == right && is_obviously_pure(left) => {
            Some(Expression::Boolean(false))
        }

        _ => None,
    }
}

fn is_zero(expression: &Expression) -> bool {
    matches!(expression, Expression::Integer(0))
        || matches!(expression, Expression::Float(bits) if f64::from_bits(*bits) == 0.0)
}

fn is_one(expression: &Expression) -> bool {
    matches!(expression, Expression::Integer(1))
        || matches!(expression, Expression::Float(bits) if f64::from_bits(*bits) == 1.0)
}

fn is_obviously_pure(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Boolean(_)
            | Expression::String(_)
            | Expression::Variable(_)
            | Expression::Function(_)
    )
}
