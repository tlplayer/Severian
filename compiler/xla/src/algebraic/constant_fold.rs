use severian_hir::{BinaryOp, Expression, UnaryOp};

pub fn fold_constant_unary(op: UnaryOp, value: &Expression) -> Option<Expression> {
    match (op, value) {
        (UnaryOp::Negate, Expression::Integer(value)) => {
            value.checked_neg().map(Expression::Integer)
        }
        (UnaryOp::Negate, Expression::Float(bits)) => {
            Some(Expression::Float((-f64::from_bits(*bits)).to_bits()))
        }
        (UnaryOp::Not, Expression::Boolean(value)) => Some(Expression::Boolean(!value)),
        _ => None,
    }
}

pub fn fold_constant_binary(
    left: &Expression,
    op: BinaryOp,
    right: &Expression,
) -> Option<Expression> {
    match (left, right) {
        (Expression::Integer(left), Expression::Integer(right)) => {
            return fold_integer(*left, op, *right);
        }
        (Expression::Float(left), Expression::Float(right)) => {
            return fold_float(f64::from_bits(*left), op, f64::from_bits(*right));
        }
        (Expression::Boolean(left), Expression::Boolean(right)) => {
            return fold_boolean(*left, op, *right);
        }
        (Expression::String(left), Expression::String(right)) => {
            return fold_string(left, op, right);
        }
        _ => {}
    }

    None
}

fn fold_integer(left: i64, op: BinaryOp, right: i64) -> Option<Expression> {
    match op {
        BinaryOp::Add => left.checked_add(right).map(Expression::Integer),
        BinaryOp::Sub => left.checked_sub(right).map(Expression::Integer),
        BinaryOp::Mul => left.checked_mul(right).map(Expression::Integer),
        BinaryOp::Div => left.checked_div(right).map(Expression::Integer),
        BinaryOp::Mod => left.checked_rem(right).map(Expression::Integer),
        BinaryOp::Power => {
            let exponent = u32::try_from(right).ok()?;
            left.checked_pow(exponent).map(Expression::Integer)
        }
        BinaryOp::Equal => Some(Expression::Boolean(left == right)),
        BinaryOp::NotEqual => Some(Expression::Boolean(left != right)),
        BinaryOp::Less => Some(Expression::Boolean(left < right)),
        BinaryOp::LessEqual => Some(Expression::Boolean(left <= right)),
        BinaryOp::Greater => Some(Expression::Boolean(left > right)),
        BinaryOp::GreaterEqual => Some(Expression::Boolean(left >= right)),
        _ => None,
    }
}

fn fold_float(left: f64, op: BinaryOp, right: f64) -> Option<Expression> {
    let value = match op {
        BinaryOp::Add => return Some(Expression::Float((left + right).to_bits())),
        BinaryOp::Sub => return Some(Expression::Float((left - right).to_bits())),
        BinaryOp::Mul => return Some(Expression::Float((left * right).to_bits())),
        BinaryOp::Div if right != 0.0 => {
            return Some(Expression::Float((left / right).to_bits()))
        }
        BinaryOp::Mod if right != 0.0 => {
            return Some(Expression::Float((left % right).to_bits()))
        }
        BinaryOp::Power => return Some(Expression::Float(left.powf(right).to_bits())),
        BinaryOp::Equal => left == right,
        BinaryOp::NotEqual => left != right,
        BinaryOp::Less => left < right,
        BinaryOp::LessEqual => left <= right,
        BinaryOp::Greater => left > right,
        BinaryOp::GreaterEqual => left >= right,
        _ => return None,
    };

    Some(Expression::Boolean(value))
}

fn fold_boolean(left: bool, op: BinaryOp, right: bool) -> Option<Expression> {
    match op {
        BinaryOp::And => Some(Expression::Boolean(left && right)),
        BinaryOp::Or => Some(Expression::Boolean(left || right)),
        BinaryOp::Equal => Some(Expression::Boolean(left == right)),
        BinaryOp::NotEqual => Some(Expression::Boolean(left != right)),
        _ => None,
    }
}

fn fold_string(left: &str, op: BinaryOp, right: &str) -> Option<Expression> {
    match op {
        BinaryOp::Add => Some(Expression::String(format!("{left}{right}"))),
        BinaryOp::Equal => Some(Expression::Boolean(left == right)),
        BinaryOp::NotEqual => Some(Expression::Boolean(left != right)),
        BinaryOp::Less => Some(Expression::Boolean(left < right)),
        BinaryOp::LessEqual => Some(Expression::Boolean(left <= right)),
        BinaryOp::Greater => Some(Expression::Boolean(left > right)),
        BinaryOp::GreaterEqual => Some(Expression::Boolean(left >= right)),
        _ => None,
    }
}
