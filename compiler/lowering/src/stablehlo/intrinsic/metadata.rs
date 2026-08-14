use crate::stablehlo::{MlirValue, StableHloLoweringError};
use severian_hir::{Expression, TensorDimension, TensorElementType, TensorType, UnaryOp};

pub(super) fn integer_list_argument(
    expression: Option<&Expression>,
    operation: &str,
) -> Result<Vec<u64>, StableHloLoweringError> {
    let Some(Expression::List(values)) = expression.map(Expression::kind) else {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "expected a compile-time integer list".into(),
        });
    };
    values
        .iter()
        .map(|value| match value.kind() {
            Expression::Integer(value) if *value >= 0 => Ok(*value as u64),
            _ => Err(StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "expected non-negative compile-time integer metadata".into(),
            }),
        })
        .collect()
}

pub(super) fn float_argument(
    expression: Option<&Expression>,
    operation: &str,
) -> Result<String, StableHloLoweringError> {
    match expression.map(Expression::kind) {
        Some(Expression::Float(bits)) => {
            let mut literal = f64::from_bits(*bits).to_string();
            if !literal.contains(['.', 'e', 'E']) {
                literal.push_str(".0");
            }
            Ok(literal)
        }
        Some(Expression::Integer(value)) => Ok(format!("{value}.0")),
        _ => Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "expected a compile-time scalar".into(),
        }),
    }
}

pub(super) fn integer_argument(
    expression: Option<&Expression>,
    operation: &str,
) -> Result<u64, StableHloLoweringError> {
    match expression.map(Expression::kind) {
        Some(Expression::Integer(value)) if *value >= 0 => Ok(*value as u64),
        _ => Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "expected a non-negative compile-time integer".into(),
        }),
    }
}

pub(super) fn signed_integer_argument(
    expression: Option<&Expression>,
    operation: &str,
) -> Result<i64, StableHloLoweringError> {
    match expression.map(Expression::kind) {
        Some(Expression::Integer(value)) => Ok(*value),
        Some(Expression::Unary {
            op: UnaryOp::Negate,
            expression,
        }) => match expression.kind() {
            Expression::Integer(value) => {
                value
                    .checked_neg()
                    .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                        function: operation.into(),
                        reason: "integer axis is outside the supported range".into(),
                    })
            }
            _ => Err(StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "expected a compile-time integer".into(),
            }),
        },
        _ => Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "expected a compile-time integer".into(),
        }),
    }
}

pub(super) fn reduced_suffix_axes(
    operation: &str,
    input: &MlirValue,
    result_type: TensorType,
) -> Result<Vec<u64>, StableHloLoweringError> {
    let input_rank = input
        .tensor_type()
        .and_then(|tensor| tensor.rank)
        .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: "reduction input is missing ranked tensor metadata".into(),
        })?;
    let result_rank =
        result_type
            .rank
            .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "reduction result is missing ranked tensor metadata".into(),
            })?;
    if result_rank >= input_rank {
        return Err(StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: format!(
                "suffix reduction must lower rank (input {input_rank}, result {result_rank})"
            ),
        });
    }
    Ok((u64::from(result_rank)..u64::from(input_rank)).collect())
}

pub(super) fn static_reduction_count(
    operation: &str,
    input: &MlirValue,
    axes: &[u64],
) -> Result<u64, StableHloLoweringError> {
    let shape = input
        .ty
        .strip_prefix("tensor<")
        .and_then(|value| value.strip_suffix('>'))
        .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
            function: operation.into(),
            reason: format!("expected ranked tensor type, got {}", input.ty),
        })?;
    let dimensions = shape.split('x').collect::<Vec<_>>();
    axes.iter().try_fold(1u64, |count, &axis| {
        let dimension = dimensions.get(axis as usize).ok_or_else(|| {
            StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: format!("axis {axis} is outside type {}", input.ty),
            }
        })?;
        let dimension =
            dimension
                .parse::<u64>()
                .map_err(|_| StableHloLoweringError::UnsupportedFunction {
                    function: operation.into(),
                    reason: "mean/norm requires static reduced dimensions".into(),
                })?;
        count
            .checked_mul(dimension)
            .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                function: operation.into(),
                reason: "reduction element count overflow".into(),
            })
    })
}

pub(crate) fn scalar_tensor(element: TensorElementType) -> TensorType {
    TensorType {
        element,
        rank: Some(0),
        dimensions: [TensorDimension::Dynamic; 8],
    }
}

pub(super) fn require_arity(
    operation: &str,
    args: &[MlirValue],
    expected: usize,
) -> Result<(), StableHloLoweringError> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(StableHloLoweringError::InvalidArity {
            operation: operation.to_string(),
            expected,
            actual: args.len(),
        })
    }
}
