use super::shape::tensor_error;
use super::*;

pub(super) fn tensor_from_shape(
    element: TensorElementType,
    expression: Option<&Expression>,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    let dimensions =
        expression.ok_or_else(|| tensor_error(span, operation, "missing shape metadata"))?;
    let dimensions = integer_shape(dimensions, operation, span)?;
    TensorType::ranked(element, &dimensions).map_err(|reason| tensor_error(span, operation, reason))
}

pub(super) fn tensor_from_reshape_shape(
    element: TensorElementType,
    expression: Option<&Expression>,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    let shape =
        expression.ok_or_else(|| tensor_error(span, operation, "missing shape metadata"))?;
    if matches!(shape.ty(), Some(ValueType::List)) && !matches!(shape.kind(), Expression::List(_)) {
        return Ok(TensorType::dynamic(element));
    }
    tensor_from_shape(element, Some(shape), operation, span)
}

pub(super) fn integer_shape(
    expression: &Expression,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<Vec<TensorDimension>, SemanticError> {
    let Expression::List(values) = expression.kind() else {
        return Err(tensor_error(
            span,
            operation,
            "expected a compile-time shape",
        ));
    };
    let dimensions = values
        .iter()
        .map(|value| match signed_integer(value) {
            Some(value) if value >= 0 => Ok(TensorDimension::Static(value as u64)),
            Some(-1) => Ok(TensorDimension::Dynamic),
            Some(_) => Err(tensor_error(
                span,
                operation,
                "shape dimensions cannot be negative except for -1",
            )),
            None if !matches!(
                value.kind(),
                Expression::Float(_)
                    | Expression::Boolean(_)
                    | Expression::String(_)
                    | Expression::List(_)
                    | Expression::Tuple(_)
                    | Expression::Map(_)
                    | Expression::Set(_)
            ) =>
            {
                Ok(TensorDimension::Dynamic)
            }
            None => Err(tensor_error(
                span,
                operation,
                "shape entries must be integers",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(dimensions)
}

pub(super) fn shape_integer_list_argument(
    arguments: &[Expression],
    index: usize,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<Vec<Option<i64>>, SemanticError> {
    let expression = arguments
        .get(index)
        .ok_or_else(|| tensor_error(span, operation, "missing integer-list metadata"))?;
    let Expression::List(values) = expression.kind() else {
        return Err(tensor_error(
            span,
            operation,
            "expected an integer metadata list",
        ));
    };
    values
        .iter()
        .map(|value| {
            if let Some(value) = signed_integer(value) {
                Ok(Some(value))
            } else if matches!(
                value.kind(),
                Expression::Float(_)
                    | Expression::Boolean(_)
                    | Expression::String(_)
                    | Expression::List(_)
                    | Expression::Tuple(_)
                    | Expression::Map(_)
                    | Expression::Set(_)
            ) {
                Err(tensor_error(
                    span,
                    operation,
                    "slice metadata entries must be integers",
                ))
            } else {
                Ok(None)
            }
        })
        .collect()
}

pub(super) fn integer_list(
    expression: &Expression,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<Vec<u64>, SemanticError> {
    let Expression::List(values) = expression.kind() else {
        return Err(tensor_error(
            span,
            operation,
            "expected a compile-time integer list",
        ));
    };
    values
        .iter()
        .map(|value| match signed_integer(value) {
            Some(value) if value >= 0 => Ok(value as u64),
            _ => Err(tensor_error(
                span,
                operation,
                "expected non-negative compile-time integer metadata",
            )),
        })
        .collect()
}

pub(super) fn validate_axis(
    tensor: TensorType,
    expression: Option<&Expression>,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<(), SemanticError> {
    let Some(rank) = tensor.rank else {
        return Ok(());
    };
    let requested = expression
        .and_then(signed_integer)
        .ok_or_else(|| tensor_error(span, operation, "expected a compile-time axis"))?;
    normalized_axis(requested, rank, operation, span).map(|_| ())
}

pub(super) fn signed_integer_argument(
    arguments: &[Expression],
    index: usize,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<i64, SemanticError> {
    arguments
        .get(index)
        .and_then(signed_integer)
        .ok_or_else(|| tensor_error(span, operation, "expected a compile-time integer"))
}

pub(super) fn signed_integer(expression: &Expression) -> Option<i64> {
    match expression.kind() {
        Expression::Integer(value) => Some(*value),
        Expression::Unary {
            op: UnaryOp::Negate,
            expression,
        } => match expression.kind() {
            Expression::Integer(value) => value.checked_neg(),
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn normalized_axis(
    requested: i64,
    rank: u8,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<usize, SemanticError> {
    let axis = if requested < 0 {
        i64::from(rank) + requested
    } else {
        requested
    };
    if axis < 0 || axis >= i64::from(rank) {
        return Err(tensor_error(
            span,
            operation,
            format!("axis {requested} is outside rank {rank}"),
        ));
    }
    Ok(axis as usize)
}
