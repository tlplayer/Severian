use super::metadata::{
    integer_list, normalized_axis, shape_integer_list_argument, signed_integer_argument,
};
use super::*;

pub(super) fn broadcast(
    (left, right): (TensorType, TensorType),
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    left.broadcast_with(right).map_err(|reason| {
        tensor_error(
            span,
            operation,
            format!(
                "{reason}: left is `{}`, right is `{}`",
                value_type_name(ValueType::Tensor(left)),
                value_type_name(ValueType::Tensor(right))
            ),
        )
    })
}

pub(super) fn require_broadcast_to(
    left: TensorType,
    mut right: TensorType,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<(), SemanticError> {
    right.element = left.element;
    let result = broadcast((left, right), operation, span)?;
    if result.is_compatible_with(right) {
        Ok(())
    } else {
        Err(tensor_error(
            span,
            operation,
            format!(
                "input `{}` does not broadcast to result `{}`",
                value_type_name(ValueType::Tensor(left)),
                value_type_name(ValueType::Tensor(right))
            ),
        ))
    }
}

/// Validate the contiguous `broadcast_in_dim` used by `broadcast_like`.
///
/// Unlike ordinary elementwise broadcasting, the input dimensions may align
/// with any contiguous window in the target. This represents both a reduced
/// prefix such as `[batch, sequence] -> [batch, sequence, hidden]` and a
/// feature weight such as `[hidden] -> [batch, sequence, hidden]` without
/// encoding model-specific axes in the compiler.
pub(super) fn require_broadcast_like_to(
    left: TensorType,
    right: TensorType,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<(), SemanticError> {
    let (Some(left_rank), Some(right_rank)) = (left.rank, right.rank) else {
        return Ok(());
    };
    let left_rank = usize::from(left_rank);
    let right_rank = usize::from(right_rank);
    if left_rank > right_rank {
        return Err(tensor_error(
            span,
            operation,
            "input rank exceeds broadcast target rank",
        ));
    }
    let compatible = |left: TensorDimension, right: TensorDimension| {
        !dimensions_conflict(left, right)
            || matches!(left, TensorDimension::Static(1) | TensorDimension::Dynamic)
    };
    let has_mapping = (0..=right_rank - left_rank).any(|start| {
        (0..left_rank).all(|axis| compatible(left.dimensions[axis], right.dimensions[start + axis]))
    });
    if has_mapping {
        Ok(())
    } else {
        Err(tensor_error(
            span,
            operation,
            format!(
                "input `{}` has no contiguous broadcast mapping to `{}`",
                value_type_name(ValueType::Tensor(left)),
                value_type_name(ValueType::Tensor(right))
            ),
        ))
    }
}

pub(crate) fn infer_matmul_operator(
    left: ValueType,
    right: ValueType,
    span: Span,
) -> Result<ValueType, SemanticError> {
    if matches!(left, ValueType::Any | ValueType::TensorAny)
        || matches!(right, ValueType::Any | ValueType::TensorAny)
    {
        return Ok(ValueType::TensorAny);
    }
    let (ValueType::Tensor(left), ValueType::Tensor(right)) = (left, right) else {
        return Err(error(span, "operator `X` requires two tensors"));
    };
    infer_matmul(left, right, span)
        .map(ValueType::Tensor)
        .map_err(|error| {
            let (Some(left_rank), Some(right_rank)) = (left.rank, right.rank) else {
                return error;
            };
            if left_rank == 0 || right_rank == 0 {
                return error;
            }
            let left_contracting = left.dimensions[left_rank as usize - 1];
            let right_contracting = right.dimensions[right_rank as usize - 2];
            if !dimensions_conflict(left_contracting, right_contracting) {
                return error;
            }
            super::error(
                span,
                format!(
                    "E002401: incompatible tensor dimensions; left is `{}`; right is `{}`; requires `{} == {}`",
                    value_type_name(ValueType::Tensor(left)),
                    value_type_name(ValueType::Tensor(right)),
                    tensor_dimension_name(left_contracting),
                    tensor_dimension_name(right_contracting),
                ),
            )
        })
}

pub(super) fn infer_matmul(
    left: TensorType,
    right: TensorType,
    span: Span,
) -> Result<TensorType, SemanticError> {
    use severian_hir::TensorIntrinsic::Matmul;
    if left.element != right.element {
        return Err(tensor_error(span, Matmul, "operand dtypes do not match"));
    }
    let (Some(left_rank), Some(right_rank)) = (left.rank, right.rank) else {
        return Ok(TensorType::dynamic(left.element));
    };
    if left_rank < 2 || right_rank < 2 {
        return Err(tensor_error(
            span,
            Matmul,
            "operands must both have rank two or greater",
        ));
    }
    let left_rank = left_rank as usize;
    let right_rank = right_rank as usize;
    let left_contracting = left.dimensions[left_rank - 1];
    let right_contracting = right.dimensions[right_rank - 2];
    if dimensions_conflict(left_contracting, right_contracting) {
        return Err(tensor_error(
            span,
            Matmul,
            format!(
                "contracting dimensions do not match: {} != {} (left `{}`, right `{}`)",
                tensor_dimension_name(left_contracting),
                tensor_dimension_name(right_contracting),
                value_type_name(ValueType::Tensor(left)),
                value_type_name(ValueType::Tensor(right))
            ),
        ));
    }

    let left_batch = TensorType::ranked(left.element, &left.dimensions[..left_rank - 2])
        .expect("input rank was already bounded");
    let right_batch = TensorType::ranked(right.element, &right.dimensions[..right_rank - 2])
        .expect("input rank was already bounded");
    let batch = left_batch
        .broadcast_with(right_batch)
        .map_err(|reason| tensor_error(span, Matmul, format!("batch {reason}")))?;
    let batch_rank = usize::from(batch.rank.expect("ranked batch remains ranked"));
    let mut dimensions = batch.dimensions[..batch_rank].to_vec();
    dimensions.push(left.dimensions[left_rank - 2]);
    dimensions.push(right.dimensions[right_rank - 1]);
    TensorType::ranked(left.element, &dimensions)
        .map_err(|reason| tensor_error(span, Matmul, reason))
}

pub(super) fn infer_transpose(
    arguments: &[Expression],
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    let input = tensor_argument(arguments, 0, operation, span)?;
    let Some(rank) = input.rank else {
        return Ok(input);
    };
    let permutation = match arguments.get(1) {
        Some(argument)
            if matches!(argument.ty(), Some(ValueType::List))
                && !matches!(argument.kind(), Expression::List(_)) =>
        {
            return TensorType::ranked(
                input.element,
                &vec![TensorDimension::Dynamic; usize::from(rank)],
            )
            .map_err(|reason| tensor_error(span, operation, reason));
        }
        Some(argument) => integer_list(argument, operation, span)?
            .into_iter()
            .map(|axis| usize::try_from(axis).ok())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| tensor_error(span, operation, "permutation axis is too large"))?,
        None => (0..usize::from(rank)).rev().collect(),
    };
    if permutation.len() != usize::from(rank) {
        return Err(tensor_error(
            span,
            operation,
            format!("permutation has {} axes for rank {rank}", permutation.len()),
        ));
    }
    let mut seen = vec![false; usize::from(rank)];
    let mut dimensions = Vec::with_capacity(permutation.len());
    for axis in permutation {
        if axis >= usize::from(rank) || seen[axis] {
            return Err(tensor_error(span, operation, "invalid tensor permutation"));
        }
        seen[axis] = true;
        dimensions.push(input.dimensions[axis]);
    }
    TensorType::ranked(input.element, &dimensions)
        .map_err(|reason| tensor_error(span, operation, reason))
}

pub(super) fn reduce_last(
    input: TensorType,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    let Some(rank) = input.rank else {
        return Ok(TensorType::dynamic(input.element));
    };
    if rank == 0 {
        return Err(tensor_error(
            span,
            operation,
            "cannot reduce a rank-zero tensor",
        ));
    }
    TensorType::ranked(input.element, &input.dimensions[..usize::from(rank - 1)])
        .map_err(|reason| tensor_error(span, operation, reason))
}

pub(super) fn validate_reshape(
    input: TensorType,
    result: TensorType,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<(), SemanticError> {
    let static_count = |tensor: TensorType| {
        let rank = usize::from(tensor.rank?);
        tensor.dimensions[..rank]
            .iter()
            .try_fold(1u64, |count, dimension| match dimension {
                TensorDimension::Static(size) => count.checked_mul(*size),
                TensorDimension::Dynamic => None,
            })
    };
    if let (Some(input_count), Some(result_count)) = (static_count(input), static_count(result)) {
        if input_count != result_count {
            return Err(tensor_error(
                span,
                operation,
                format!("reshape changes element count from {input_count} to {result_count}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn infer_gather(
    arguments: &[Expression],
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    let table = tensor_argument(arguments, 0, operation, span)?;
    let indices = tensor_argument(arguments, 1, operation, span)?;
    let (Some(table_rank), Some(index_rank)) = (table.rank, indices.rank) else {
        return Ok(TensorType::dynamic(table.element));
    };
    if table_rank != 2 {
        return Err(tensor_error(
            span,
            operation,
            "embedding table must have rank two",
        ));
    }
    let mut dimensions = indices.dimensions[..usize::from(index_rank)].to_vec();
    dimensions.push(table.dimensions[1]);
    TensorType::ranked(table.element, &dimensions)
        .map_err(|reason| tensor_error(span, operation, reason))
}

pub(super) fn infer_slice(
    arguments: &[Expression],
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    let input = tensor_argument(arguments, 0, operation, span)?;
    let starts = shape_integer_list_argument(arguments, 1, operation, span)?;
    let limits = shape_integer_list_argument(arguments, 2, operation, span)?;
    let strides = shape_integer_list_argument(arguments, 3, operation, span)?;
    if starts.len() != limits.len() || starts.len() != strides.len() {
        return Err(tensor_error(
            span,
            operation,
            "slice metadata lengths do not match",
        ));
    }
    if input
        .rank
        .is_some_and(|rank| usize::from(rank) != starts.len())
    {
        return Err(tensor_error(
            span,
            operation,
            "slice metadata does not match input rank",
        ));
    }
    let dimensions = starts
        .into_iter()
        .zip(limits)
        .zip(strides)
        .enumerate()
        .map(|(axis, ((start, limit), stride))| {
            let Some(stride) = stride else {
                return Ok(TensorDimension::Dynamic);
            };
            if stride <= 0 {
                return Err(tensor_error(
                    span,
                    operation,
                    "slice stride must be positive",
                ));
            }
            let Some(start) = start else {
                return Ok(TensorDimension::Dynamic);
            };
            let limit = match limit {
                Some(-1) => match input.dimensions[axis] {
                    TensorDimension::Static(size) => i64::try_from(size).ok(),
                    TensorDimension::Dynamic => None,
                },
                limit => limit,
            };
            let Some(limit) = limit else {
                return Ok(TensorDimension::Dynamic);
            };
            if start < 0 || limit < start {
                return Err(tensor_error(span, operation, "invalid slice interval"));
            }
            Ok(TensorDimension::Static(
                ((limit - start) as u64).div_ceil(stride as u64),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    TensorType::ranked(input.element, &dimensions)
        .map_err(|reason| tensor_error(span, operation, reason))
}

pub(super) fn infer_concatenate(
    arguments: &[Expression],
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    let Some(Expression::List(values)) = arguments.first().map(Expression::kind) else {
        return Err(tensor_error(span, operation, "expected a tensor list"));
    };
    let mut tensors = values.iter().map(|value| match value.ty() {
        Some(ValueType::Tensor(tensor)) => Ok(tensor),
        _ => Err(tensor_error(
            span,
            operation,
            "list contains a non-tensor value",
        )),
    });
    let mut result = tensors
        .next()
        .ok_or_else(|| tensor_error(span, operation, "cannot concatenate an empty list"))??;
    let Some(rank) = result.rank else {
        return Ok(result);
    };
    let axis = normalized_axis(
        signed_integer_argument(arguments, 1, operation, span)?,
        rank,
        operation,
        span,
    )?;
    let mut axis_size = match result.dimensions[axis] {
        TensorDimension::Static(size) => Some(size),
        TensorDimension::Dynamic => None,
    };
    for tensor in tensors {
        let tensor = tensor?;
        if tensor.element != result.element || tensor.rank != result.rank {
            return Err(tensor_error(
                span,
                operation,
                "tensor list types do not match",
            ));
        }
        for dimension in 0..usize::from(rank) {
            if dimension != axis
                && dimensions_conflict(result.dimensions[dimension], tensor.dimensions[dimension])
            {
                return Err(tensor_error(
                    span,
                    operation,
                    format!("dimension {dimension} does not match"),
                ));
            }
        }
        axis_size = match (axis_size, tensor.dimensions[axis]) {
            (Some(left), TensorDimension::Static(right)) => left.checked_add(right),
            _ => None,
        };
    }
    result.dimensions[axis] = axis_size.map_or(TensorDimension::Dynamic, TensorDimension::Static);
    Ok(result)
}

fn dimensions_conflict(left: TensorDimension, right: TensorDimension) -> bool {
    matches!(
        (left, right),
        (TensorDimension::Static(left), TensorDimension::Static(right)) if left != right
    )
}

fn tensor_dimension_name(dimension: TensorDimension) -> String {
    match dimension {
        TensorDimension::Static(value) => value.to_string(),
        TensorDimension::Dynamic => "dynamic".into(),
    }
}

pub(super) fn tensor_error(
    span: Span,
    operation: severian_hir::TensorIntrinsic,
    reason: impl std::fmt::Display,
) -> SemanticError {
    error(
        span,
        format!("E002402: invalid tensor {}: {reason}", operation.name()),
    )
}
