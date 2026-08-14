use super::*;

mod metadata;
mod shape;

use metadata::{tensor_from_shape, validate_axis};
pub(super) use shape::infer_matmul_operator;
use shape::{
    broadcast, infer_concatenate, infer_gather, infer_matmul, infer_slice, infer_transpose,
    reduce_last, require_broadcast_like_to, require_broadcast_to, tensor_error, validate_reshape,
};

/// Resolve tensor result facts while source spans and compile-time metadata are
/// still available. Backends consume these facts; they do not re-infer them.
pub(super) fn infer_call_result(
    target: &CallTarget,
    arguments: &[Expression],
    declared: ValueType,
    span: Span,
) -> Result<ValueType, SemanticError> {
    let Some(intrinsic) = target.tensor_intrinsic() else {
        return Ok(declared);
    };
    if arguments
        .iter()
        .any(|argument| matches!(argument.ty(), Some(ValueType::Any | ValueType::TensorAny)))
    {
        return Ok(declared);
    }
    use severian_hir::TensorIntrinsic as Op;
    let result = match intrinsic {
        Op::Add | Op::Subtract | Op::Multiply | Op::Divide => {
            broadcast(binary_tensor(arguments, intrinsic, span)?, intrinsic, span)?
        }
        Op::Matmul => {
            let (left, right) = binary_tensor(arguments, intrinsic, span)?;
            infer_matmul(left, right, span)?
        }
        Op::Reshape => {
            let input = tensor_argument(arguments, 0, intrinsic, span)?;
            let result = tensor_from_shape(input.element, arguments.get(1), intrinsic, span)?;
            validate_reshape(input, result, intrinsic, span)?;
            result
        }
        Op::Transpose => infer_transpose(arguments, intrinsic, span)?,
        Op::Broadcast => {
            let input = tensor_argument(arguments, 0, intrinsic, span)?;
            let result = tensor_from_shape(input.element, arguments.get(1), intrinsic, span)?;
            require_broadcast_to(input, result, intrinsic, span)?;
            result
        }
        Op::BroadcastLike => {
            let input = tensor_argument(arguments, 0, intrinsic, span)?;
            let result = tensor_argument(arguments, 1, intrinsic, span)?;
            require_broadcast_like_to(input, result, intrinsic, span)?;
            result
        }
        Op::Convert | Op::ConvertLike => {
            let input = tensor_argument(arguments, 0, intrinsic, span)?;
            let ValueType::Tensor(declared) = declared else {
                return Err(tensor_error(
                    span,
                    intrinsic,
                    "conversion result has no dtype",
                ));
            };
            TensorType {
                element: declared.element,
                rank: input.rank,
                dimensions: input.dimensions,
            }
        }
        Op::Sum => {
            let input = tensor_argument(arguments, 0, intrinsic, span)?;
            TensorType::ranked(input.element, &[]).expect("rank-zero tensor is representable")
        }
        Op::SumLast | Op::MeanLast | Op::MaxLast => reduce_last(
            tensor_argument(arguments, 0, intrinsic, span)?,
            intrinsic,
            span,
        )?,
        Op::Gather => infer_gather(arguments, intrinsic, span)?,
        Op::DynamicSlice => {
            let input = tensor_argument(arguments, 0, intrinsic, span)?;
            tensor_from_shape(input.element, arguments.get(2), intrinsic, span)?
        }
        Op::Slice => infer_slice(arguments, intrinsic, span)?,
        Op::Concatenate => infer_concatenate(arguments, intrinsic, span)?,
        Op::Where => {
            let on_true = tensor_argument(arguments, 1, intrinsic, span)?;
            let on_false = tensor_argument(arguments, 2, intrinsic, span)?;
            let result = broadcast((on_true, on_false), intrinsic, span)?;
            let condition = tensor_argument(arguments, 0, intrinsic, span)?;
            require_broadcast_to(condition, result, intrinsic, span)?;
            result
        }
        Op::SoftmaxAxis => {
            let input = tensor_argument(arguments, 0, intrinsic, span)?;
            validate_axis(input, arguments.get(1), intrinsic, span)?;
            input
        }
        Op::Scale
        | Op::AddScalar
        | Op::Relu
        | Op::Silu
        | Op::Exp
        | Op::Tanh
        | Op::Rsqrt
        | Op::Sigmoid
        | Op::Gelu
        | Op::Softmax
        | Op::LayerNorm
        | Op::DynamicUpdateSlice
        | Op::DynamicUpdateSliceAxis
        | Op::Cosine
        | Op::Sine => tensor_argument(arguments, 0, intrinsic, span)?,
    };
    Ok(ValueType::Tensor(result))
}

fn tensor_argument(
    arguments: &[Expression],
    index: usize,
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<TensorType, SemanticError> {
    match arguments.get(index).and_then(Expression::ty) {
        Some(ValueType::Tensor(tensor)) => Ok(tensor),
        _ => Err(tensor_error(
            span,
            operation,
            format!("argument {} is not a resolved tensor", index + 1),
        )),
    }
}

fn binary_tensor(
    arguments: &[Expression],
    operation: severian_hir::TensorIntrinsic,
    span: Span,
) -> Result<(TensorType, TensorType), SemanticError> {
    Ok((
        tensor_argument(arguments, 0, operation, span)?,
        tensor_argument(arguments, 1, operation, span)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(element: TensorElementType, dimensions: &[u64]) -> TensorType {
        TensorType::ranked(
            element,
            &dimensions
                .iter()
                .copied()
                .map(TensorDimension::Static)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn matmul_resolves_batch_and_matrix_dimensions() {
        let result = infer_matmul(
            tensor(TensorElementType::F32, &[2, 4, 16]),
            tensor(TensorElementType::F32, &[16, 8]),
            Span::dummy(),
        )
        .unwrap();
        assert_eq!(result, tensor(TensorElementType::F32, &[2, 4, 8]));
    }

    #[test]
    fn incompatible_matmul_is_an_error() {
        let error = infer_matmul(
            tensor(TensorElementType::F32, &[4, 16]),
            tensor(TensorElementType::F32, &[32, 8]),
            Span::dummy(),
        )
        .unwrap_err();
        assert!(error.message.contains("16 != 32"));
    }
}
