mod metadata;

pub(crate) use metadata::scalar_tensor;
use metadata::{
    float_argument, integer_argument, integer_list_argument, reduced_suffix_axes, require_arity,
    signed_integer_argument, static_reduction_count,
};

use super::{
    activation, indexing, linear, normalization, reduction, MlirValue, StableHloEmitter,
    StableHloLoweringError,
};
use severian_hir::{
    CallTarget, Expression, TensorDimension, TensorElementType, TensorIntrinsic, TensorType,
};

pub(super) fn lower(
    target: &CallTarget,
    source_args: &[Expression],
    args: &[MlirValue],
    result_type: TensorType,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    let operation = target.tensor_intrinsic().ok_or_else(|| {
        StableHloLoweringError::UnsupportedOperation(target.lowering_symbol().into())
    })?;
    let op = operation.name().to_owned();
    match operation {
        TensorIntrinsic::Gather => {
            require_arity(&op, args, 2)?;
            let table = args[0]
                .tensor_type()
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let ids = args[1]
                .tensor_type()
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let table_rank =
                table
                    .rank
                    .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                        function: target.name.clone(),
                        reason: "gather requires a ranked embedding table".into(),
                    })?;
            let index_rank =
                ids.rank
                    .ok_or_else(|| StableHloLoweringError::UnsupportedFunction {
                        function: target.name.clone(),
                        reason: "gather requires ranked indices".into(),
                    })?;
            if table_rank != 2 {
                return Err(StableHloLoweringError::InvalidRank {
                    operation: op,
                    expected: 2,
                    actual: Some(table_rank),
                });
            }
            let TensorDimension::Static(vocabulary_size) = table.dimensions[0] else {
                return Err(StableHloLoweringError::UnsupportedOperation(
                    target.name.clone(),
                ));
            };
            let TensorDimension::Static(embedding_size) = table.dimensions[1] else {
                return Err(StableHloLoweringError::UnsupportedOperation(
                    target.name.clone(),
                ));
            };
            Ok(indexing::embedding_lookup(
                emitter,
                &args[0],
                &args[1],
                u64::from(index_rank),
                vocabulary_size,
                embedding_size,
                result_type,
            ))
        }
        TensorIntrinsic::Transpose => {
            require_arity(&op, args, 1)?;
            let axes = match source_args.get(1) {
                Some(axes) => integer_list_argument(Some(axes), &target.name)?,
                None => {
                    let rank = result_type
                        .rank
                        .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
                    (0..u64::from(rank)).rev().collect()
                }
            };
            Ok(emitter.transpose(&args[0], &axes, result_type))
        }
        TensorIntrinsic::Reshape => {
            require_arity(&op, args, 1)?;
            let _shape = integer_list_argument(source_args.get(1), &target.name)?;
            Ok(emitter.reshape(&args[0], result_type))
        }
        TensorIntrinsic::Broadcast | TensorIntrinsic::BroadcastLike => {
            let expected = usize::from(operation == TensorIntrinsic::BroadcastLike) + 1;
            require_arity(&op, args, expected)?;
            if operation == TensorIntrinsic::Broadcast {
                let _shape = integer_list_argument(source_args.get(1), &target.name)?;
            }
            let input_rank = args[0]
                .tensor_type()
                .and_then(|tensor| tensor.rank)
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let result_rank = result_type
                .rank
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            if input_rank > result_rank {
                return Err(StableHloLoweringError::InvalidRank {
                    operation: op,
                    expected: usize::from(input_rank),
                    actual: Some(result_rank),
                });
            }
            let input_type = args[0].tensor_type().unwrap();
            let result_rank = usize::from(result_rank);
            let width = usize::from(input_rank);
            let first = (0..=result_rank - width)
                .max_by_key(|&start| {
                    (0..width)
                        .filter(|&axis| {
                            input_type.dimensions[axis] == result_type.dimensions[start + axis]
                        })
                        .count()
                })
                .unwrap_or(result_rank - width);
            let dimensions = (first..first + width)
                .map(|axis| axis as u64)
                .collect::<Vec<_>>();
            Ok(emitter.broadcast_in_dim(&args[0], &dimensions, result_type))
        }
        TensorIntrinsic::Scale | TensorIntrinsic::AddScalar => {
            require_arity(&op, args, 1)?;
            let literal = float_argument(source_args.get(1), &target.name)?;
            let scalar = emitter.splat(&literal, result_type);
            if operation == TensorIntrinsic::Scale {
                Ok(emitter.multiply(&args[0], &scalar, result_type))
            } else {
                Ok(emitter.add(&args[0], &scalar, result_type))
            }
        }
        TensorIntrinsic::DynamicUpdateSlice => {
            require_arity(&op, args, 2)?;
            let starts = integer_list_argument(source_args.get(2), &target.name)?
                .into_iter()
                .map(|start| emitter.scalar(&start.to_string(), TensorElementType::I64))
                .collect::<Vec<_>>();
            Ok(emitter.dynamic_update_slice(&args[0], &args[1], &starts, result_type))
        }
        TensorIntrinsic::DynamicUpdateSliceAxis => {
            require_arity(&op, args, 3)?;
            let axis = integer_argument(source_args.get(3), &target.name)?;
            let rank = args[0]
                .tensor_type()
                .and_then(|tensor| tensor.rank)
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            if axis >= u64::from(rank) {
                return Err(StableHloLoweringError::UnsupportedFunction {
                    function: target.name.clone(),
                    reason: format!("axis {axis} is outside rank {rank}"),
                });
            }
            let scalar_index = emitter.reshape(&args[2], scalar_tensor(TensorElementType::I64));
            let starts = (0..u64::from(rank))
                .map(|dimension| {
                    if dimension == axis {
                        scalar_index.clone()
                    } else {
                        emitter.scalar("0", TensorElementType::I64)
                    }
                })
                .collect::<Vec<_>>();
            Ok(emitter.dynamic_update_slice(&args[0], &args[1], &starts, result_type))
        }
        TensorIntrinsic::DynamicSlice => {
            require_arity(&op, args, 1)?;
            let starts = integer_list_argument(source_args.get(1), &target.name)?
                .into_iter()
                .map(|start| emitter.scalar(&start.to_string(), TensorElementType::I64))
                .collect::<Vec<_>>();
            let sizes = integer_list_argument(source_args.get(2), &target.name)?;
            Ok(emitter.dynamic_slice(&args[0], &starts, &sizes, result_type))
        }
        TensorIntrinsic::Slice => {
            require_arity(&op, args, 1)?;
            let starts = integer_list_argument(source_args.get(1), &target.name)?;
            let limits = integer_list_argument(source_args.get(2), &target.name)?;
            let strides = integer_list_argument(source_args.get(3), &target.name)?;
            Ok(emitter.slice(&args[0], &starts, &limits, &strides, result_type))
        }
        TensorIntrinsic::Cosine => {
            require_arity(&op, args, 1)?;
            Ok(emitter.cosine(&args[0], result_type))
        }
        TensorIntrinsic::Sine => {
            require_arity(&op, args, 1)?;
            Ok(emitter.sine(&args[0], result_type))
        }
        TensorIntrinsic::Concatenate => {
            if args.is_empty() {
                return Err(StableHloLoweringError::InvalidArity {
                    operation: op,
                    expected: 1,
                    actual: 0,
                });
            }
            let axis = integer_argument(source_args.get(1), &target.name)?;
            Ok(emitter.concatenate(args, axis, result_type))
        }
        TensorIntrinsic::SoftmaxAxis => {
            require_arity(&op, args, 1)?;
            let rank = result_type
                .rank
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let requested = signed_integer_argument(source_args.get(1), &target.name)?;
            let axis = if requested < 0 {
                i64::from(rank) + requested
            } else {
                requested
            };
            if axis < 0 || axis >= i64::from(rank) {
                return Err(StableHloLoweringError::UnsupportedFunction {
                    function: target.name.clone(),
                    reason: format!("axis {requested} is outside rank {rank}"),
                });
            }
            let reduced_type = normalization::reduced_axis_type(result_type, axis as u8)?;
            Ok(normalization::softmax_axis(
                emitter,
                &args[0],
                result_type,
                reduced_type,
                axis as u64,
            ))
        }
        TensorIntrinsic::Where => {
            require_arity(&op, args, 3)?;
            Ok(emitter.select(&args[0], &args[1], &args[2], result_type))
        }
        _ => lower_tensor_call(operation, args, result_type, emitter),
    }
}

fn lower_tensor_call(
    operation: TensorIntrinsic,
    args: &[MlirValue],
    result_type: TensorType,
    emitter: &mut StableHloEmitter,
) -> Result<MlirValue, StableHloLoweringError> {
    let op = operation.name().to_owned();

    match operation {
        TensorIntrinsic::Add => {
            require_arity(&op, args, 2)?;
            Ok(emitter.add(&args[0], &args[1], result_type))
        }

        TensorIntrinsic::Subtract => {
            require_arity(&op, args, 2)?;
            Ok(emitter.subtract(&args[0], &args[1], result_type))
        }

        TensorIntrinsic::Multiply => {
            require_arity(&op, args, 2)?;
            Ok(emitter.multiply(&args[0], &args[1], result_type))
        }

        TensorIntrinsic::Divide => {
            require_arity(&op, args, 2)?;
            Ok(emitter.divide(&args[0], &args[1], result_type))
        }

        TensorIntrinsic::Matmul => {
            require_arity(&op, args, 2)?;
            match result_type.rank {
                Some(2) => Ok(linear::matmul_2d(emitter, &args[0], &args[1], result_type)),
                Some(3) => Ok(linear::linear_last_dimension(
                    emitter,
                    &args[0],
                    &args[1],
                    result_type,
                )),
                Some(4) => Ok(linear::batched_matmul(
                    emitter,
                    &args[0],
                    &args[1],
                    result_type,
                )),
                rank => Err(StableHloLoweringError::InvalidRank {
                    operation: op,
                    expected: 2,
                    actual: rank,
                }),
            }
        }

        TensorIntrinsic::Reshape => {
            require_arity(&op, args, 1)?;
            Ok(emitter.reshape(&args[0], result_type))
        }

        TensorIntrinsic::Transpose => {
            require_arity(&op, args, 1)?;
            let rank = result_type
                .rank
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let axes = (0..u64::from(rank)).rev().collect::<Vec<_>>();
            Ok(emitter.transpose(&args[0], &axes, result_type))
        }

        TensorIntrinsic::Broadcast | TensorIntrinsic::BroadcastLike => {
            if args.is_empty() || args.len() > 2 {
                return Err(StableHloLoweringError::InvalidArity {
                    operation: op,
                    expected: 1,
                    actual: args.len(),
                });
            }
            let input_rank = args[0]
                .tensor_type()
                .and_then(|tensor| tensor.rank)
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let result_rank = result_type
                .rank
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            if input_rank > result_rank {
                return Err(StableHloLoweringError::InvalidRank {
                    operation: op,
                    expected: usize::from(input_rank),
                    actual: Some(result_rank),
                });
            }
            let input_type = args[0].tensor_type().unwrap();
            let width = usize::from(input_rank);
            let result_width = usize::from(result_rank);
            let first = (0..=result_width - width)
                .max_by_key(|&start| {
                    (0..width)
                        .filter(|&axis| {
                            input_type.dimensions[axis] == result_type.dimensions[start + axis]
                        })
                        .count()
                })
                .unwrap_or(result_width - width);
            let dimensions = (first..first + width)
                .map(|axis| axis as u64)
                .collect::<Vec<_>>();
            Ok(emitter.broadcast_in_dim(&args[0], &dimensions, result_type))
        }

        TensorIntrinsic::Relu => {
            require_arity(&op, args, 1)?;
            Ok(activation::relu(emitter, &args[0], result_type))
        }

        TensorIntrinsic::Silu => {
            require_arity(&op, args, 1)?;
            Ok(activation::silu(emitter, &args[0], result_type))
        }

        TensorIntrinsic::Exp => {
            require_arity(&op, args, 1)?;
            Ok(emitter.exponential(&args[0], result_type))
        }

        TensorIntrinsic::Tanh => {
            require_arity(&op, args, 1)?;
            Ok(emitter.tanh(&args[0], result_type))
        }

        TensorIntrinsic::Rsqrt => {
            require_arity(&op, args, 1)?;
            Ok(emitter.rsqrt(&args[0], result_type))
        }

        TensorIntrinsic::Sigmoid => {
            require_arity(&op, args, 1)?;
            Ok(emitter.logistic(&args[0], result_type))
        }

        TensorIntrinsic::Sum | TensorIntrinsic::SumLast => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_sum(emitter, &args[0], &axes, result_type))
        }

        TensorIntrinsic::MaxLast => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            Ok(reduction::reduce_max(emitter, &args[0], &axes, result_type))
        }

        TensorIntrinsic::MeanLast => {
            require_arity(&op, args, 1)?;
            let axes = reduced_suffix_axes(&op, &args[0], result_type)?;
            let count = static_reduction_count(&op, &args[0], &axes)?;
            Ok(reduction::mean(
                emitter,
                &args[0],
                &axes,
                result_type,
                count,
            ))
        }

        TensorIntrinsic::Gelu => {
            require_arity(&op, args, 1)?;
            Ok(activation::gelu_tanh(emitter, &args[0], result_type))
        }

        TensorIntrinsic::Softmax => {
            require_arity(&op, args, 1)?;
            let reduced_type = normalization::last_axis_reduced_type(result_type)?;
            Ok(normalization::softmax_last_axis(
                emitter,
                &args[0],
                result_type,
                reduced_type,
            ))
        }

        TensorIntrinsic::LayerNorm => {
            require_arity(&op, args, 3)?;
            let input_type = args[0]
                .tensor_type()
                .ok_or_else(|| StableHloLoweringError::UnsupportedOperation(op.clone()))?;
            let reduced_type = normalization::last_axis_reduced_type(input_type)?;
            let axis = u64::from(input_type.rank.unwrap() - 1);
            let hidden_size = static_reduction_count(&op, &args[0], &[axis])?;
            Ok(normalization::layer_norm(
                emitter,
                &args[0],
                &args[1],
                &args[2],
                input_type,
                reduced_type,
                hidden_size,
                1e-5,
            ))
        }

        TensorIntrinsic::Convert | TensorIntrinsic::ConvertLike => {
            let expected = usize::from(operation == TensorIntrinsic::ConvertLike) + 1;
            require_arity(&op, args, expected)?;
            Ok(emitter.convert(&args[0], result_type))
        }

        TensorIntrinsic::Scale
        | TensorIntrinsic::AddScalar
        | TensorIntrinsic::SoftmaxAxis
        | TensorIntrinsic::Gather
        | TensorIntrinsic::DynamicSlice
        | TensorIntrinsic::DynamicUpdateSlice
        | TensorIntrinsic::DynamicUpdateSliceAxis
        | TensorIntrinsic::Slice
        | TensorIntrinsic::Cosine
        | TensorIntrinsic::Sine
        | TensorIntrinsic::Concatenate
        | TensorIntrinsic::Where => Err(StableHloLoweringError::UnsupportedOperation(op)),
    }
}

pub fn argument(name: impl Into<String>, tensor: TensorType) -> MlirValue {
    MlirValue::from_tensor(name, tensor)
}
