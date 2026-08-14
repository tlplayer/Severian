use super::*;
use crate::MirLoweringError;
use severian_hir::TensorIntrinsic;

pub(crate) fn resolve_tensor_op(
    intrinsic: TensorIntrinsic,
    source_arguments: &[severian_hir::Expression],
    inputs: Vec<TensorOperand>,
    result: TensorType,
) -> Result<TensorOp, MirLoweringError> {
    use ElementwiseKind as Elementwise;
    use TensorIntrinsic as Intrinsic;
    let elementwise = |kind, arity| {
        expect_input_count(intrinsic, &inputs, arity)?;
        Ok(TensorOp::Elementwise(ElementwiseOp {
            kind,
            inputs: inputs.clone(),
            result,
        }))
    };
    match intrinsic {
        Intrinsic::Add => elementwise(Elementwise::Add, 2),
        Intrinsic::Subtract => elementwise(Elementwise::Subtract, 2),
        Intrinsic::Multiply => elementwise(Elementwise::Multiply, 2),
        Intrinsic::Divide => elementwise(Elementwise::Divide, 2),
        Intrinsic::Relu => elementwise(Elementwise::Relu, 1),
        Intrinsic::Silu => elementwise(Elementwise::Silu, 1),
        Intrinsic::Exp => elementwise(Elementwise::Exp, 1),
        Intrinsic::Tanh => elementwise(Elementwise::Tanh, 1),
        Intrinsic::Rsqrt => elementwise(Elementwise::Rsqrt, 1),
        Intrinsic::Sigmoid => elementwise(Elementwise::Sigmoid, 1),
        Intrinsic::Gelu => elementwise(Elementwise::Gelu, 1),
        Intrinsic::Cosine => elementwise(Elementwise::Cosine, 1),
        Intrinsic::Sine => elementwise(Elementwise::Sine, 1),
        Intrinsic::Where => elementwise(Elementwise::Where, 3),
        Intrinsic::Matmul => {
            expect_input_count(intrinsic, &inputs, 2)?;
            Ok(TensorOp::Matmul(MatmulOp {
                left: inputs[0],
                right: inputs[1],
                result,
                accumulation: result.element,
            }))
        }
        Intrinsic::Reshape => {
            expect_input_count(intrinsic, &inputs, 1)?;
            Ok(TensorOp::Reshape(ReshapeOp {
                input: inputs[0],
                result,
            }))
        }
        Intrinsic::Transpose => {
            expect_input_count(intrinsic, &inputs, 1)?;
            let input = inputs[0];
            let permutation = match source_arguments.get(1) {
                Some(argument) => integer_list(argument).ok_or_else(|| {
                    failure(
                        intrinsic,
                        "permutation argument must be a list of non-negative integers",
                    )
                })?,
                None => {
                    let rank = input.ty.rank.ok_or_else(|| {
                        failure(
                            intrinsic,
                            "cannot infer a default permutation for an unranked tensor",
                        )
                    })?;
                    (0..u64::from(rank)).rev().collect()
                }
            };
            Ok(TensorOp::Transpose(TransposeOp {
                input,
                permutation,
                result,
            }))
        }
        Intrinsic::Broadcast | Intrinsic::BroadcastLike => {
            let expected = usize::from(intrinsic == Intrinsic::BroadcastLike) + 1;
            expect_input_count(intrinsic, &inputs, expected)?;
            let input = inputs[0];
            let dimensions = broadcast_dimensions(input.ty, result).ok_or_else(|| {
                failure(
                    intrinsic,
                    "input and result ranks cannot define broadcast dimensions",
                )
            })?;
            Ok(TensorOp::Broadcast(BroadcastOp {
                input,
                dimensions,
                result,
            }))
        }
        Intrinsic::Scale | Intrinsic::AddScalar => {
            expect_input_count(intrinsic, &inputs, 1)?;
            let kind = if intrinsic == Intrinsic::Scale {
                ScalarKind::Multiply
            } else {
                ScalarKind::Add
            };
            let value = source_arguments
                .get(1)
                .and_then(scalar_bits)
                .ok_or_else(|| failure(intrinsic, "scalar argument must be a numeric literal"))?;
            Ok(TensorOp::Scalar(ScalarOp {
                kind,
                input: inputs[0],
                value,
                result,
            }))
        }
        Intrinsic::Sum | Intrinsic::SumLast | Intrinsic::MeanLast | Intrinsic::MaxLast => {
            expect_input_count(intrinsic, &inputs, 1)?;
            let input = inputs[0];
            let rank = input
                .ty
                .rank
                .ok_or_else(|| failure(intrinsic, "reduction input must have a resolved rank"))?;
            let (kind, axes) = match intrinsic {
                Intrinsic::Sum => (ReductionKind::Sum, (0..u64::from(rank)).collect()),
                Intrinsic::SumLast => (ReductionKind::Sum, vec![last_axis(intrinsic, rank)?]),
                Intrinsic::MeanLast => (ReductionKind::Mean, vec![last_axis(intrinsic, rank)?]),
                Intrinsic::MaxLast => (ReductionKind::Maximum, vec![last_axis(intrinsic, rank)?]),
                _ => unreachable!(),
            };
            Ok(TensorOp::Reduction(ReductionOp {
                kind,
                input,
                axes,
                result,
            }))
        }
        Intrinsic::Softmax | Intrinsic::SoftmaxAxis => {
            expect_input_count(intrinsic, &inputs, 1)?;
            let input = inputs[0];
            let rank = input.ty.rank.ok_or_else(|| {
                failure(intrinsic, "normalization input must have a resolved rank")
            })?;
            let axis = if intrinsic == Intrinsic::Softmax {
                last_axis(intrinsic, rank)?
            } else {
                let axis = source_arguments
                    .get(1)
                    .and_then(signed_integer)
                    .ok_or_else(|| {
                        failure(intrinsic, "axis argument must be an integer literal")
                    })?;
                normalize_axis(axis, rank).ok_or_else(|| {
                    failure(
                        intrinsic,
                        format!("axis {axis} is outside tensor rank {rank}"),
                    )
                })?
            };
            Ok(TensorOp::Normalization(NormalizationOp {
                kind: NormalizationKind::Softmax,
                input,
                axis,
                epsilon: None,
                result,
            }))
        }
        Intrinsic::LayerNorm => {
            expect_input_count(intrinsic, &inputs, 1)?;
            let input = inputs[0];
            let rank = input.ty.rank.ok_or_else(|| {
                failure(intrinsic, "normalization input must have a resolved rank")
            })?;
            let epsilon = source_arguments
                .get(1)
                .map(|argument| {
                    scalar_bits(argument).ok_or_else(|| {
                        failure(intrinsic, "epsilon argument must be a numeric literal")
                    })
                })
                .transpose()?;
            Ok(TensorOp::Normalization(NormalizationOp {
                kind: NormalizationKind::LayerNorm,
                input,
                axis: last_axis(intrinsic, rank)?,
                epsilon,
                result,
            }))
        }
        Intrinsic::Gather => {
            expect_input_count(intrinsic, &inputs, 2)?;
            Ok(TensorOp::Gather(GatherOp {
                table: inputs[0],
                indices: inputs[1],
                result,
            }))
        }
        Intrinsic::Convert => {
            expect_input_count(intrinsic, &inputs, 1)?;
            Ok(TensorOp::Convert(ConvertOp {
                input: inputs[0],
                result,
            }))
        }
        Intrinsic::Slice => {
            expect_input_count(intrinsic, &inputs, 1)?;
            Ok(TensorOp::Slice(SliceOp {
                input: inputs[0],
                starts: integer_list_argument(intrinsic, source_arguments, 1, "starts")?,
                limits: integer_list_argument(intrinsic, source_arguments, 2, "limits")?,
                strides: integer_list_argument(intrinsic, source_arguments, 3, "strides")?,
                result,
            }))
        }
        Intrinsic::DynamicSlice => {
            expect_input_count(intrinsic, &inputs, 1)?;
            Ok(TensorOp::DynamicSlice(DynamicSliceOp {
                input: inputs[0],
                starts: integer_list_argument(intrinsic, source_arguments, 1, "starts")?,
                sizes: integer_list_argument(intrinsic, source_arguments, 2, "sizes")?,
                result,
            }))
        }
        Intrinsic::DynamicUpdateSlice => {
            expect_input_count(intrinsic, &inputs, 2)?;
            Ok(TensorOp::DynamicUpdateSlice(DynamicUpdateSliceOp {
                input: inputs[0],
                update: inputs[1],
                starts: integer_list_argument(intrinsic, source_arguments, 2, "starts")?,
                dynamic_index: None,
                axis: None,
                result,
            }))
        }
        Intrinsic::DynamicUpdateSliceAxis => {
            expect_input_count(intrinsic, &inputs, 3)?;
            let axis = source_arguments
                .get(3)
                .and_then(unsigned_integer)
                .ok_or_else(|| {
                    failure(
                        intrinsic,
                        "axis argument must be a non-negative integer literal",
                    )
                })?;
            Ok(TensorOp::DynamicUpdateSlice(DynamicUpdateSliceOp {
                input: inputs[0],
                update: inputs[1],
                starts: Vec::new(),
                dynamic_index: Some(inputs[2]),
                axis: Some(axis),
                result,
            }))
        }
        Intrinsic::Concatenate => {
            let values = source_arguments
                .first()
                .and_then(|argument| match argument.kind() {
                    severian_hir::Expression::List(values) => Some(values),
                    _ => None,
                })
                .ok_or_else(|| failure(intrinsic, "values argument must be a list of tensors"))?;
            if values.is_empty() {
                return Err(failure(intrinsic, "requires at least one tensor input"));
            }
            if inputs.len() != values.len() {
                return Err(failure(
                    intrinsic,
                    format!(
                        "values list contains {} item(s), but only {} have resolved tensor types",
                        values.len(),
                        inputs.len()
                    ),
                ));
            }
            let axis = source_arguments
                .get(1)
                .and_then(unsigned_integer)
                .ok_or_else(|| {
                    failure(
                        intrinsic,
                        "axis argument must be a non-negative integer literal",
                    )
                })?;
            Ok(TensorOp::Concatenate(ConcatenateOp {
                inputs,
                axis,
                result,
            }))
        }
    }
}

fn failure(intrinsic: TensorIntrinsic, message: impl Into<String>) -> MirLoweringError {
    MirLoweringError::tensor(intrinsic, message)
}

fn expect_input_count(
    intrinsic: TensorIntrinsic,
    inputs: &[TensorOperand],
    expected: usize,
) -> Result<(), MirLoweringError> {
    if inputs.len() == expected {
        Ok(())
    } else {
        Err(failure(
            intrinsic,
            format!(
                "expected {expected} tensor operand(s), but resolved {}",
                inputs.len()
            ),
        ))
    }
}

fn integer_list_argument(
    intrinsic: TensorIntrinsic,
    arguments: &[severian_hir::Expression],
    index: usize,
    name: &str,
) -> Result<Vec<u64>, MirLoweringError> {
    arguments.get(index).and_then(integer_list).ok_or_else(|| {
        failure(
            intrinsic,
            format!("`{name}` argument must be a list of non-negative integers"),
        )
    })
}

fn last_axis(intrinsic: TensorIntrinsic, rank: u8) -> Result<u64, MirLoweringError> {
    rank.checked_sub(1).map(u64::from).ok_or_else(|| {
        failure(
            intrinsic,
            "operation requires a tensor with rank at least one",
        )
    })
}

fn tensor_operand(value: ValueRef) -> Option<TensorOperand> {
    let severian_hir::ValueType::Tensor(ty) = value.ty? else {
        return None;
    };
    Some(TensorOperand { value, ty })
}

pub(crate) fn tensor_operands(
    arguments: &[severian_hir::Expression],
    mut lower: impl FnMut(&severian_hir::Expression) -> Result<ValueRef, MirLoweringError>,
) -> Result<Vec<TensorOperand>, MirLoweringError> {
    let mut inputs = Vec::new();
    for argument in arguments {
        if let Some(input) = tensor_operand(lower(argument)?) {
            inputs.push(input);
            continue;
        }
        if let severian_hir::Expression::List(values) = argument.kind() {
            inputs.extend(
                values
                    .iter()
                    .map(|value| lower(value).map(tensor_operand))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten(),
            );
        }
    }
    Ok(inputs)
}

fn integer_list(expression: &severian_hir::Expression) -> Option<Vec<u64>> {
    let severian_hir::Expression::List(values) = expression.kind() else {
        return None;
    };
    values.iter().map(unsigned_integer).collect()
}

fn unsigned_integer(expression: &severian_hir::Expression) -> Option<u64> {
    signed_integer(expression).and_then(|value| u64::try_from(value).ok())
}

fn signed_integer(expression: &severian_hir::Expression) -> Option<i64> {
    match expression.kind() {
        severian_hir::Expression::Integer(value) => Some(*value),
        severian_hir::Expression::Unary {
            op: severian_hir::UnaryOp::Negate,
            expression,
        } => match expression.kind() {
            severian_hir::Expression::Integer(value) => value.checked_neg(),
            _ => None,
        },
        _ => None,
    }
}

fn scalar_bits(expression: &severian_hir::Expression) -> Option<u64> {
    match expression.kind() {
        severian_hir::Expression::Float(bits) => Some(*bits),
        severian_hir::Expression::Integer(value) => Some((*value as f64).to_bits()),
        severian_hir::Expression::Unary {
            op: severian_hir::UnaryOp::Negate,
            expression,
        } => match expression.kind() {
            severian_hir::Expression::Float(bits) => Some((-f64::from_bits(*bits)).to_bits()),
            severian_hir::Expression::Integer(value) => {
                value.checked_neg().map(|value| (value as f64).to_bits())
            }
            _ => None,
        },
        _ => None,
    }
}

fn normalize_axis(axis: i64, rank: u8) -> Option<u64> {
    let axis = if axis < 0 {
        i64::from(rank) + axis
    } else {
        axis
    };
    (0..i64::from(rank)).contains(&axis).then_some(axis as u64)
}

fn broadcast_dimensions(input: TensorType, result: TensorType) -> Option<Vec<u64>> {
    let input_rank = usize::from(input.rank?);
    let result_rank = usize::from(result.rank?);
    if input_rank > result_rank {
        return None;
    }
    let first = (0..=result_rank - input_rank)
        .max_by_key(|&start| {
            (0..input_rank)
                .filter(|&axis| input.dimensions[axis] == result.dimensions[start + axis])
                .count()
        })
        .unwrap_or(result_rank - input_rank);
    Some(
        (first..first + input_rank)
            .map(|axis| axis as u64)
            .collect(),
    )
}
