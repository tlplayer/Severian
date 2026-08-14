use super::*;
use severian_hir::TensorIntrinsic;

pub(crate) fn resolve_tensor_op(
    intrinsic: TensorIntrinsic,
    source_arguments: &[severian_hir::Expression],
    inputs: Vec<TensorOperand>,
    result: TensorType,
) -> Option<TensorOp> {
    use ElementwiseKind as Elementwise;
    use TensorIntrinsic as Intrinsic;
    let elementwise = |kind, arity| {
        (inputs.len() == arity).then(|| {
            TensorOp::Elementwise(ElementwiseOp {
                kind,
                inputs: inputs.clone(),
                result,
            })
        })
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
        Intrinsic::Matmul => Some(TensorOp::Matmul(MatmulOp {
            left: *inputs.first()?,
            right: *inputs.get(1)?,
            result,
            accumulation: result.element,
        })),
        Intrinsic::Reshape => Some(TensorOp::Reshape(ReshapeOp {
            input: *inputs.first()?,
            result,
        })),
        Intrinsic::Transpose => {
            let input = *inputs.first()?;
            let permutation = source_arguments.get(1).and_then(integer_list).or_else(|| {
                input
                    .ty
                    .rank
                    .map(|rank| (0..u64::from(rank)).rev().collect())
            })?;
            Some(TensorOp::Transpose(TransposeOp {
                input,
                permutation,
                result,
            }))
        }
        Intrinsic::Broadcast | Intrinsic::BroadcastLike => {
            let input = *inputs.first()?;
            Some(TensorOp::Broadcast(BroadcastOp {
                input,
                dimensions: broadcast_dimensions(input.ty, result)?,
                result,
            }))
        }
        Intrinsic::Scale | Intrinsic::AddScalar => {
            let kind = if intrinsic == Intrinsic::Scale {
                ScalarKind::Multiply
            } else {
                ScalarKind::Add
            };
            Some(TensorOp::Scalar(ScalarOp {
                kind,
                input: *inputs.first()?,
                value: scalar_bits(source_arguments.get(1)?)?,
                result,
            }))
        }
        Intrinsic::Sum | Intrinsic::SumLast | Intrinsic::MeanLast | Intrinsic::MaxLast => {
            let input = *inputs.first()?;
            let rank = input.ty.rank?;
            let (kind, axes) = match intrinsic {
                Intrinsic::Sum => (ReductionKind::Sum, (0..u64::from(rank)).collect()),
                Intrinsic::SumLast => (ReductionKind::Sum, vec![u64::from(rank.checked_sub(1)?)]),
                Intrinsic::MeanLast => (ReductionKind::Mean, vec![u64::from(rank.checked_sub(1)?)]),
                Intrinsic::MaxLast => (
                    ReductionKind::Maximum,
                    vec![u64::from(rank.checked_sub(1)?)],
                ),
                _ => unreachable!(),
            };
            Some(TensorOp::Reduction(ReductionOp {
                kind,
                input,
                axes,
                result,
            }))
        }
        Intrinsic::Softmax | Intrinsic::SoftmaxAxis => {
            let input = *inputs.first()?;
            let rank = input.ty.rank?;
            let axis = if intrinsic == Intrinsic::Softmax {
                u64::from(rank.checked_sub(1)?)
            } else {
                normalize_axis(signed_integer(source_arguments.get(1)?)?, rank)?
            };
            Some(TensorOp::Normalization(NormalizationOp {
                kind: NormalizationKind::Softmax,
                input,
                axis,
                epsilon: None,
                result,
            }))
        }
        Intrinsic::LayerNorm => Some(TensorOp::Normalization(NormalizationOp {
            kind: NormalizationKind::LayerNorm,
            input: *inputs.first()?,
            axis: u64::from(inputs.first()?.ty.rank?.checked_sub(1)?),
            epsilon: source_arguments.get(1).and_then(scalar_bits),
            result,
        })),
        Intrinsic::Gather => Some(TensorOp::Gather(GatherOp {
            table: *inputs.first()?,
            indices: *inputs.get(1)?,
            result,
        })),
        Intrinsic::Convert => Some(TensorOp::Convert(ConvertOp {
            input: *inputs.first()?,
            result,
        })),
        Intrinsic::Slice => Some(TensorOp::Slice(SliceOp {
            input: *inputs.first()?,
            starts: integer_list(source_arguments.get(1)?)?,
            limits: integer_list(source_arguments.get(2)?)?,
            strides: integer_list(source_arguments.get(3)?)?,
            result,
        })),
        Intrinsic::DynamicSlice => Some(TensorOp::DynamicSlice(DynamicSliceOp {
            input: *inputs.first()?,
            starts: integer_list(source_arguments.get(1)?)?,
            sizes: integer_list(source_arguments.get(2)?)?,
            result,
        })),
        Intrinsic::DynamicUpdateSlice => Some(TensorOp::DynamicUpdateSlice(DynamicUpdateSliceOp {
            input: *inputs.first()?,
            update: *inputs.get(1)?,
            starts: integer_list(source_arguments.get(2)?)?,
            dynamic_index: None,
            axis: None,
            result,
        })),
        Intrinsic::DynamicUpdateSliceAxis => {
            Some(TensorOp::DynamicUpdateSlice(DynamicUpdateSliceOp {
                input: *inputs.first()?,
                update: *inputs.get(1)?,
                starts: Vec::new(),
                dynamic_index: Some(*inputs.get(2)?),
                axis: unsigned_integer(source_arguments.get(3)?),
                result,
            }))
        }
        Intrinsic::Concatenate => Some(TensorOp::Concatenate(ConcatenateOp {
            inputs,
            axis: unsigned_integer(source_arguments.get(1)?)?,
            result,
        })),
    }
}

fn tensor_operand(value: ValueRef) -> Option<TensorOperand> {
    let severian_hir::ValueType::Tensor(ty) = value.ty? else {
        return None;
    };
    Some(TensorOperand { value, ty })
}

pub(crate) fn tensor_operands(
    arguments: &[severian_hir::Expression],
    mut lower: impl FnMut(&severian_hir::Expression) -> ValueRef,
) -> Vec<TensorOperand> {
    let mut inputs = Vec::new();
    for argument in arguments {
        if let Some(input) = tensor_operand(lower(argument)) {
            inputs.push(input);
            continue;
        }
        if let severian_hir::Expression::List(values) = argument.kind() {
            inputs.extend(
                values
                    .iter()
                    .filter_map(|value| tensor_operand(lower(value))),
            );
        }
    }
    inputs
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
