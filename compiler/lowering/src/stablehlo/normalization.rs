use severian_hir::{TensorDimension, TensorType};

use super::{
    reduction::mean, MlirValue, StableHloEmitter, StableHloLoweringError, StableHloReduction,
};

pub fn last_axis_reduced_type(
    input_type: TensorType,
) -> Result<TensorType, StableHloLoweringError> {
    let rank = input_type
        .rank
        .ok_or_else(|| StableHloLoweringError::InvalidRank {
            operation: "last-axis reduction".into(),
            expected: 1,
            actual: None,
        })?;
    if rank == 0 {
        return Err(StableHloLoweringError::InvalidRank {
            operation: "last-axis reduction".into(),
            expected: 1,
            actual: Some(0),
        });
    }
    let mut dimensions = input_type.dimensions;
    dimensions[usize::from(rank - 1)] = TensorDimension::Dynamic;
    Ok(TensorType {
        element: input_type.element,
        rank: Some(rank - 1),
        dimensions,
    })
}

pub fn softmax_last_axis(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    input_type: TensorType,
    reduced_type: TensorType,
) -> MlirValue {
    let rank = u64::from(input_type.rank.expect("softmax requires a ranked tensor"));
    assert!(rank > 0, "softmax requires rank greater than zero");
    let axis = rank - 1;
    let negative_infinity = emitter.scalar("-inf", input_type.element);
    let maximum = emitter.reduce(
        input,
        &negative_infinity,
        &[axis],
        StableHloReduction::Maximum,
        reduced_type,
    );
    let broadcast_dimensions = (0..axis).collect::<Vec<_>>();
    let maximum = emitter.broadcast_in_dim(&maximum, &broadcast_dimensions, input_type);
    let shifted = emitter.subtract(input, &maximum, input_type);
    let exponentials = emitter.exponential(&shifted, input_type);
    let zero = emitter.scalar("0.0", input_type.element);
    let denominator = emitter.reduce(
        &exponentials,
        &zero,
        &[axis],
        StableHloReduction::Add,
        reduced_type,
    );
    let denominator = emitter.broadcast_in_dim(&denominator, &broadcast_dimensions, input_type);
    emitter.divide(&exponentials, &denominator, input_type)
}

pub fn rms_norm(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    weight: &MlirValue,
    input_type: TensorType,
    reduced_type: TensorType,
    hidden_size: u64,
    epsilon: f64,
) -> MlirValue {
    let rank = u64::from(input_type.rank.expect("RMSNorm requires a ranked tensor"));
    assert!(rank > 0, "RMSNorm requires rank greater than zero");
    let square = emitter.multiply(input, input, input_type);
    let mean_square = mean(emitter, &square, &[rank - 1], reduced_type, hidden_size);
    let epsilon = emitter.splat(&epsilon.to_string(), reduced_type);
    let stabilized = emitter.add(&mean_square, &epsilon, reduced_type);
    let inverse_rms = emitter.rsqrt(&stabilized, reduced_type);
    let outer_dimensions = (0..rank - 1).collect::<Vec<_>>();
    let inverse_rms = emitter.broadcast_in_dim(&inverse_rms, &outer_dimensions, input_type);
    let normalized = emitter.multiply(input, &inverse_rms, input_type);
    let weight = emitter.broadcast_in_dim(weight, &[rank - 1], input_type);
    emitter.multiply(&normalized, &weight, input_type)
}

pub fn layer_norm(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    weight: &MlirValue,
    bias: &MlirValue,
    input_type: TensorType,
    reduced_type: TensorType,
    hidden_size: u64,
    epsilon: f64,
) -> MlirValue {
    let rank = u64::from(input_type.rank.expect("LayerNorm requires a ranked tensor"));
    assert!(rank > 0, "LayerNorm requires rank greater than zero");
    let outer_dimensions = (0..rank - 1).collect::<Vec<_>>();
    let mean_value = mean(emitter, input, &[rank - 1], reduced_type, hidden_size);
    let mean_broadcast = emitter.broadcast_in_dim(&mean_value, &outer_dimensions, input_type);
    let centered = emitter.subtract(input, &mean_broadcast, input_type);
    let square = emitter.multiply(&centered, &centered, input_type);
    let variance = mean(emitter, &square, &[rank - 1], reduced_type, hidden_size);
    let epsilon = emitter.splat(&epsilon.to_string(), reduced_type);
    let variance = emitter.add(&variance, &epsilon, reduced_type);
    let inverse_stddev = emitter.rsqrt(&variance, reduced_type);
    let inverse_stddev = emitter.broadcast_in_dim(&inverse_stddev, &outer_dimensions, input_type);
    let normalized = emitter.multiply(&centered, &inverse_stddev, input_type);
    let weight = emitter.broadcast_in_dim(weight, &[rank - 1], input_type);
    let bias = emitter.broadcast_in_dim(bias, &[rank - 1], input_type);
    let scaled = emitter.multiply(&normalized, &weight, input_type);
    emitter.add(&scaled, &bias, input_type)
}
