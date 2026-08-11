use severian_hir::TensorType;

use super::{activation::silu, MlirValue, StableHloEmitter};

pub fn matmul_2d(
    emitter: &mut StableHloEmitter,
    lhs: &MlirValue,
    rhs: &MlirValue,
    result_type: TensorType,
) -> MlirValue {
    emitter.dot_general(lhs, rhs, &[], &[], &[1], &[0], result_type)
}

pub fn batched_matmul(
    emitter: &mut StableHloEmitter,
    lhs: &MlirValue,
    rhs: &MlirValue,
    result_type: TensorType,
) -> MlirValue {
    emitter.dot_general(lhs, rhs, &[0, 1], &[0, 1], &[3], &[2], result_type)
}

pub fn linear_last_dimension(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    weight: &MlirValue,
    result_type: TensorType,
) -> MlirValue {
    let rank = u64::from(
        result_type
            .rank
            .expect("linear projection requires a ranked tensor"),
    );
    assert!(
        rank > 0,
        "linear projection requires rank greater than zero"
    );
    emitter.dot_general(input, weight, &[], &[], &[rank - 1], &[0], result_type)
}

pub fn llama_mlp(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    gate_weight: &MlirValue,
    up_weight: &MlirValue,
    down_weight: &MlirValue,
    intermediate_type: TensorType,
    output_type: TensorType,
) -> MlirValue {
    let gate = linear_last_dimension(emitter, input, gate_weight, intermediate_type);
    let up = linear_last_dimension(emitter, input, up_weight, intermediate_type);
    let gate = silu(emitter, &gate, intermediate_type);
    let hidden = emitter.multiply(&gate, &up, intermediate_type);
    linear_last_dimension(emitter, &hidden, down_weight, output_type)
}
