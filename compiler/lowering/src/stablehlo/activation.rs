use severian_hir::{TensorElementType, TensorType};

use super::{MlirValue, StableHloEmitter};

pub fn relu(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    tensor_type: TensorType,
) -> MlirValue {
    let zero = emitter.splat(
        match tensor_type.element {
            TensorElementType::F8E4M3FN
            | TensorElementType::F8E5M2
            | TensorElementType::F16
            | TensorElementType::BF16
            | TensorElementType::F32
            | TensorElementType::F64
            | TensorElementType::C64
            | TensorElementType::C128 => "0.0",
            TensorElementType::Bool => "false",
            TensorElementType::I8
            | TensorElementType::I16
            | TensorElementType::I32
            | TensorElementType::I64
            | TensorElementType::U8
            | TensorElementType::U16
            | TensorElementType::U32
            | TensorElementType::U64 => "0",
        },
        tensor_type,
    );
    emitter.maximum(input, &zero, tensor_type)
}

pub fn silu(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    tensor_type: TensorType,
) -> MlirValue {
    let sigmoid = emitter.logistic(input, tensor_type);
    emitter.multiply(input, &sigmoid, tensor_type)
}

/// GELU's standard tanh approximation, expressed entirely as StableHLO.
pub fn gelu_tanh(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    tensor_type: TensorType,
) -> MlirValue {
    let x2 = emitter.multiply(input, input, tensor_type);
    let x3 = emitter.multiply(&x2, input, tensor_type);
    let cubic_scale = emitter.splat("0.044715", tensor_type);
    let cubic = emitter.multiply(&x3, &cubic_scale, tensor_type);
    let polynomial = emitter.add(input, &cubic, tensor_type);
    let sqrt_two_over_pi = emitter.splat("0.7978845608028654", tensor_type);
    let scaled = emitter.multiply(&polynomial, &sqrt_two_over_pi, tensor_type);
    let tanh = emitter.tanh(&scaled, tensor_type);
    let one = emitter.splat("1.0", tensor_type);
    let shifted = emitter.add(&one, &tanh, tensor_type);
    let half = emitter.splat("0.5", tensor_type);
    let half_x = emitter.multiply(input, &half, tensor_type);
    emitter.multiply(&half_x, &shifted, tensor_type)
}
