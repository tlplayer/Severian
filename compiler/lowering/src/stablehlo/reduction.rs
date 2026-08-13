use severian_hir::{TensorElementType, TensorType};

use super::{ops::list, scalar_tensor, MlirValue, StableHloEmitter};
use crate::tensor::tensor_type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableHloReduction {
    Add,
    Maximum,
    Minimum,
}

impl StableHloEmitter {
    pub fn reduce(
        &mut self,
        input: &MlirValue,
        initial: &MlirValue,
        dimensions: &[u64],
        reduction: StableHloReduction,
        result_type: TensorType,
    ) -> MlirValue {
        let result = self.fresh();
        let result_ty = tensor_type(result_type);
        let operation = match reduction {
            StableHloReduction::Add => "add",
            StableHloReduction::Maximum => "maximum",
            StableHloReduction::Minimum => "minimum",
        };
        let scalar_ty = tensor_type(scalar_tensor(result_type.element));
        self.line(format!(
            "{result} = \"stablehlo.reduce\"({}, {}) ({{\n      ^bb0(%left: {scalar_ty}, %right: {scalar_ty}):\n        %combined = \"stablehlo.{operation}\"(%left, %right) : ({scalar_ty}, {scalar_ty}) -> {scalar_ty}\n        \"stablehlo.return\"(%combined) : ({scalar_ty}) -> ()\n    }}) {{dimensions = array<i64: {}>}} : ({}, {}) -> {result_ty}",
            input.name,
            initial.name,
            list(dimensions),
            input.ty,
            initial.ty,
        ));
        MlirValue::from_tensor(result, result_type)
    }
}

pub fn reduce_sum(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    dimensions: &[u64],
    result_type: TensorType,
) -> MlirValue {
    let zero = emitter.scalar(zero_literal(result_type.element), result_type.element);
    emitter.reduce(
        input,
        &zero,
        dimensions,
        StableHloReduction::Add,
        result_type,
    )
}

pub fn reduce_max(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    dimensions: &[u64],
    result_type: TensorType,
) -> MlirValue {
    let initial = emitter.scalar(minimum_literal(result_type.element), result_type.element);
    emitter.reduce(
        input,
        &initial,
        dimensions,
        StableHloReduction::Maximum,
        result_type,
    )
}

pub fn reduce_min(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    dimensions: &[u64],
    result_type: TensorType,
) -> MlirValue {
    let initial = emitter.scalar(maximum_literal(result_type.element), result_type.element);
    emitter.reduce(
        input,
        &initial,
        dimensions,
        StableHloReduction::Minimum,
        result_type,
    )
}

fn zero_literal(element: TensorElementType) -> &'static str {
    match element {
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
    }
}

fn minimum_literal(element: TensorElementType) -> &'static str {
    match element {
        TensorElementType::Bool => "false",
        TensorElementType::I8 => "-128",
        TensorElementType::I16 => "-32768",
        TensorElementType::I32 => "-2147483648",
        TensorElementType::I64 => "-9223372036854775808",
        TensorElementType::U8
        | TensorElementType::U16
        | TensorElementType::U32
        | TensorElementType::U64 => "0",
        TensorElementType::F8E4M3FN => "0xFE",
        TensorElementType::F8E5M2 => "0xFC",
        TensorElementType::F16 => "0xFC00",
        TensorElementType::BF16 => "0xFF80",
        TensorElementType::F32 => "0xFF800000",
        TensorElementType::F64 => "0xFFF0000000000000",
        TensorElementType::C64 | TensorElementType::C128 => "0.0",
    }
}

fn maximum_literal(element: TensorElementType) -> &'static str {
    match element {
        TensorElementType::Bool => "true",
        TensorElementType::I8 => "127",
        TensorElementType::I16 => "32767",
        TensorElementType::I32 => "2147483647",
        TensorElementType::I64 => "9223372036854775807",
        TensorElementType::U8 => "255",
        TensorElementType::U16 => "65535",
        TensorElementType::U32 => "4294967295",
        TensorElementType::U64 => "18446744073709551615",
        TensorElementType::F8E4M3FN => "0x7E",
        TensorElementType::F8E5M2 => "0x7C",
        TensorElementType::F16 => "0x7C00",
        TensorElementType::BF16 => "0x7F80",
        TensorElementType::F32 => "0x7F800000",
        TensorElementType::F64 => "0x7FF0000000000000",
        TensorElementType::C64 | TensorElementType::C128 => "0.0",
    }
}

pub fn mean(
    emitter: &mut StableHloEmitter,
    input: &MlirValue,
    axes: &[u64],
    reduced_type: TensorType,
    element_count: u64,
) -> MlirValue {
    let sum = reduce_sum(emitter, input, axes, reduced_type);
    let denominator = emitter.splat(&format!("{element_count}.0"), reduced_type);
    emitter.divide(&sum, &denominator, reduced_type)
}
