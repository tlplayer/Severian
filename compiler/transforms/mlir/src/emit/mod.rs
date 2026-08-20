use severian_lir::{
    BinaryOperation, Constant, LoweredFloatFormat, LoweredType, Module, Operation, UnaryOperation,
    ValueId,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirError {
    InvalidValue(ValueId),
    UnsupportedType(LoweredType),
    UnsupportedOperation(String),
}

impl fmt::Display for MlirError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MlirError {}

pub fn render(module: &Module) -> Result<String, MlirError> {
    let mut output = String::from("module {\n  func.func @main() {\n");
    for operation in &module.operations {
        match operation {
            Operation::Constant { value, result } => {
                let ty = value_type(module, *result)?;
                let spelling = mlir_type(ty)?;
                let literal = match value {
                    Constant::Integer(value) | Constant::Float(value) => value,
                    Constant::Boolean(true) => "1",
                    Constant::Boolean(false) => "0",
                    other => {
                        return Err(MlirError::UnsupportedOperation(format!(
                            "MLIR constant lowering is unavailable for {other:?}"
                        )))
                    }
                };
                output.push_str(&format!(
                    "    %v{} = arith.constant {literal} : {spelling}\n",
                    result.0
                ));
            }
            Operation::Unary {
                operator,
                operand,
                result,
            } => {
                let ty = mlir_type(value_type(module, *result)?)?;
                return Err(MlirError::UnsupportedOperation(format!(
                    "MLIR unary {operator:?} for %v{} -> %v{} : {ty} requires a dedicated lowering",
                    operand.0, result.0
                )));
            }
            Operation::Binary {
                operator,
                left,
                right,
                result,
            } => {
                let input_type = value_type(module, *left)?;
                let spelling = mlir_type(input_type)?;
                let instruction = mlir_binary(*operator, input_type)?;
                output.push_str(&format!(
                    "    %v{} = {instruction} %v{}, %v{} : {spelling}\n",
                    result.0, left.0, right.0
                ));
            }
        }
    }
    output.push_str("    return\n  }\n}\n");
    Ok(output)
}

fn value_type(module: &Module, id: ValueId) -> Result<LoweredType, MlirError> {
    module
        .values
        .get(id.0 as usize)
        .filter(|value| value.id == id)
        .map(|value| value.ty)
        .ok_or(MlirError::InvalidValue(id))
}

fn mlir_type(ty: LoweredType) -> Result<String, MlirError> {
    Ok(match ty {
        LoweredType::Integer { bits, .. } => format!("i{bits}"),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(16),
        } => "f16".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(32),
        } => "f32".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::Ieee(64),
        } => "f64".into(),
        LoweredType::Float {
            format: LoweredFloatFormat::BrainFloat16,
        } => "bf16".into(),
        LoweredType::Boolean => "i1".into(),
        unsupported => return Err(MlirError::UnsupportedType(unsupported)),
    })
}

fn mlir_binary(operator: BinaryOperation, ty: LoweredType) -> Result<&'static str, MlirError> {
    let float = matches!(ty, LoweredType::Float { .. });
    let signed = matches!(ty, LoweredType::Integer { signed: true, .. });
    Ok(match (operator, float) {
        (BinaryOperation::Add, false) => "arith.addi",
        (BinaryOperation::Subtract, false) => "arith.subi",
        (BinaryOperation::Multiply, false) => "arith.muli",
        (BinaryOperation::Divide, false) if signed => "arith.divsi",
        (BinaryOperation::Divide, false) => "arith.divui",
        (BinaryOperation::Remainder, false) if signed => "arith.remsi",
        (BinaryOperation::Remainder, false) => "arith.remui",
        (BinaryOperation::Add, true) => "arith.addf",
        (BinaryOperation::Subtract, true) => "arith.subf",
        (BinaryOperation::Multiply, true) => "arith.mulf",
        (BinaryOperation::Divide, true) => "arith.divf",
        (BinaryOperation::Remainder, true) => "arith.remf",
        _ => {
            return Err(MlirError::UnsupportedOperation(format!(
                "MLIR binary lowering is unavailable for {operator:?} on {ty:?}"
            )))
        }
    })
}

#[allow(dead_code)]
fn _unary_is_lir(_: UnaryOperation) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfloat_is_not_silently_mapped_to_f32() {
        assert_eq!(
            mlir_type(LoweredType::Float {
                format: LoweredFloatFormat::BrainFloat16
            })
            .unwrap(),
            "bf16"
        );
    }

    #[test]
    fn unsupported_aggregate_is_explicit() {
        assert!(matches!(
            mlir_type(LoweredType::String),
            Err(MlirError::UnsupportedType(LoweredType::String))
        ));
    }
}
