#![forbid(unsafe_code)]

use severian_lir::{
    BinaryOperation, Constant, LoweredFloatFormat, LoweredType, Module as LirModule,
    Operation as LirOperation, UnaryOperation, Value, ValueId,
};
use severian_mir::{Module as MirModule, Operation as MirOperation};
use severian_universal::{
    BinaryOperator, FloatFormat, IntegerWidth, LiteralValue, PrimitiveRepresentation, TargetSpec,
    TypeContext, TypeId, UnaryOperator,
};
use std::fmt;

pub fn lower(
    mir: &MirModule,
    types: &TypeContext,
    target: &TargetSpec,
) -> Result<LirModule, LoweringError> {
    let values = mir
        .values
        .iter()
        .map(|value| {
            Ok(Value {
                id: ValueId(value.id.0),
                ty: lower_type(value.type_id, types, target)?,
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let operations = mir
        .operations
        .iter()
        .map(|operation| match operation {
            MirOperation::Constant { value, result } => LirOperation::Constant {
                value: lower_constant(value),
                result: ValueId(result.0),
            },
            MirOperation::Unary {
                operator,
                operand,
                result,
            } => LirOperation::Unary {
                operator: lower_unary(*operator),
                operand: ValueId(operand.0),
                result: ValueId(result.0),
            },
            MirOperation::Binary {
                operator,
                left,
                right,
                result,
            } => LirOperation::Binary {
                operator: lower_binary(*operator),
                left: ValueId(left.0),
                right: ValueId(right.0),
                result: ValueId(result.0),
            },
        })
        .collect();
    Ok(LirModule {
        values,
        operations,
        last_binding: mir.bindings.last().map(|(_, value)| ValueId(value.0)),
    })
}

fn lower_constant(value: &LiteralValue) -> Constant {
    match value {
        LiteralValue::Integer(value) => Constant::Integer(value.clone()),
        LiteralValue::Float(value) => Constant::Float(value.clone()),
        LiteralValue::Boolean(value) => Constant::Boolean(*value),
        LiteralValue::String(value) => Constant::String(value.clone()),
        LiteralValue::Bytes(value) => Constant::Bytes(value.clone()),
        LiteralValue::None => Constant::None,
        LiteralValue::Unit => Constant::Unit,
    }
}

fn lower_unary(operator: UnaryOperator) -> UnaryOperation {
    match operator {
        UnaryOperator::Positive => UnaryOperation::Positive,
        UnaryOperator::Negative => UnaryOperation::Negative,
        UnaryOperator::Not => UnaryOperation::Not,
    }
}

fn lower_binary(operator: BinaryOperator) -> BinaryOperation {
    match operator {
        BinaryOperator::Add => BinaryOperation::Add,
        BinaryOperator::Subtract => BinaryOperation::Subtract,
        BinaryOperator::Multiply => BinaryOperation::Multiply,
        BinaryOperator::Divide => BinaryOperation::Divide,
        BinaryOperator::Remainder => BinaryOperation::Remainder,
        BinaryOperator::Power => BinaryOperation::Power,
        BinaryOperator::Equal => BinaryOperation::Equal,
        BinaryOperator::NotEqual => BinaryOperation::NotEqual,
        BinaryOperator::Less => BinaryOperation::Less,
        BinaryOperator::LessEqual => BinaryOperation::LessEqual,
        BinaryOperator::Greater => BinaryOperation::Greater,
        BinaryOperator::GreaterEqual => BinaryOperation::GreaterEqual,
        BinaryOperator::And => BinaryOperation::And,
        BinaryOperator::Or => BinaryOperation::Or,
    }
}

fn lower_type(
    id: TypeId,
    types: &TypeContext,
    target: &TargetSpec,
) -> Result<LoweredType, LoweringError> {
    let primitive = types.primitive(id).ok_or(LoweringError::NotPrimitive(id))?;
    Ok(match primitive.representation {
        PrimitiveRepresentation::Integer { bits, signed } => LoweredType::Integer {
            bits: match bits {
                IntegerWidth::Fixed(bits) => bits,
                IntegerWidth::Machine => target.layout.machine_integer_bits,
            },
            signed,
        },
        PrimitiveRepresentation::PointerInteger { signed } => LoweredType::Integer {
            bits: target.layout.pointer_bits,
            signed,
        },
        PrimitiveRepresentation::Float { format } => LoweredType::Float {
            format: match format {
                FloatFormat::Ieee(bits) => LoweredFloatFormat::Ieee(bits),
                FloatFormat::BrainFloat16 => LoweredFloatFormat::BrainFloat16,
                FloatFormat::Machine => LoweredFloatFormat::Ieee(target.layout.machine_float_bits),
            },
        },
        PrimitiveRepresentation::Boolean => LoweredType::Boolean,
        PrimitiveRepresentation::String => LoweredType::String,
        PrimitiveRepresentation::Bytes => LoweredType::Bytes,
        PrimitiveRepresentation::None => LoweredType::None,
        PrimitiveRepresentation::Unit => LoweredType::Unit,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    NotPrimitive(TypeId),
}

impl fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LoweringError {}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_mir::{Module, Value as MirValue, ValueId as MirValueId};
    use severian_universal::{PrimitiveCategory, UniversalContext};

    fn pointer_context(pointer_bits: u16) -> (UniversalContext, TypeId) {
        let mut types = TypeContext::new();
        let id = types.register_declaration("core.usize", "usize").unwrap();
        types
            .define_primitive(
                id,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::PointerInteger { signed: false },
                false,
            )
            .unwrap();
        let target = TargetSpec {
            name: "test".into(),
            layout: severian_universal::TargetLayout {
                pointer_bits,
                machine_integer_bits: 64,
                machine_float_bits: 64,
            },
        };
        (UniversalContext::new(types, target), id)
    }

    #[test]
    fn usize_is_resolved_only_from_target_layout() {
        for bits in [32, 64] {
            let (context, type_id) = pointer_context(bits);
            let mir = Module {
                values: vec![MirValue {
                    id: MirValueId(0),
                    type_id,
                }],
                ..Module::default()
            };
            assert_eq!(
                lower(&mir, &context.types, &context.target).unwrap().values[0].ty,
                LoweredType::Integer {
                    bits,
                    signed: false,
                }
            );
        }
    }
}
