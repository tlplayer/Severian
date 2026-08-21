#![forbid(unsafe_code)]

use severian_lir::{
    BinaryOperation, Block as LirBlock, Constant, Function as LirFunction, FunctionId,
    FunctionLinkage, LoweredFloatFormat, LoweredType, Module as LirModule,
    Operation as LirOperation, UnaryOperation, Value, ValueId,
};
use severian_mir::{Block as MirBlock, Module as MirModule, Operation as MirOperation};
use severian_target::TargetSpec;
use severian_universal::{
    BinaryOperator, FloatFormat, IntegerWidth, LiteralValue, PrimitiveRepresentation, TypeContext,
    TypeId, UnaryOperator,
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
    let initializer = lower_block(&mir.initializer, mir)?;
    let functions = mir
        .functions
        .iter()
        .map(|function| {
            Ok(LirFunction {
                id: FunctionId(function.id.0),
                name: function.name.clone(),
                parameters: function
                    .parameters
                    .iter()
                    .map(|value| ValueId(value.0))
                    .collect(),
                result: lower_type(function.result, types, target)?,
                body: function
                    .body
                    .as_ref()
                    .map(|body| lower_block(body, mir))
                    .transpose()?,
                linkage: match &function.call_type {
                    severian_mir::CallType::Severian => FunctionLinkage::Internal,
                    severian_mir::CallType::External(call) => FunctionLinkage::External {
                        symbol: call.symbol.0.clone(),
                    },
                },
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    Ok(LirModule {
        values,
        globals: mir.globals.iter().map(|value| ValueId(value.0)).collect(),
        initializer,
        functions,
        entry: mir.entry.map(|entry| FunctionId(entry.0)),
    })
}

fn lower_block(block: &MirBlock, module: &MirModule) -> Result<LirBlock, LoweringError> {
    let mut operations = Vec::new();
    for operation in &block.operations {
        let operation = match operation {
            MirOperation::Coverage { point } => LirOperation::Coverage {
                key: point
                    .key
                    .clone()
                    .expect("source attachment assigns every coverage key"),
            },
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
            MirOperation::Call {
                function,
                arguments,
                result,
            } => LirOperation::Call {
                function: FunctionId(function.0),
                arguments: arguments.iter().map(|value| ValueId(value.0)).collect(),
                result: ValueId(result.0),
            },
            MirOperation::Return { value } => LirOperation::Return {
                value: value.map(|value| ValueId(value.0)),
            },
            MirOperation::Assert {
                condition,
                message,
                origin,
            } => LirOperation::Assert {
                condition: ValueId(condition.0),
                message: message.map(|message| ValueId(message.0)),
                location: origin.location.as_ref().map(|location| {
                    severian_lir::AssertionLocation {
                        file: location.file.clone(),
                        line: location.line,
                        column: location.column,
                        expression: location.expression.clone(),
                    }
                }),
            },
            MirOperation::If {
                condition,
                then_block,
                else_block,
            } => LirOperation::If {
                condition: ValueId(condition.0),
                then_block: lower_block(then_block, module)?,
                else_block: lower_block(else_block, module)?,
            },
            MirOperation::Match { subject, arms } => {
                let subject_type = module
                    .values
                    .iter()
                    .find(|value| value.id == *subject)
                    .ok_or(LoweringError::UnknownValue(*subject))?
                    .type_id;
                let arm = arms
                    .iter()
                    .find(|arm| arm.type_id == Some(subject_type))
                    .or_else(|| arms.iter().find(|arm| arm.type_id.is_none()))
                    .ok_or(LoweringError::NonExhaustiveMatch(subject_type))?;
                operations.extend(lower_block(&arm.body, module)?.operations);
                continue;
            }
            MirOperation::CompiledRegionCall {
                artifact,
                inputs,
                outputs,
            } => LirOperation::ArtifactCall {
                artifact: *artifact,
                inputs: inputs.iter().map(|value| ValueId(value.0)).collect(),
                outputs: outputs.iter().map(|value| ValueId(value.0)).collect(),
            },
        };
        operations.push(operation);
    }
    Ok(LirBlock { operations })
}

fn lower_constant(value: &LiteralValue) -> Constant {
    match value {
        LiteralValue::Integer(value) => Constant::Integer(value.clone()),
        LiteralValue::Float(value) => Constant::Float(value.clone()),
        LiteralValue::Boolean(value) => Constant::Boolean(*value),
        LiteralValue::Character(value) => Constant::Integer(u32::from(*value).to_string()),
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
        BinaryOperator::Contains => BinaryOperation::Contains,
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
                IntegerWidth::Machine => target.machine_integer_bits(),
            },
            signed,
        },
        PrimitiveRepresentation::PointerInteger { signed } => LoweredType::Integer {
            bits: target.pointer_bits(),
            signed,
        },
        PrimitiveRepresentation::Float { format } => LoweredType::Float {
            format: match format {
                FloatFormat::Ieee(bits) => LoweredFloatFormat::Ieee(bits),
                FloatFormat::BrainFloat16 => LoweredFloatFormat::BrainFloat16,
                FloatFormat::Machine => LoweredFloatFormat::Ieee(target.machine_float_bits()),
            },
        },
        PrimitiveRepresentation::Boolean => LoweredType::Boolean,
        PrimitiveRepresentation::Character => LoweredType::Integer {
            bits: 32,
            signed: false,
        },
        PrimitiveRepresentation::String => LoweredType::String,
        PrimitiveRepresentation::Bytes => LoweredType::Bytes,
        PrimitiveRepresentation::None => LoweredType::None,
        PrimitiveRepresentation::Unit => LoweredType::Unit,
        PrimitiveRepresentation::Arguments => LoweredType::Arguments,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringError {
    NotPrimitive(TypeId),
    UnknownValue(severian_mir::ValueId),
    NonExhaustiveMatch(TypeId),
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
    use severian_universal::{PrimitiveCategory, TypeContextBuilder, UniversalContext};

    fn pointer_context() -> (UniversalContext, TypeId) {
        let mut types = TypeContextBuilder::new();
        let id = types.register_declaration("core.usize", "usize").unwrap();
        types
            .define_primitive(
                id,
                PrimitiveCategory::Integer,
                PrimitiveRepresentation::PointerInteger { signed: false },
                false,
            )
            .unwrap();
        (UniversalContext::new(types.build()), id)
    }

    #[test]
    fn usize_is_resolved_only_from_target_layout() {
        for bits in [32, 64] {
            let (context, type_id) = pointer_context();
            let mir = Module {
                values: vec![MirValue {
                    id: MirValueId(0),
                    type_id,
                }],
                ..Module::default()
            };
            assert_eq!(
                lower(
                    &mir,
                    &context.types,
                    &if bits == 32 {
                        TargetSpec::new("wasm32-unknown-wasi")
                    } else {
                        TargetSpec::new("x86_64-unknown-linux")
                    },
                )
                .unwrap()
                .values[0]
                    .ty,
                LoweredType::Integer {
                    bits,
                    signed: false,
                }
            );
        }
    }
}
