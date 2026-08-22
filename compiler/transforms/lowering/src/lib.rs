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
    let mut values = mir
        .values
        .iter()
        .map(|value| {
            Ok(Value {
                id: ValueId(value.id.0),
                ty: lower_mir_type(value.type_id, mir, types, target)?,
            })
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let initializer = lower_block(&mir.initializer, mir, types, target, &mut values)?;
    let mut functions = Vec::new();
    for function in &mir.functions {
        functions.push(LirFunction {
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
                .map(|body| lower_block(body, mir, types, target, &mut values))
                .transpose()?,
            linkage: match &function.call_type {
                severian_mir::CallType::Severian => FunctionLinkage::Internal,
                severian_mir::CallType::External(call) => FunctionLinkage::External {
                    symbol: call.symbol.0.clone(),
                },
            },
        });
    }
    Ok(LirModule {
        values,
        globals: mir.globals.iter().map(|value| ValueId(value.0)).collect(),
        initializer,
        functions,
        entry: mir.entry.map(|entry| FunctionId(entry.0)),
        traits: mir
            .traits
            .iter()
            .map(|declaration| {
                Ok(severian_lir::TraitDeclaration {
                    id: severian_lir::TraitId {
                        package: declaration.definition.package,
                        module: declaration.definition.module,
                        declaration: declaration.definition.declaration.0,
                    },
                    name: declaration.name.clone(),
                    methods: declaration
                        .methods
                        .iter()
                        .map(|method| {
                            Ok(severian_lir::TraitMethodDeclaration {
                                name: method.name.clone(),
                                parameters: method
                                    .parameters
                                    .iter()
                                    .map(|parameter| match parameter {
                                        severian_mir::TraitType::SelfType => {
                                            Ok(severian_lir::TraitType::SelfType)
                                        }
                                        severian_mir::TraitType::Concrete(ty) => {
                                            lower_type(*ty, types, target)
                                                .map(severian_lir::TraitType::Concrete)
                                        }
                                    })
                                    .collect::<Result<Vec<_>, _>>()?,
                                result: match method.result {
                                    severian_mir::TraitType::SelfType => {
                                        severian_lir::TraitType::SelfType
                                    }
                                    severian_mir::TraitType::Concrete(ty) => {
                                        severian_lir::TraitType::Concrete(lower_type(
                                            ty, types, target,
                                        )?)
                                    }
                                },
                            })
                        })
                        .collect::<Result<Vec<_>, LoweringError>>()?,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?,
        classes: mir
            .classes
            .iter()
            .enumerate()
            .map(|(id, declaration)| {
                Ok(severian_lir::ClassDeclaration {
                    id: id as u32,
                    name: declaration.name.clone(),
                    fields: declaration
                        .fields
                        .iter()
                        .map(|field| {
                            Ok(severian_lir::ClassFieldDeclaration {
                                name: field.name.clone(),
                                ty: lower_mir_type(field.ty, mir, types, target)?,
                            })
                        })
                        .collect::<Result<Vec<_>, LoweringError>>()?,
                })
            })
            .collect::<Result<Vec<_>, LoweringError>>()?,
    })
}

fn lower_block(
    block: &MirBlock,
    module: &MirModule,
    types: &TypeContext,
    target: &TargetSpec,
    values: &mut Vec<Value>,
) -> Result<LirBlock, LoweringError> {
    let mut operations = Vec::new();
    let mut owned_strings = Vec::new();
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
            } => {
                let left_type = mir_value_type(module, *left)?;
                if lower_type(left_type, types, target)? == LoweredType::String {
                    match operator {
                        BinaryOperator::Add => {
                            let result = ValueId(result.0);
                            owned_strings.push(result);
                            LirOperation::RuntimeCall {
                                symbol: "__sev_string_concat".into(),
                                arguments: vec![ValueId(left.0), ValueId(right.0)],
                                result: Some(result),
                            }
                        }
                        BinaryOperator::Equal
                        | BinaryOperator::NotEqual
                        | BinaryOperator::Less
                        | BinaryOperator::LessEqual
                        | BinaryOperator::Greater
                        | BinaryOperator::GreaterEqual => {
                            let comparison = new_value(
                                values,
                                LoweredType::Integer {
                                    bits: 32,
                                    signed: true,
                                },
                            );
                            let zero = new_value(
                                values,
                                LoweredType::Integer {
                                    bits: 32,
                                    signed: true,
                                },
                            );
                            operations.push(LirOperation::RuntimeCall {
                                symbol: "__sev_string_compare".into(),
                                arguments: vec![ValueId(left.0), ValueId(right.0)],
                                result: Some(comparison),
                            });
                            operations.push(LirOperation::Constant {
                                value: Constant::Integer("0".into()),
                                result: zero,
                            });
                            LirOperation::Binary {
                                operator: lower_binary(*operator),
                                left: comparison,
                                right: zero,
                                result: ValueId(result.0),
                            }
                        }
                        _ => {
                            return Err(LoweringError::UnsupportedStringOperation(*operator));
                        }
                    }
                } else {
                    LirOperation::Binary {
                        operator: lower_binary(*operator),
                        left: ValueId(left.0),
                        right: ValueId(right.0),
                        result: ValueId(result.0),
                    }
                }
            }
            MirOperation::Aggregate {
                class,
                fields,
                result,
            } => LirOperation::Aggregate {
                class: module
                    .classes
                    .iter()
                    .position(|known| known.id == *class)
                    .ok_or(LoweringError::NotPrimitive(*class))? as u32,
                fields: fields.iter().map(|value| ValueId(value.0)).collect(),
                result: ValueId(result.0),
            },
            MirOperation::FieldGet {
                object,
                field,
                result,
            } => LirOperation::FieldGet {
                object: ValueId(object.0),
                field: *field,
                result: ValueId(result.0),
            },
            MirOperation::FieldSet {
                object,
                field,
                value,
                result,
            } => LirOperation::FieldSet {
                object: ValueId(object.0),
                field: *field,
                value: ValueId(value.0),
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
                then_block: lower_block(then_block, module, types, target, values)?,
                else_block: lower_block(else_block, module, types, target, values)?,
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
                operations
                    .extend(lower_block(&arm.body, module, types, target, values)?.operations);
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
    insert_owned_string_releases(&mut operations, &owned_strings, module)?;
    Ok(LirBlock { operations })
}

fn mir_value_type(
    module: &MirModule,
    value: severian_mir::ValueId,
) -> Result<TypeId, LoweringError> {
    module
        .values
        .iter()
        .find(|known| known.id == value)
        .map(|known| known.type_id)
        .ok_or(LoweringError::UnknownValue(value))
}

fn lower_mir_type(
    type_id: TypeId,
    module: &MirModule,
    types: &TypeContext,
    target: &TargetSpec,
) -> Result<LoweredType, LoweringError> {
    if let Some(id) = module.classes.iter().position(|class| class.id == type_id) {
        Ok(LoweredType::Aggregate(id as u32))
    } else {
        lower_type(type_id, types, target)
    }
}

fn new_value(values: &mut Vec<Value>, ty: LoweredType) -> ValueId {
    let id = ValueId(values.len() as u32);
    values.push(Value { id, ty });
    id
}

fn insert_owned_string_releases(
    operations: &mut Vec<LirOperation>,
    owned: &[ValueId],
    module: &MirModule,
) -> Result<(), LoweringError> {
    let mut releases = Vec::new();
    for value in owned {
        if module.globals.iter().any(|global| global.0 == value.0)
            || operations
                .iter()
                .any(|operation| returns_value(operation, *value))
        {
            return Err(LoweringError::OwnedStringEscapes(*value));
        }
        // Storing the allocation in an aggregate transfers its lifetime to
        // that aggregate. Releasing it after the insert leaves a dangling
        // field for subsequent method/property reads. Aggregate destruction
        // will become the corresponding release point once destructors are
        // represented in LIR; until then, retain the allocation.
        if operations.iter().any(|operation| match operation {
            LirOperation::Aggregate { fields, .. } => fields.contains(value),
            LirOperation::FieldSet {
                value: field_value, ..
            } => field_value == value,
            _ => false,
        }) {
            continue;
        }
        let definition = operations
            .iter()
            .position(|operation| {
                matches!(operation, LirOperation::RuntimeCall { result: Some(result), .. } if result == value)
            })
            .expect("every owned string is produced by a runtime call");
        let last_use = operations
            .iter()
            .enumerate()
            .filter(|(_, operation)| operation_uses_value(operation, *value))
            .map(|(index, _)| index)
            .max()
            .unwrap_or(definition);
        releases.push((last_use + 1, *value));
    }
    releases.sort_by_key(|(index, _)| std::cmp::Reverse(*index));
    for (index, value) in releases {
        operations.insert(
            index,
            LirOperation::RuntimeCall {
                symbol: "__sev_string_release".into(),
                arguments: vec![value],
                result: None,
            },
        );
    }
    Ok(())
}

fn operation_uses_value(operation: &LirOperation, value: ValueId) -> bool {
    match operation {
        LirOperation::Aggregate { fields, .. } => fields.contains(&value),
        LirOperation::FieldGet { object, .. } => *object == value,
        LirOperation::FieldSet {
            object,
            value: field_value,
            ..
        } => *object == value || *field_value == value,
        LirOperation::Unary { operand, .. } => *operand == value,
        LirOperation::Binary { left, right, .. } => *left == value || *right == value,
        LirOperation::Call { arguments, .. } | LirOperation::RuntimeCall { arguments, .. } => {
            arguments.contains(&value)
        }
        LirOperation::Return { value: returned } => *returned == Some(value),
        LirOperation::Assert {
            condition, message, ..
        } => *condition == value || *message == Some(value),
        LirOperation::If {
            condition,
            then_block,
            else_block,
        } => {
            *condition == value
                || then_block
                    .operations
                    .iter()
                    .any(|operation| operation_uses_value(operation, value))
                || else_block
                    .operations
                    .iter()
                    .any(|operation| operation_uses_value(operation, value))
        }
        LirOperation::ArtifactCall {
            inputs, outputs, ..
        } => inputs.contains(&value) || outputs.contains(&value),
        LirOperation::Coverage { .. } | LirOperation::Constant { .. } => false,
    }
}

fn returns_value(operation: &LirOperation, value: ValueId) -> bool {
    match operation {
        LirOperation::Return {
            value: Some(returned),
        } => *returned == value,
        LirOperation::If {
            then_block,
            else_block,
            ..
        } => then_block
            .operations
            .iter()
            .chain(&else_block.operations)
            .any(|operation| returns_value(operation, value)),
        _ => false,
    }
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
    UnsupportedStringOperation(BinaryOperator),
    OwnedStringEscapes(ValueId),
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

    fn string_context() -> (UniversalContext, TypeId, TypeId) {
        let mut types = TypeContextBuilder::new();
        let string = types.register_declaration("core.string", "string").unwrap();
        types
            .define_primitive(
                string,
                PrimitiveCategory::Text,
                PrimitiveRepresentation::String,
                true,
            )
            .unwrap();
        let boolean = types.register_declaration("core.bool", "bool").unwrap();
        types
            .define_primitive(
                boolean,
                PrimitiveCategory::Boolean,
                PrimitiveRepresentation::Boolean,
                true,
            )
            .unwrap();
        (UniversalContext::new(types.build()), string, boolean)
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

    #[test]
    fn string_operations_become_owned_runtime_calls_before_emission() {
        let (context, string, boolean) = string_context();
        let mir = Module {
            values: vec![
                MirValue {
                    id: MirValueId(0),
                    type_id: string,
                },
                MirValue {
                    id: MirValueId(1),
                    type_id: string,
                },
                MirValue {
                    id: MirValueId(2),
                    type_id: string,
                },
                MirValue {
                    id: MirValueId(3),
                    type_id: boolean,
                },
            ],
            initializer: MirBlock {
                operations: vec![
                    MirOperation::Constant {
                        value: LiteralValue::String("left".into()),
                        result: MirValueId(0),
                    },
                    MirOperation::Constant {
                        value: LiteralValue::String("right".into()),
                        result: MirValueId(1),
                    },
                    MirOperation::Binary {
                        operator: BinaryOperator::Add,
                        left: MirValueId(0),
                        right: MirValueId(1),
                        result: MirValueId(2),
                    },
                    MirOperation::Binary {
                        operator: BinaryOperator::Equal,
                        left: MirValueId(2),
                        right: MirValueId(1),
                        result: MirValueId(3),
                    },
                ],
            },
            ..Module::default()
        };
        let lir = lower(&mir, &context.types, &TargetSpec::host()).unwrap();
        let symbols = lir
            .initializer
            .operations
            .iter()
            .filter_map(|operation| match operation {
                LirOperation::RuntimeCall { symbol, .. } => Some(symbol.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            symbols,
            [
                "__sev_string_concat",
                "__sev_string_compare",
                "__sev_string_release"
            ]
        );
    }

    #[test]
    fn owned_string_escape_fails_closed_until_transfer_is_modeled() {
        let (context, string, _) = string_context();
        let mir = Module {
            values: (0..3)
                .map(|id| MirValue {
                    id: MirValueId(id),
                    type_id: string,
                })
                .collect(),
            initializer: MirBlock {
                operations: vec![
                    MirOperation::Constant {
                        value: LiteralValue::String("left".into()),
                        result: MirValueId(0),
                    },
                    MirOperation::Constant {
                        value: LiteralValue::String("right".into()),
                        result: MirValueId(1),
                    },
                    MirOperation::Binary {
                        operator: BinaryOperator::Add,
                        left: MirValueId(0),
                        right: MirValueId(1),
                        result: MirValueId(2),
                    },
                    MirOperation::Return {
                        value: Some(MirValueId(2)),
                    },
                ],
            },
            ..Module::default()
        };
        assert!(matches!(
            lower(&mir, &context.types, &TargetSpec::host()),
            Err(LoweringError::OwnedStringEscapes(ValueId(2)))
        ));
    }
}
