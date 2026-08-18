use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use crate::{
    AbiAddressSpaceExpr, AbiArgument, AbiArgumentExpr, AbiConstExpr, AbiInstance, AbiParameter,
    AbiParameterExpr, AbiRegistry, AbiSchema, AbiSchemaId, AbiSignature, AbiSignatureExpr, AbiType,
    AbiTypeExpr, AbiValue, AbiValueExpr, ArrayType, PointerType, RecordField, RecordType,
    ResourceRepr, ResourceType, SchemaParamId, SchemaParamKind, UnionField, UnionType,
};

#[derive(Clone, Debug, Default)]
struct Bindings {
    values: HashMap<SchemaParamId, AbiArgument>,
}

impl Bindings {
    fn from_schema(schema: &AbiSchema, arguments: &[AbiArgument]) -> Result<Self, InstantiateError> {
        if schema.parameters.len() != arguments.len() {
            return Err(InstantiateError::ArgumentCount {
                schema: schema.id.clone(),
                expected: schema.parameters.len(),
                actual: arguments.len(),
            });
        }

        let mut values = HashMap::new();
        for (parameter, argument) in schema.parameters.iter().zip(arguments) {
            if parameter.kind != argument.kind() {
                return Err(InstantiateError::ArgumentKind {
                    schema: schema.id.clone(),
                    parameter: parameter.name.clone(),
                    expected: parameter.kind,
                    actual: argument.kind(),
                });
            }
            if values.insert(parameter.id, argument.clone()).is_some() {
                return Err(InstantiateError::DuplicateParameter {
                    schema: schema.id.clone(),
                    parameter: parameter.id,
                });
            }
        }
        Ok(Self { values })
    }

    fn require(&self, id: SchemaParamId) -> Result<&AbiArgument, InstantiateError> {
        self.values.get(&id).ok_or(InstantiateError::UnknownParameter(id))
    }
}

pub fn instantiate_schema(
    registry: &AbiRegistry,
    schema_id: &AbiSchemaId,
    arguments: Vec<AbiArgument>,
) -> Result<AbiInstance, InstantiateError> {
    let mut stack = Vec::new();
    let ty = instantiate_schema_inner(registry, schema_id, &arguments, &mut stack)?;
    Ok(AbiInstance { schema: schema_id.clone(), arguments, ty })
}

fn instantiate_schema_inner(
    registry: &AbiRegistry,
    schema_id: &AbiSchemaId,
    arguments: &[AbiArgument],
    stack: &mut Vec<AbiSchemaId>,
) -> Result<AbiType, InstantiateError> {
    if stack.contains(schema_id) {
        let mut cycle = stack.clone();
        cycle.push(schema_id.clone());
        return Err(InstantiateError::SchemaCycle(cycle));
    }

    let schema = registry
        .schema(schema_id)
        .ok_or_else(|| InstantiateError::UnknownSchema(schema_id.clone()))?;
    let bindings = Bindings::from_schema(schema, arguments)?;

    stack.push(schema_id.clone());
    let result = instantiate_type_expr(registry, &schema.body, &bindings, stack);
    stack.pop();
    result
}

fn instantiate_type_expr(
    registry: &AbiRegistry,
    expr: &AbiTypeExpr,
    bindings: &Bindings,
    stack: &mut Vec<AbiSchemaId>,
) -> Result<AbiType, InstantiateError> {
    Ok(match expr {
        AbiTypeExpr::Unit => AbiType::Unit,
        AbiTypeExpr::Int(v) => AbiType::Int(*v),
        AbiTypeExpr::Float(v) => AbiType::Float(*v),
        AbiTypeExpr::TypeParam(id) => match bindings.require(*id)? {
            AbiArgument::Type(ty) => ty.clone(),
            other => return Err(InstantiateError::ParameterKind {
                parameter: *id,
                expected: SchemaParamKind::Type,
                actual: other.kind(),
            }),
        },
        AbiTypeExpr::Pointer(v) => AbiType::Pointer(PointerType {
            pointee: Box::new(instantiate_type_expr(registry, &v.pointee, bindings, stack)?),
            mutability: v.mutability,
            nullability: v.nullability,
            address_space: instantiate_address_space(&v.address_space, bindings)?,
        }),
        AbiTypeExpr::Array(v) => AbiType::Array(ArrayType {
            element: Box::new(instantiate_type_expr(registry, &v.element, bindings, stack)?),
            length: instantiate_const(&v.length, bindings)?,
        }),
        AbiTypeExpr::Record(v) => AbiType::Record(RecordType {
            id: v.id.clone(),
            repr: v.repr,
            fields: v.fields.iter().map(|field| {
                Ok(RecordField {
                    name: field.name.clone(),
                    ty: instantiate_type_expr(registry, &field.ty, bindings, stack)?,
                })
            }).collect::<Result<Vec<_>, InstantiateError>>()?,
        }),
        AbiTypeExpr::Union(v) => AbiType::Union(UnionType {
            id: v.id.clone(),
            fields: v.fields.iter().map(|field| {
                Ok(UnionField {
                    name: field.name.clone(),
                    ty: instantiate_type_expr(registry, &field.ty, bindings, stack)?,
                })
            }).collect::<Result<Vec<_>, InstantiateError>>()?,
        }),
        AbiTypeExpr::Enum(v) => AbiType::Enum(v.clone()),
        AbiTypeExpr::Function(v) => AbiType::Function(Box::new(instantiate_signature_expr(
            registry, v, bindings, stack,
        )?)),
        AbiTypeExpr::Resource(v) => AbiType::Resource(ResourceType {
            id: v.id.clone(),
            repr: match &v.repr {
                crate::ResourceReprExpr::Pointer { address_space } => ResourceRepr::Pointer {
                    address_space: instantiate_address_space(address_space, bindings)?,
                },
                crate::ResourceReprExpr::Integer(int) => ResourceRepr::Integer(*int),
            },
        }),
        AbiTypeExpr::Opaque(v) => AbiType::Opaque(v.clone()),
        AbiTypeExpr::Apply(application) => {
            let args = application.arguments.iter().map(|arg| {
                instantiate_argument_expr(registry, arg, bindings, stack)
            }).collect::<Result<Vec<_>, InstantiateError>>()?;
            instantiate_schema_inner(registry, &application.schema, &args, stack)?
        }
    })
}

fn instantiate_argument_expr(
    registry: &AbiRegistry,
    expr: &AbiArgumentExpr,
    bindings: &Bindings,
    stack: &mut Vec<AbiSchemaId>,
) -> Result<AbiArgument, InstantiateError> {
    match expr {
        AbiArgumentExpr::Type(ty) => Ok(AbiArgument::Type(instantiate_type_expr(
            registry, ty, bindings, stack,
        )?)),
        AbiArgumentExpr::Const(value) => Ok(AbiArgument::Const(instantiate_const(value, bindings)?)),
        AbiArgumentExpr::AddressSpace(value) => Ok(AbiArgument::AddressSpace(
            instantiate_address_space(value, bindings)?,
        )),
    }
}

fn instantiate_const(expr: &AbiConstExpr, bindings: &Bindings) -> Result<u64, InstantiateError> {
    match expr {
        AbiConstExpr::Value(v) => Ok(*v),
        AbiConstExpr::Param(id) => match bindings.require(*id)? {
            AbiArgument::Const(v) => Ok(*v),
            other => Err(InstantiateError::ParameterKind {
                parameter: *id,
                expected: SchemaParamKind::Const,
                actual: other.kind(),
            }),
        },
    }
}

fn instantiate_address_space(
    expr: &AbiAddressSpaceExpr,
    bindings: &Bindings,
) -> Result<crate::AddressSpaceId, InstantiateError> {
    match expr {
        AbiAddressSpaceExpr::Value(v) => Ok(v.clone()),
        AbiAddressSpaceExpr::Param(id) => match bindings.require(*id)? {
            AbiArgument::AddressSpace(v) => Ok(v.clone()),
            other => Err(InstantiateError::ParameterKind {
                parameter: *id,
                expected: SchemaParamKind::AddressSpace,
                actual: other.kind(),
            }),
        },
    }
}

fn instantiate_signature_expr(
    registry: &AbiRegistry,
    expr: &AbiSignatureExpr,
    bindings: &Bindings,
    stack: &mut Vec<AbiSchemaId>,
) -> Result<AbiSignature, InstantiateError> {
    Ok(AbiSignature {
        abi: expr.abi.clone(),
        parameters: expr.parameters.iter().map(|parameter| {
            instantiate_parameter_expr(registry, parameter, bindings, stack)
        }).collect::<Result<Vec<_>, InstantiateError>>()?,
        returns: instantiate_value_expr(registry, &expr.returns, bindings, stack)?,
        variadic: expr.variadic,
    })
}

fn instantiate_parameter_expr(
    registry: &AbiRegistry,
    expr: &AbiParameterExpr,
    bindings: &Bindings,
    stack: &mut Vec<AbiSchemaId>,
) -> Result<AbiParameter, InstantiateError> {
    Ok(AbiParameter {
        name: expr.name.clone(),
        mode: expr.mode,
        value: instantiate_value_expr(registry, &expr.value, bindings, stack)?,
    })
}

fn instantiate_value_expr(
    registry: &AbiRegistry,
    expr: &AbiValueExpr,
    bindings: &Bindings,
    stack: &mut Vec<AbiSchemaId>,
) -> Result<AbiValue, InstantiateError> {
    Ok(AbiValue {
        ty: instantiate_type_expr(registry, &expr.ty, bindings, stack)?,
        ownership: expr.ownership,
        lifetime: expr.lifetime,
    })
}

pub fn validate_schema_parameters(schema: &AbiSchema) -> Result<(), InstantiateError> {
    let mut ids = HashSet::new();
    for parameter in &schema.parameters {
        if !ids.insert(parameter.id) {
            return Err(InstantiateError::DuplicateParameter {
                schema: schema.id.clone(),
                parameter: parameter.id,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstantiateError {
    UnknownSchema(AbiSchemaId),
    SchemaCycle(Vec<AbiSchemaId>),
    ArgumentCount {
        schema: AbiSchemaId,
        expected: usize,
        actual: usize,
    },
    ArgumentKind {
        schema: AbiSchemaId,
        parameter: String,
        expected: SchemaParamKind,
        actual: SchemaParamKind,
    },
    DuplicateParameter {
        schema: AbiSchemaId,
        parameter: SchemaParamId,
    },
    UnknownParameter(SchemaParamId),
    ParameterKind {
        parameter: SchemaParamId,
        expected: SchemaParamKind,
        actual: SchemaParamKind,
    },
}

impl fmt::Display for InstantiateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSchema(id) => write!(f, "unknown ABI schema `{id}`"),
            Self::SchemaCycle(ids) => {
                let names = ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(" -> ");
                write!(f, "cyclic ABI schema expansion: {names}")
            }
            Self::ArgumentCount { schema, expected, actual } => {
                write!(f, "ABI schema `{schema}` expects {expected} arguments, got {actual}")
            }
            Self::ArgumentKind { schema, parameter, expected, actual } => write!(
                f,
                "ABI schema `{schema}` parameter `{parameter}` expects {expected:?}, got {actual:?}",
            ),
            Self::DuplicateParameter { schema, parameter } => write!(
                f,
                "ABI schema `{schema}` declares parameter {} more than once",
                parameter.0,
            ),
            Self::UnknownParameter(parameter) => {
                write!(f, "unknown ABI schema parameter {}", parameter.0)
            }
            Self::ParameterKind { parameter, expected, actual } => write!(
                f,
                "ABI schema parameter {} expects {expected:?}, got {actual:?}",
                parameter.0,
            ),
        }
    }
}

impl Error for InstantiateError {}
