use std::error::Error;
use std::fmt;

use crate::{
    AbiId, AbiSchemaId, AbiSignature, AbiSpec, AbiType, IntWidth, RecordId, RecordRepr, UnionId,
};

pub fn validate_signature(signature: &AbiSignature, spec: &AbiSpec) -> Result<(), AbiError> {
    if signature.abi != spec.id {
        return Err(AbiError::WrongAbi {
            signature: signature.abi.clone(),
            spec: spec.id.clone(),
        });
    }

    if signature.variadic && !spec.supports_variadic {
        return Err(AbiError::VariadicUnsupported(spec.id.clone()));
    }

    for parameter in &signature.parameters {
        validate_type(&parameter.value.ty, Position::Parameter)?;
    }
    validate_type(&signature.returns.ty, Position::Return)?;
    Ok(())
}

pub fn validate_type(ty: &AbiType, position: Position) -> Result<(), AbiError> {
    match ty {
        AbiType::Unit => {
            if position == Position::Parameter {
                return Err(AbiError::UnitParameter);
            }
        }
        AbiType::Int(int) => validate_int_width(int.width)?,
        AbiType::Float(_) => {}
        AbiType::Pointer(pointer) => validate_pointee_type(&pointer.pointee)?,
        AbiType::Array(array) => {
            if array.length == 0 {
                return Err(AbiError::ZeroLengthArray);
            }
            validate_type(&array.element, Position::Nested)?;
        }
        AbiType::Record(record) => {
            if record.repr == RecordRepr::Transparent && record.fields.len() != 1 {
                return Err(AbiError::InvalidTransparentRecord(record.id.clone()));
            }
            for field in &record.fields {
                validate_type(&field.ty, Position::Nested)?;
            }
        }
        AbiType::Union(union) => {
            if union.fields.is_empty() {
                return Err(AbiError::EmptyUnion(union.id.clone()));
            }
            for field in &union.fields {
                validate_type(&field.ty, Position::Nested)?;
            }
        }
        AbiType::Enum(enumeration) => validate_int_width(enumeration.repr.width)?,
        AbiType::Function(signature) => {
            for parameter in &signature.parameters {
                validate_type(&parameter.value.ty, Position::Parameter)?;
            }
            validate_type(&signature.returns.ty, Position::Return)?;
        }
        AbiType::Resource(resource) => {
            if let crate::ResourceRepr::Integer(int) = &resource.repr {
                validate_int_width(int.width)?;
            }
        }
        AbiType::Opaque(id) => return Err(AbiError::OpaqueByValue(id.clone())),
    }
    Ok(())
}

fn validate_int_width(width: IntWidth) -> Result<(), AbiError> {
    if let IntWidth::Fixed(bits) = width {
        if bits == 0 || bits % 8 != 0 {
            return Err(AbiError::InvalidIntegerWidth(bits));
        }
    }
    Ok(())
}

fn validate_pointee_type(ty: &AbiType) -> Result<(), AbiError> {
    match ty {
        AbiType::Opaque(_) => Ok(()),
        _ => validate_type(ty, Position::Nested),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    Parameter,
    Return,
    Nested,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbiError {
    WrongAbi { signature: AbiId, spec: AbiId },
    VariadicUnsupported(AbiId),
    UnitParameter,
    InvalidIntegerWidth(u16),
    ZeroLengthArray,
    InvalidTransparentRecord(RecordId),
    EmptyUnion(UnionId),
    OpaqueByValue(crate::OpaqueId),
    DuplicateAbi(AbiId),
    UnknownAbi(AbiId),
    DuplicateSchema(AbiSchemaId),
    UnknownSchema(AbiSchemaId),
}

impl fmt::Display for AbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongAbi { signature, spec } => write!(f, "ABI signature `{signature}` cannot be validated as `{spec}`"),
            Self::VariadicUnsupported(id) => write!(f, "ABI `{id}` does not support variadic calls"),
            Self::UnitParameter => write!(f, "unit is not a valid ABI parameter"),
            Self::InvalidIntegerWidth(bits) => write!(f, "invalid ABI integer width `{bits}`"),
            Self::ZeroLengthArray => write!(f, "zero-length arrays are not ABI-safe"),
            Self::InvalidTransparentRecord(id) => write!(f, "transparent record `{id}` must contain exactly one field"),
            Self::EmptyUnion(id) => write!(f, "ABI union `{id}` has no fields"),
            Self::OpaqueByValue(id) => write!(f, "opaque ABI type `{id}` must be behind a pointer or resource"),
            Self::DuplicateAbi(id) => write!(f, "ABI `{id}` is already registered"),
            Self::UnknownAbi(id) => write!(f, "ABI `{id}` is not registered"),
            Self::DuplicateSchema(id) => write!(f, "ABI schema `{id}` is already registered"),
            Self::UnknownSchema(id) => write!(f, "ABI schema `{id}` is not registered"),
        }
    }
}

impl Error for AbiError {}
