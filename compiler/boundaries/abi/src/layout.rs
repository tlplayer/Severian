use crate::{AbiType, RecordRepresentation, ScalarType, TargetDataLayout};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldLayout {
    pub name: String,
    pub offset: u64,
    pub layout: Layout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutKind {
    Scalar,
    Pointer,
    Array { stride: u64, length: u64 },
    Record { fields: Vec<FieldLayout> },
    Union { fields: Vec<FieldLayout> },
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub size: u64,
    pub alignment: u32,
    pub kind: LayoutKind,
}

pub fn layout_of(ty: &AbiType, target: &TargetDataLayout) -> Result<Layout, LayoutError> {
    match ty {
        AbiType::Void => Ok(Layout {
            size: 0,
            alignment: 1,
            kind: LayoutKind::Scalar,
        }),
        AbiType::Scalar(ScalarType::Boolean) => scalar_layout(1, false, target),
        AbiType::Scalar(ScalarType::Integer { bits, .. }) => scalar_layout(*bits, false, target),
        AbiType::Scalar(ScalarType::Float { format }) => scalar_layout(format.bits(), true, target),
        AbiType::Pointer { .. } => Ok(Layout {
            size: target.pointer.size,
            alignment: target.pointer.alignment,
            kind: LayoutKind::Pointer,
        }),
        AbiType::Function(_) => Ok(Layout {
            size: target.pointer.size,
            alignment: target.pointer.alignment,
            kind: LayoutKind::Function,
        }),
        AbiType::Array { element, length } => {
            let element = layout_of(element, target)?;
            let stride = align_to(element.size, element.alignment)?;
            Ok(Layout {
                size: stride.checked_mul(*length).ok_or(LayoutError::Overflow)?,
                alignment: element.alignment,
                kind: LayoutKind::Array {
                    stride,
                    length: *length,
                },
            })
        }
        AbiType::Record(record) => record_layout(record, target),
        AbiType::Union(record) => union_layout(record, target),
        AbiType::Enum(enumeration) => match enumeration.underlying {
            ScalarType::Boolean => scalar_layout(1, false, target),
            ScalarType::Integer { bits, .. } => scalar_layout(bits, false, target),
            ScalarType::Float { .. } => Err(LayoutError::InvalidEnumRepresentation),
        },
        AbiType::Opaque { name } => Err(LayoutError::UnsizedOpaque(name.clone())),
    }
}

fn scalar_layout(bits: u16, float: bool, target: &TargetDataLayout) -> Result<Layout, LayoutError> {
    if bits == 0 {
        return Err(LayoutError::ZeroWidthScalar);
    }
    let scalar = target.scalar(bits, float);
    Ok(Layout {
        size: scalar.size,
        alignment: scalar.alignment,
        kind: LayoutKind::Scalar,
    })
}

fn record_layout(
    record: &crate::RecordType,
    target: &TargetDataLayout,
) -> Result<Layout, LayoutError> {
    if record.representation == RecordRepresentation::Transparent && record.fields.len() != 1 {
        return Err(LayoutError::InvalidTransparentRecord);
    }
    let pack = match record.representation {
        RecordRepresentation::Packed(0) => return Err(LayoutError::InvalidPacking),
        RecordRepresentation::Packed(alignment) => Some(alignment),
        _ => None,
    };
    let mut fields = Vec::with_capacity(record.fields.len());
    let mut offset = 0u64;
    let mut alignment = 1u32;
    for field in &record.fields {
        let layout = layout_of(&field.ty, target)?;
        let field_alignment = pack.map_or(layout.alignment, |pack| layout.alignment.min(pack));
        offset = align_to(offset, field_alignment)?;
        fields.push(FieldLayout {
            name: field.name.clone(),
            offset,
            layout: layout.clone(),
        });
        offset = offset
            .checked_add(layout.size)
            .ok_or(LayoutError::Overflow)?;
        alignment = alignment.max(field_alignment);
    }
    if record.fields.is_empty() {
        alignment = target.aggregate_alignment.max(1);
    }
    Ok(Layout {
        size: align_to(offset, alignment)?,
        alignment,
        kind: LayoutKind::Record { fields },
    })
}

fn union_layout(
    record: &crate::RecordType,
    target: &TargetDataLayout,
) -> Result<Layout, LayoutError> {
    let mut fields = Vec::with_capacity(record.fields.len());
    let mut size = 0u64;
    let mut alignment = 1u32;
    for field in &record.fields {
        let layout = layout_of(&field.ty, target)?;
        size = size.max(layout.size);
        alignment = alignment.max(layout.alignment);
        fields.push(FieldLayout {
            name: field.name.clone(),
            offset: 0,
            layout,
        });
    }
    Ok(Layout {
        size: align_to(size, alignment)?,
        alignment,
        kind: LayoutKind::Union { fields },
    })
}

pub fn align_to(value: u64, alignment: u32) -> Result<u64, LayoutError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(LayoutError::InvalidAlignment(alignment));
    }
    let mask = u64::from(alignment - 1);
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(LayoutError::Overflow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    InvalidAlignment(u32),
    InvalidEnumRepresentation,
    InvalidPacking,
    InvalidTransparentRecord,
    Overflow,
    UnsizedOpaque(String),
    ZeroWidthScalar,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid ABI layout: {self:?}")
    }
}

impl std::error::Error for LayoutError {}
