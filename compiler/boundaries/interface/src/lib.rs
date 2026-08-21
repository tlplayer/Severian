#![forbid(unsafe_code)]

use severian_universal::{
    FloatFormat, IntegerWidth, PrimitiveCategory, PrimitiveRepresentation, TypeDefinition,
    TypeDefinitionKind,
};
use std::fmt;

pub const INTERFACE_VERSION: u16 = 1;

/// Versioned package-interface DTO. It mirrors universal data for transport;
/// it deliberately has no lookup, assignability, literal, or operator logic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveRecord {
    pub version: u16,
    pub declaration_id: [u8; 16],
    pub path: String,
    pub category: CategoryRecord,
    pub representation: RepresentationRecord,
    pub default_literal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoryRecord {
    Boolean,
    Integer,
    Float,
    Text,
    Bytes,
    Absence,
    Unit,
    Arguments,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepresentationRecord {
    Integer { bits: WidthRecord, signed: bool },
    Float { format: FloatRecord },
    PointerInteger { signed: bool },
    Boolean,
    String,
    Bytes,
    None,
    Unit,
    Arguments,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidthRecord {
    Fixed(u16),
    Machine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatRecord {
    Ieee(u16),
    BrainFloat16,
    Machine,
}

impl TryFrom<&TypeDefinition> for PrimitiveRecord {
    type Error = InterfaceError;

    fn try_from(definition: &TypeDefinition) -> Result<Self, Self::Error> {
        let TypeDefinitionKind::Primitive(primitive) = &definition.kind else {
            return Err(InterfaceError::NotPrimitive(definition.path.clone()));
        };
        Ok(Self {
            version: INTERFACE_VERSION,
            declaration_id: definition.declaration.0.to_be_bytes(),
            path: definition.path.clone(),
            category: primitive.category.into(),
            representation: primitive.representation.into(),
            default_literal: primitive.default_literal,
        })
    }
}

impl From<PrimitiveCategory> for CategoryRecord {
    fn from(category: PrimitiveCategory) -> Self {
        match category {
            PrimitiveCategory::Boolean => Self::Boolean,
            PrimitiveCategory::Integer => Self::Integer,
            PrimitiveCategory::Float => Self::Float,
            PrimitiveCategory::Text => Self::Text,
            PrimitiveCategory::Bytes => Self::Bytes,
            PrimitiveCategory::Absence => Self::Absence,
            PrimitiveCategory::Unit => Self::Unit,
            PrimitiveCategory::Arguments => Self::Arguments,
        }
    }
}

impl From<PrimitiveRepresentation> for RepresentationRecord {
    fn from(representation: PrimitiveRepresentation) -> Self {
        match representation {
            PrimitiveRepresentation::Integer { bits, signed } => Self::Integer {
                bits: match bits {
                    IntegerWidth::Fixed(bits) => WidthRecord::Fixed(bits),
                    IntegerWidth::Machine => WidthRecord::Machine,
                },
                signed,
            },
            PrimitiveRepresentation::Float { format } => Self::Float {
                format: match format {
                    FloatFormat::Ieee(bits) => FloatRecord::Ieee(bits),
                    FloatFormat::BrainFloat16 => FloatRecord::BrainFloat16,
                    FloatFormat::Machine => FloatRecord::Machine,
                },
            },
            PrimitiveRepresentation::PointerInteger { signed } => Self::PointerInteger { signed },
            PrimitiveRepresentation::Boolean => Self::Boolean,
            PrimitiveRepresentation::String => Self::String,
            PrimitiveRepresentation::Bytes => Self::Bytes,
            PrimitiveRepresentation::None => Self::None,
            PrimitiveRepresentation::Unit => Self::Unit,
            PrimitiveRepresentation::Arguments => Self::Arguments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterfaceError {
    NotPrimitive(String),
    UnsupportedVersion(u16),
    Truncated,
    InvalidMagic,
    InvalidTag(u8),
    InvalidUtf8,
    TrailingData,
}

impl PrimitiveRecord {
    pub fn validate_version(&self) -> Result<(), InterfaceError> {
        (self.version == INTERFACE_VERSION)
            .then_some(())
            .ok_or(InterfaceError::UnsupportedVersion(self.version))
    }

    pub fn encode(&self) -> Result<Vec<u8>, InterfaceError> {
        self.validate_version()?;
        let path = self.path.as_bytes();
        let path_length = u32::try_from(path.len()).map_err(|_| InterfaceError::Truncated)?;
        let mut bytes = Vec::with_capacity(32 + path.len());
        bytes.extend_from_slice(b"SEVPKGI\0");
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.declaration_id);
        bytes.extend_from_slice(&path_length.to_be_bytes());
        bytes.extend_from_slice(path);
        bytes.push(category_tag(self.category));
        encode_representation(&mut bytes, &self.representation);
        bytes.push(u8::from(self.default_literal));
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, InterfaceError> {
        let mut decoder = Decoder { bytes, cursor: 0 };
        if decoder.take(8)? != b"SEVPKGI\0" {
            return Err(InterfaceError::InvalidMagic);
        }
        let version = u16::from_be_bytes(decoder.array()?);
        if version != INTERFACE_VERSION {
            return Err(InterfaceError::UnsupportedVersion(version));
        }
        let declaration_id = decoder.array()?;
        let path_length = u32::from_be_bytes(decoder.array()?) as usize;
        let path = std::str::from_utf8(decoder.take(path_length)?)
            .map_err(|_| InterfaceError::InvalidUtf8)?
            .to_owned();
        let category = decode_category(decoder.byte()?)?;
        let representation = decode_representation(&mut decoder)?;
        let default_literal = match decoder.byte()? {
            0 => false,
            1 => true,
            tag => return Err(InterfaceError::InvalidTag(tag)),
        };
        if decoder.cursor != bytes.len() {
            return Err(InterfaceError::TrailingData);
        }
        Ok(Self {
            version,
            declaration_id,
            path,
            category,
            representation,
            default_literal,
        })
    }
}

fn category_tag(category: CategoryRecord) -> u8 {
    match category {
        CategoryRecord::Boolean => 0,
        CategoryRecord::Integer => 1,
        CategoryRecord::Float => 2,
        CategoryRecord::Text => 3,
        CategoryRecord::Bytes => 4,
        CategoryRecord::Absence => 5,
        CategoryRecord::Unit => 6,
        CategoryRecord::Arguments => 7,
    }
}

fn decode_category(tag: u8) -> Result<CategoryRecord, InterfaceError> {
    Ok(match tag {
        0 => CategoryRecord::Boolean,
        1 => CategoryRecord::Integer,
        2 => CategoryRecord::Float,
        3 => CategoryRecord::Text,
        4 => CategoryRecord::Bytes,
        5 => CategoryRecord::Absence,
        6 => CategoryRecord::Unit,
        7 => CategoryRecord::Arguments,
        tag => return Err(InterfaceError::InvalidTag(tag)),
    })
}

fn encode_representation(bytes: &mut Vec<u8>, representation: &RepresentationRecord) {
    match representation {
        RepresentationRecord::Integer { bits, signed } => {
            bytes.push(0);
            encode_width(bytes, *bits);
            bytes.push(u8::from(*signed));
        }
        RepresentationRecord::Float { format } => {
            bytes.push(1);
            match format {
                FloatRecord::Ieee(bits) => {
                    bytes.push(0);
                    bytes.extend_from_slice(&bits.to_be_bytes());
                }
                FloatRecord::BrainFloat16 => bytes.push(1),
                FloatRecord::Machine => bytes.push(2),
            }
        }
        RepresentationRecord::PointerInteger { signed } => {
            bytes.extend_from_slice(&[2, u8::from(*signed)]);
        }
        RepresentationRecord::Boolean => bytes.push(3),
        RepresentationRecord::String => bytes.push(4),
        RepresentationRecord::Bytes => bytes.push(5),
        RepresentationRecord::None => bytes.push(6),
        RepresentationRecord::Unit => bytes.push(7),
        RepresentationRecord::Arguments => bytes.push(8),
    }
}

fn encode_width(bytes: &mut Vec<u8>, width: WidthRecord) {
    match width {
        WidthRecord::Fixed(bits) => {
            bytes.push(0);
            bytes.extend_from_slice(&bits.to_be_bytes());
        }
        WidthRecord::Machine => bytes.push(1),
    }
}

fn decode_representation(
    decoder: &mut Decoder<'_>,
) -> Result<RepresentationRecord, InterfaceError> {
    Ok(match decoder.byte()? {
        0 => RepresentationRecord::Integer {
            bits: match decoder.byte()? {
                0 => WidthRecord::Fixed(u16::from_be_bytes(decoder.array()?)),
                1 => WidthRecord::Machine,
                tag => return Err(InterfaceError::InvalidTag(tag)),
            },
            signed: decode_bool(decoder.byte()?)?,
        },
        1 => RepresentationRecord::Float {
            format: match decoder.byte()? {
                0 => FloatRecord::Ieee(u16::from_be_bytes(decoder.array()?)),
                1 => FloatRecord::BrainFloat16,
                2 => FloatRecord::Machine,
                tag => return Err(InterfaceError::InvalidTag(tag)),
            },
        },
        2 => RepresentationRecord::PointerInteger {
            signed: decode_bool(decoder.byte()?)?,
        },
        3 => RepresentationRecord::Boolean,
        4 => RepresentationRecord::String,
        5 => RepresentationRecord::Bytes,
        6 => RepresentationRecord::None,
        7 => RepresentationRecord::Unit,
        8 => RepresentationRecord::Arguments,
        tag => return Err(InterfaceError::InvalidTag(tag)),
    })
}

fn decode_bool(tag: u8) -> Result<bool, InterfaceError> {
    match tag {
        0 => Ok(false),
        1 => Ok(true),
        tag => Err(InterfaceError::InvalidTag(tag)),
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Decoder<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], InterfaceError> {
        let end = self
            .cursor
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(InterfaceError::Truncated)?;
        let value = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8, InterfaceError> {
        Ok(self.take(1)?[0])
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], InterfaceError> {
        self.take(N)?
            .try_into()
            .map_err(|_| InterfaceError::Truncated)
    }
}

impl fmt::Display for InterfaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for InterfaceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versioned_primitive_record_round_trips() {
        let record = PrimitiveRecord {
            version: INTERFACE_VERSION,
            declaration_id: 42u128.to_be_bytes(),
            path: "core.primitives.i32".into(),
            category: CategoryRecord::Integer,
            representation: RepresentationRecord::Integer {
                bits: WidthRecord::Fixed(32),
                signed: true,
            },
            default_literal: false,
        };
        assert_eq!(
            PrimitiveRecord::decode(&record.encode().unwrap()).unwrap(),
            record
        );
    }
}
