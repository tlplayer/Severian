use super::host_buffer::RawBuffer;
use crate::{Result, XlaError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    Pred,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
}

impl ElementType {
    pub fn byte_width(self) -> usize {
        match self {
            Self::Pred | Self::S8 | Self::U8 => 1,
            Self::S16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::S32 | Self::U32 | Self::F32 => 4,
            Self::S64 | Self::U64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub element_type: ElementType,
    pub dimensions: Vec<i64>,
}

impl Shape {
    pub fn new(element_type: ElementType, dimensions: impl Into<Vec<i64>>) -> Self {
        Self {
            element_type,
            dimensions: dimensions.into(),
        }
    }

    pub fn element_count(&self) -> Result<usize> {
        self.dimensions.iter().try_fold(1usize, |acc, &dimension| {
            if dimension < 0 {
                return Err(XlaError::Pjrt(format!(
                    "dynamic/negative host dimension {dimension} cannot be uploaded"
                )));
            }

            acc.checked_mul(dimension as usize)
                .ok_or_else(|| XlaError::Pjrt("buffer element count overflow".into()))
        })
    }

    pub fn byte_len(&self) -> Result<usize> {
        self.element_count()?
            .checked_mul(self.element_type.byte_width())
            .ok_or_else(|| XlaError::Pjrt("buffer byte size overflow".into()))
    }
}

#[derive(Debug, Clone)]
pub struct HostBuffer {
    pub shape: Shape,
    pub bytes: Vec<u8>,
}

impl HostBuffer {
    pub fn new(shape: Shape, bytes: Vec<u8>) -> Result<Self> {
        let expected = shape.byte_len()?;
        if bytes.len() != expected {
            return Err(XlaError::Pjrt(format!(
                "host buffer has {} bytes; shape requires {expected}",
                bytes.len()
            )));
        }

        Ok(Self { shape, bytes })
    }

    pub fn from_f32(dimensions: impl Into<Vec<i64>>, values: &[f32]) -> Result<Self> {
        let shape = Shape::new(ElementType::F32, dimensions);
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        Self::new(shape, bytes)
    }

    pub fn from_i64(dimensions: impl Into<Vec<i64>>, values: &[i64]) -> Result<Self> {
        let shape = Shape::new(ElementType::S64, dimensions);
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        Self::new(shape, bytes)
    }
}

pub struct Buffer {
    raw: RawBuffer,
}

impl Buffer {
    pub(crate) fn from_raw(raw: RawBuffer) -> Self { Self { raw } }

    pub(crate) fn raw(&self) -> &RawBuffer { &self.raw }

    pub fn shape(&self) -> &Shape { self.raw.shape() }
}
