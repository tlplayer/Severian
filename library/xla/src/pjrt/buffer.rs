use super::compile::RawClient;
use super::host_buffer::RawBuffer;
use crate::{Result, XlaError};
use std::sync::Arc;

pub use severian_dtype::DType as ElementType;

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
        let shape = Shape::new(ElementType::I64, dimensions);
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        Self::new(shape, bytes)
    }

    pub fn from_bf16_bytes(dimensions: impl Into<Vec<i64>>, bytes: &[u8]) -> Result<Self> {
        Self::new(Shape::new(ElementType::BF16, dimensions), bytes.to_vec())
    }
}

pub struct Buffer {
    raw: RawBuffer,
    _client: Arc<RawClient>,
}

impl Buffer {
    pub(crate) fn from_raw(raw: RawBuffer, client: Arc<RawClient>) -> Self {
        Self {
            raw,
            _client: client,
        }
    }

    pub fn shape(&self) -> &Shape {
        self.raw.shape()
    }

    pub fn is_on_cpu(&self) -> Result<bool> {
        self.raw.is_on_cpu()
    }

    pub fn is_on_device(&self, device: &super::device::Device) -> Result<bool> {
        Ok(self.raw.device()? == device.raw().raw())
    }

    pub fn to_host_bytes(&self) -> Result<Vec<u8>> {
        self.raw.to_host()
    }

    pub fn to_f32(&self) -> Result<Vec<f32>> {
        if self.shape().element_type != ElementType::F32 {
            return Err(XlaError::Pjrt("buffer element type is not f32".into()));
        }
        Ok(self
            .to_host_bytes()?
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect())
    }

    pub(crate) fn raw(&self) -> &RawBuffer {
        &self.raw
    }
}
