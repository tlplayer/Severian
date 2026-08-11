//! Native, sharded Safetensors checkpoint access.
//!
//! Shards are memory mapped and tensor payloads remain byte ranges into those
//! mappings until an explicit PJRT upload. No framework object or private
//! checkpoint representation sits between the Hugging Face files and PJRT.

use crate::{Buffer, HostBuffer, PjrtClient, Result, XlaError};
use memmap2::{Mmap, MmapOptions};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorDType {
    BF16,
    F16,
    F32,
    F64,
    I32,
    I64,
    U8,
}

impl SafeTensorDType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "BF16" => Ok(Self::BF16),
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "U8" => Ok(Self::U8),
            other => Err(XlaError::Pjrt(format!(
                "unsupported Safetensors dtype `{other}`"
            ))),
        }
    }

    fn byte_width(self) -> usize {
        match self {
            Self::BF16 | Self::F16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorEntry {
    pub name: String,
    pub dtype: SafeTensorDType,
    pub shape: Vec<i64>,
    pub start: usize,
    pub end: usize,
    pub shard: PathBuf,
}

#[derive(Clone)]
pub struct MappedTensor {
    entry: SafeTensorEntry,
    mapping: Arc<Mmap>,
}

impl MappedTensor {
    pub fn entry(&self) -> &SafeTensorEntry {
        &self.entry
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mapping[self.entry.start..self.entry.end]
    }
}

pub struct SafeTensorStore {
    model_directory: PathBuf,
    weight_map: BTreeMap<String, PathBuf>,
    entries: Mutex<HashMap<PathBuf, Arc<BTreeMap<String, SafeTensorEntry>>>>,
    mappings: Mutex<HashMap<PathBuf, Arc<Mmap>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorValidation {
    pub tensors: usize,
    pub shards: usize,
    pub payload_bytes: u64,
    pub bf16_payload_bytes: u64,
}

impl SafeTensorStore {
    pub fn open(model_directory: impl AsRef<Path>) -> Result<Self> {
        let model_directory = model_directory.as_ref().to_path_buf();
        let index_path = model_directory.join("model.safetensors.index.json");
        let index: Value = serde_json::from_slice(&std::fs::read(&index_path)?)
            .map_err(|error| XlaError::Pjrt(format!(
                "invalid Safetensors index {}: {error}",
                index_path.display()
            )))?;
        let raw_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| XlaError::Pjrt(
                "Safetensors index is missing object `weight_map`".into(),
            ))?;
        let mut weight_map = BTreeMap::new();
        for (name, shard) in raw_map {
            let shard = shard.as_str().ok_or_else(|| XlaError::Pjrt(format!(
                "Safetensors shard for `{name}` is not a string"
            )))?;
            let shard = PathBuf::from(shard);
            if shard.is_absolute() || shard.components().any(|part| matches!(
                part,
                std::path::Component::ParentDir
            )) {
                return Err(XlaError::Pjrt(format!(
                    "unsafe Safetensors shard path `{}`",
                    shard.display()
                )));
            }
            weight_map.insert(name.clone(), shard);
        }
        Ok(Self {
            model_directory,
            weight_map,
            entries: Mutex::new(HashMap::new()),
            mappings: Mutex::new(HashMap::new()),
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Result<MappedTensor> {
        let shard = self.weight_map.get(name).ok_or_else(|| {
            XlaError::Pjrt(format!("tensor `{name}` is absent from the checkpoint index"))
        })?;
        let entries = self.shard_entries(shard)?;
        let entry = entries.get(name).cloned().ok_or_else(|| XlaError::Pjrt(format!(
            "tensor `{name}` is indexed in {} but absent from its header",
            shard.display()
        )))?;
        let mapping = self.shard_mapping(shard)?;
        Ok(MappedTensor { entry, mapping })
    }

    /// Parses every referenced shard header and validates every indexed tensor
    /// range against its dtype and shape without copying payloads out of mmap.
    pub fn validate_all(&self) -> Result<SafeTensorValidation> {
        let names = self.weight_map.keys().cloned().collect::<Vec<_>>();
        let mut payload_bytes = 0u64;
        let mut bf16_payload_bytes = 0u64;
        for name in &names {
            let tensor = self.get(name)?;
            let bytes = u64::try_from(tensor.bytes().len())
                .map_err(|_| XlaError::Pjrt("tensor payload byte count overflow".into()))?;
            payload_bytes = payload_bytes.checked_add(bytes)
                .ok_or_else(|| XlaError::Pjrt("checkpoint payload byte count overflow".into()))?;
            if tensor.entry.dtype == SafeTensorDType::BF16 {
                bf16_payload_bytes = bf16_payload_bytes.checked_add(bytes)
                    .ok_or_else(|| XlaError::Pjrt("BF16 payload byte count overflow".into()))?;
            }
        }
        let shards = self.weight_map.values().collect::<std::collections::BTreeSet<_>>().len();
        Ok(SafeTensorValidation {
            tensors: names.len(),
            shards,
            payload_bytes,
            bf16_payload_bytes,
        })
    }

    pub fn upload_bf16(
        &self,
        client: &PjrtClient,
        name: &str,
        device: Option<&crate::Device>,
    ) -> Result<Buffer> {
        let tensor = self.get(name)?;
        if tensor.entry.dtype != SafeTensorDType::BF16 {
            return Err(XlaError::Pjrt(format!(
                "tensor `{name}` is {:?}, expected BF16",
                tensor.entry.dtype
            )));
        }
        let host = HostBuffer::from_bf16_bytes(tensor.entry.shape.clone(), tensor.bytes())?;
        client.buffer_from_host(host, device)
    }

    fn shard_mapping(&self, shard: &Path) -> Result<Arc<Mmap>> {
        if let Some(mapping) = self.mappings.lock().unwrap().get(shard).cloned() {
            return Ok(mapping);
        }
        let path = self.model_directory.join(shard);
        let file = File::open(&path)?;
        // SAFETY: the mapping is read-only, owns an independent file-backed VM
        // region, and is retained by Arc for every returned tensor view.
        let mapping = Arc::new(unsafe { MmapOptions::new().map(&file) }.map_err(|error| {
            XlaError::Io(error)
        })?);
        self.mappings
            .lock()
            .unwrap()
            .insert(shard.to_path_buf(), Arc::clone(&mapping));
        Ok(mapping)
    }

    fn shard_entries(&self, shard: &Path) -> Result<Arc<BTreeMap<String, SafeTensorEntry>>> {
        if let Some(entries) = self.entries.lock().unwrap().get(shard).cloned() {
            return Ok(entries);
        }
        let mapping = self.shard_mapping(shard)?;
        if mapping.len() < 8 {
            return Err(XlaError::Pjrt(format!(
                "Safetensors shard {} is shorter than its length prefix",
                shard.display()
            )));
        }
        let header_size = u64::from_le_bytes(mapping[..8].try_into().unwrap());
        let header_size = usize::try_from(header_size)
            .map_err(|_| XlaError::Pjrt("Safetensors header is too large".into()))?;
        let data_start = 8usize.checked_add(header_size)
            .ok_or_else(|| XlaError::Pjrt("Safetensors header size overflow".into()))?;
        if data_start > mapping.len() {
            return Err(XlaError::Pjrt(format!(
                "Safetensors header in {} extends past end of file",
                shard.display()
            )));
        }
        let header: Value = serde_json::from_slice(&mapping[8..data_start]).map_err(|error| {
            XlaError::Pjrt(format!("invalid Safetensors header {}: {error}", shard.display()))
        })?;
        let object = header.as_object().ok_or_else(|| {
            XlaError::Pjrt("Safetensors header root must be an object".into())
        })?;
        let mut entries = BTreeMap::new();
        for (name, metadata) in object {
            if name == "__metadata__" {
                continue;
            }
            let metadata = metadata.as_object().ok_or_else(|| XlaError::Pjrt(format!(
                "metadata for tensor `{name}` is not an object"
            )))?;
            let dtype = SafeTensorDType::parse(
                metadata.get("dtype").and_then(Value::as_str).ok_or_else(|| {
                    XlaError::Pjrt(format!("tensor `{name}` has no dtype"))
                })?,
            )?;
            let shape = metadata.get("shape").and_then(Value::as_array).ok_or_else(|| {
                XlaError::Pjrt(format!("tensor `{name}` has no shape"))
            })?.iter().map(|dimension| {
                dimension.as_i64().filter(|value| *value >= 0).ok_or_else(|| {
                    XlaError::Pjrt(format!("tensor `{name}` has an invalid dimension"))
                })
            }).collect::<Result<Vec<_>>>()?;
            let offsets = metadata.get("data_offsets").and_then(Value::as_array)
                .filter(|offsets| offsets.len() == 2)
                .ok_or_else(|| XlaError::Pjrt(format!("tensor `{name}` has invalid offsets")))?;
            let relative_start = offsets[0].as_u64().and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| XlaError::Pjrt(format!("tensor `{name}` has invalid start offset")))?;
            let relative_end = offsets[1].as_u64().and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| XlaError::Pjrt(format!("tensor `{name}` has invalid end offset")))?;
            let start = data_start.checked_add(relative_start)
                .ok_or_else(|| XlaError::Pjrt("tensor offset overflow".into()))?;
            let end = data_start.checked_add(relative_end)
                .ok_or_else(|| XlaError::Pjrt("tensor offset overflow".into()))?;
            let elements = shape.iter().try_fold(1usize, |count, &dimension| {
                count.checked_mul(dimension as usize)
                    .ok_or_else(|| XlaError::Pjrt("tensor element count overflow".into()))
            })?;
            let expected = elements.checked_mul(dtype.byte_width())
                .ok_or_else(|| XlaError::Pjrt("tensor byte size overflow".into()))?;
            if start > end || end > mapping.len() || end - start != expected {
                return Err(XlaError::Pjrt(format!(
                    "tensor `{name}` byte range does not match its dtype and shape"
                )));
            }
            entries.insert(name.clone(), SafeTensorEntry {
                name: name.clone(), dtype, shape, start, end,
                shard: shard.to_path_buf(),
            });
        }
        let entries = Arc::new(entries);
        self.entries.lock().unwrap().insert(shard.to_path_buf(), Arc::clone(&entries));
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_indexed_bf16_tensor_without_reencoding_payload() {
        let directory = std::env::temp_dir().join(format!(
            "severian-safetensors-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(
            directory.join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        let header = r#"{"model.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut shard = File::create(directory.join("model-00001-of-00001.safetensors")).unwrap();
        shard.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        shard.write_all(header.as_bytes()).unwrap();
        let payload = [0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40];
        shard.write_all(&payload).unwrap();
        drop(shard);

        let store = SafeTensorStore::open(&directory).unwrap();
        let tensor = store.get("model.weight").unwrap();
        assert_eq!(tensor.entry().dtype, SafeTensorDType::BF16);
        assert_eq!(tensor.entry().shape, [2, 2]);
        assert_eq!(tensor.bytes(), payload);

        std::fs::remove_dir_all(directory).unwrap();
    }
}
