use super::{
    compile::RawClient,
    devices::{RawDevice, RawDeviceInfo},
};
use crate::Result;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    NvidiaGpu,
    AmdGpu,
    Tpu,
    Accelerator,
    Unknown,
}

#[derive(Clone)]
pub struct Device {
    raw: RawDevice,
    _client: Arc<RawClient>,
    pub id: i32,
    pub process_index: i32,
    pub local_hardware_id: Option<i32>,
    pub kind: DeviceKind,
    pub platform: String,
    pub description: String,
    pub addressable: bool,
}

impl Device {
    pub(crate) fn from_raw(raw: RawDevice, client: Arc<RawClient>) -> Result<Self> {
        let RawDeviceInfo {
            id,
            process_index,
            local_hardware_id,
            kind,
            addressable,
            ..
        } = raw.info(&client)?;
        let normalized_kind = kind.to_ascii_lowercase();
        let device_kind = if normalized_kind.contains("cpu") {
            DeviceKind::Cpu
        } else if normalized_kind.contains("cuda") || normalized_kind.contains("nvidia") {
            DeviceKind::NvidiaGpu
        } else if normalized_kind.contains("rocm") || normalized_kind.contains("amd") {
            DeviceKind::AmdGpu
        } else if normalized_kind.contains("tpu") {
            DeviceKind::Tpu
        } else if normalized_kind.contains("gpu") || normalized_kind.contains("accelerator") {
            DeviceKind::Accelerator
        } else {
            DeviceKind::Unknown
        };
        Ok(Self {
            raw,
            _client: Arc::clone(&client),
            id,
            process_index,
            local_hardware_id,
            kind: device_kind,
            platform: client.platform_name()?,
            description: kind,
            addressable,
        })
    }

    pub(crate) fn raw(&self) -> RawDevice {
        self.raw
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self.kind, DeviceKind::NvidiaGpu | DeviceKind::AmdGpu)
    }

    pub fn is_cpu(&self) -> bool {
        self.kind == DeviceKind::Cpu
    }

    pub fn is_accelerator(&self) -> bool {
        !matches!(self.kind, DeviceKind::Cpu | DeviceKind::Unknown)
    }
}
