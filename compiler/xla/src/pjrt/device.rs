#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    NvidiaGpu,
    AmdGpu,
    Tpu,
    Accelerator,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySpace {
    pub id: usize,
    pub kind: String,
    pub capacity_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub id: DeviceId,
    pub process_index: usize,
    pub local_hardware_id: usize,
    pub kind: DeviceKind,
    pub platform: String,
    pub description: String,
    pub addressable: bool,
    pub memory_spaces: Vec<MemorySpace>,
}

impl Device {
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
