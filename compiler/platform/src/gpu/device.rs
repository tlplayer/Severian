#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    pub ordinal: usize,
    pub vendor: GpuVendor,
    pub name: String,
    /// NVIDIA `sm_*` or AMD `gfx*` target architecture when known.
    pub architecture: Option<String>,
    pub memory_bytes: Option<u64>,
    pub pci_bus_id: Option<String>,
    pub runtime: Option<String>,
}

impl GpuDevice {
    pub fn is_nvidia(&self) -> bool {
        self.vendor == GpuVendor::Nvidia
    }

    pub fn is_amd(&self) -> bool {
        self.vendor == GpuVendor::Amd
    }
}
