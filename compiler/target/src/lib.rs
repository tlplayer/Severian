#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    X86,
    X86_64,
    Aarch64,
    Wasm32,
    Wasm64,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatingSystem {
    Linux,
    MacOs,
    Windows,
    Wasi,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Accelerator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionBackend {
    Native,
    MlirVector,
    Rocdl,
    StableHloXla,
    Triton,
}

impl ExecutionBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::MlirVector => "mlir-vector",
            Self::Rocdl => "rocdl",
            Self::StableHloXla => "stablehlo-xla",
            Self::Triton => "triton",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetError {
    MissingGpuDevice,
    MissingRocmDriver(String),
}

impl fmt::Display for TargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingGpuDevice => formatter
                .write_str("GPU placement was requested, but no supported AMD GPU was discovered"),
            Self::MissingRocmDriver(device) => write!(
                formatter,
                "GPU `{device}` was discovered, but its ROCm driver/runtime component is missing"
            ),
        }
    }
}

impl std::error::Error for TargetError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub name: String,
    pub kind: DeviceKind,
    pub architecture: String,
    pub features: FeatureSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FeatureSet(BTreeSet<String>);

impl FeatureSet {
    pub fn from_names(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self(names.into_iter().map(Into::into).collect())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CapabilitySet(BTreeSet<String>);

impl CapabilitySet {
    pub fn from_names(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self(names.into_iter().map(Into::into).collect())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
    }

    pub fn insert(&mut self, name: impl Into<String>) {
        self.0.insert(name.into());
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

/// Neutral compiler target authority. Consumers derive ABI layouts, compile
/// routes, dialect choices, and backend strategies from this description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetSpec {
    pub triple: String,
    pub architecture: Architecture,
    pub operating_system: OperatingSystem,
    pub features: FeatureSet,
    pub devices: Vec<Device>,
    pub capabilities: CapabilitySet,
}

impl TargetSpec {
    pub fn host() -> Self {
        let architecture = Architecture::parse(std::env::consts::ARCH);
        let operating_system = OperatingSystem::parse(std::env::consts::OS);
        Self {
            triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            architecture,
            operating_system,
            features: FeatureSet::default(),
            devices: discover_host_devices(Path::new("/dev"), Path::new("/sys/class/drm")),
            capabilities: CapabilitySet::default(),
        }
    }

    pub fn rocm_device(&self) -> Option<&Device> {
        self.devices.iter().find(|device| {
            device.kind == DeviceKind::Gpu
                && device.architecture.starts_with("gfx")
                && device.features.contains("driver.rocm")
        })
    }

    pub fn amd_gpu(&self) -> Option<&Device> {
        self.devices
            .iter()
            .find(|device| device.kind == DeviceKind::Gpu && device.features.contains("vendor.amd"))
    }

    pub fn rediscover_devices(&self) -> Self {
        let mut refreshed = self.clone();
        refreshed.devices = discover_host_devices(Path::new("/dev"), Path::new("/sys/class/drm"));
        refreshed
    }

    pub fn select_execution_backend(
        &self,
        placement: severian_universal::ExecutionPlacement,
    ) -> Result<ExecutionBackend, TargetError> {
        use severian_universal::ExecutionPlacement;
        match placement {
            ExecutionPlacement::Host => Ok(ExecutionBackend::Native),
            ExecutionPlacement::Simd => Ok(ExecutionBackend::MlirVector),
            ExecutionPlacement::Gpu => {
                if self.rocm_device().is_some() {
                    Ok(ExecutionBackend::Rocdl)
                } else if let Some(device) = self.amd_gpu() {
                    Err(TargetError::MissingRocmDriver(device.name.clone()))
                } else {
                    Err(TargetError::MissingGpuDevice)
                }
            }
        }
    }

    pub fn new(triple: impl Into<String>) -> Self {
        let triple = triple.into();
        Self {
            architecture: Architecture::parse(&triple),
            operating_system: OperatingSystem::parse(&triple),
            triple,
            features: FeatureSet::default(),
            devices: Vec::new(),
            capabilities: CapabilitySet::default(),
        }
    }

    pub const fn pointer_bits(&self) -> u16 {
        match self.architecture {
            Architecture::X86 | Architecture::Wasm32 => 32,
            Architecture::X86_64 | Architecture::Aarch64 | Architecture::Wasm64 => 64,
            Architecture::Other => usize::BITS as u16,
        }
    }

    pub const fn machine_integer_bits(&self) -> u16 {
        64
    }

    pub const fn machine_float_bits(&self) -> u16 {
        64
    }
}

fn discover_host_devices(dev: &Path, drm: &Path) -> Vec<Device> {
    let mut devices = vec![Device {
        name: "host-cpu".into(),
        kind: DeviceKind::Cpu,
        architecture: std::env::consts::ARCH.into(),
        features: FeatureSet::default(),
    }];
    let rocm = dev.join("kfd").exists()
        && (Path::new("/usr/bin/rocminfo").exists()
            || Path::new("/opt/rocm/bin/rocminfo").exists());
    let gpu_architecture = rocm_architecture().unwrap_or_else(|| "amdgpu".into());
    let Ok(entries) = std::fs::read_dir(drm) else {
        return devices;
    };
    let mut render_nodes = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            if !name.starts_with("renderD") {
                return None;
            }
            let vendor = std::fs::read_to_string(entry.path().join("device/vendor")).ok()?;
            (vendor.trim().eq_ignore_ascii_case("0x1002")).then_some(name)
        })
        .collect::<Vec<_>>();
    render_nodes.sort();
    for name in render_nodes {
        let mut features = vec!["vendor.amd"];
        if rocm {
            features.extend(["driver.rocm", "mlir.rocdl"]);
        }
        devices.push(Device {
            name,
            kind: DeviceKind::Gpu,
            architecture: gpu_architecture.clone(),
            features: FeatureSet::from_names(features),
        });
    }
    devices
}

fn rocm_architecture() -> Option<String> {
    let program = if Path::new("/usr/bin/rocminfo").exists() {
        "/usr/bin/rocminfo"
    } else if Path::new("/opt/rocm/bin/rocminfo").exists() {
        "/opt/rocm/bin/rocminfo"
    } else {
        return None;
    };
    let output = Command::new(program).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .map(|value| value.trim_matches(|character: char| !character.is_ascii_alphanumeric()))
        .find(|value| {
            value.strip_prefix("gfx").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_alphanumeric())
            })
        })
        .map(str::to_owned)
}

impl Architecture {
    fn parse(value: &str) -> Self {
        if value.contains("aarch64") {
            Self::Aarch64
        } else if value.contains("x86_64") {
            Self::X86_64
        } else if value.contains("wasm64") {
            Self::Wasm64
        } else if value.contains("wasm32") {
            Self::Wasm32
        } else if value.contains("x86") || value.contains("i686") {
            Self::X86
        } else {
            Self::Other
        }
    }
}

impl OperatingSystem {
    fn parse(value: &str) -> Self {
        if value.contains("windows") {
            Self::Windows
        } else if value.contains("darwin") || value.contains("macos") {
            Self::MacOs
        } else if value.contains("wasi") {
            Self::Wasi
        } else if value.contains("linux") {
            Self::Linux
        } else {
            Self::Other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_target_tracks_devices_and_capabilities_without_abi_layout() {
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.capabilities = CapabilitySet::from_names(["mlir.llvm", "xla"]);
        target.devices.push(Device {
            name: "gpu0".into(),
            kind: DeviceKind::Gpu,
            architecture: "gfx1101".into(),
            features: FeatureSet::from_names(["wave32"]),
        });
        assert_eq!(target.pointer_bits(), 64);
        assert!(target.capabilities.contains("xla"));
        assert_eq!(target.devices[0].architecture, "gfx1101");
    }

    #[test]
    fn execution_selection_never_falls_back_from_gpu_to_cpu() {
        let target = TargetSpec::new("x86_64-unknown-linux");
        assert_eq!(
            target.select_execution_backend(severian_universal::ExecutionPlacement::Gpu,),
            Err(TargetError::MissingGpuDevice)
        );
        assert_eq!(
            target
                .select_execution_backend(severian_universal::ExecutionPlacement::Simd,)
                .unwrap(),
            ExecutionBackend::MlirVector
        );
    }

    #[test]
    fn discovered_rocm_device_selects_rocdl() {
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.devices.push(Device {
            name: "renderD128".into(),
            kind: DeviceKind::Gpu,
            architecture: "gfx1100".into(),
            features: FeatureSet::from_names(["vendor.amd", "driver.rocm", "mlir.rocdl"]),
        });
        assert_eq!(
            target
                .select_execution_backend(severian_universal::ExecutionPlacement::Gpu,)
                .unwrap(),
            ExecutionBackend::Rocdl
        );
    }

    #[test]
    fn amd_hardware_without_rocm_requests_provisioning() {
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.devices.push(Device {
            name: "renderD128".into(),
            kind: DeviceKind::Gpu,
            architecture: "gfx1100".into(),
            features: FeatureSet::from_names(["vendor.amd"]),
        });
        assert_eq!(
            target.select_execution_backend(severian_universal::ExecutionPlacement::Gpu),
            Err(TargetError::MissingRocmDriver("renderD128".into()))
        );
    }
}
