#![forbid(unsafe_code)]

use std::collections::BTreeSet;

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
            devices: vec![Device {
                name: "host-cpu".into(),
                kind: DeviceKind::Cpu,
                architecture: std::env::consts::ARCH.into(),
                features: FeatureSet::default(),
            }],
            capabilities: CapabilitySet::default(),
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
}
