use super::{amd, device::GpuVendor, nvidia, GpuDevice};
use crate::target::{Backend, TargetError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTarget {
    pub vendor: GpuVendor,
    pub architecture: Option<String>,
    pub device_ordinal: Option<usize>,
    pub features: Vec<String>,
}

impl GpuTarget {
    pub fn nvidia(architecture: Option<String>) -> Result<Self, TargetError> {
        let architecture = architecture
            .map(|value| {
                nvidia::normalize_architecture(&value)
                    .ok_or_else(|| TargetError::InvalidGpuTarget(value))
            })
            .transpose()?;

        Ok(Self {
            vendor: GpuVendor::Nvidia,
            architecture,
            device_ordinal: None,
            features: Vec::new(),
        })
    }

    pub fn amd(architecture: Option<String>) -> Result<Self, TargetError> {
        let architecture = architecture
            .map(|value| {
                amd::normalize_architecture(&value)
                    .ok_or_else(|| TargetError::InvalidGpuTarget(value))
            })
            .transpose()?;

        Ok(Self {
            vendor: GpuVendor::Amd,
            architecture,
            device_ordinal: None,
            features: Vec::new(),
        })
    }

    pub fn from_device(device: &GpuDevice) -> Result<Self, TargetError> {
        match device.vendor {
            GpuVendor::Nvidia => {
                let mut target = Self::nvidia(device.architecture.clone())?;
                target.device_ordinal = Some(device.ordinal);
                Ok(target)
            }
            GpuVendor::Amd => {
                let mut target = Self::amd(device.architecture.clone())?;
                target.device_ordinal = Some(device.ordinal);
                Ok(target)
            }
            GpuVendor::Unknown => Err(TargetError::InvalidGpuTarget(device.name.clone())),
        }
    }

    pub const fn backend(&self) -> Backend {
        match self.vendor {
            GpuVendor::Nvidia => Backend::Nvidia,
            GpuVendor::Amd => Backend::Amd,
            GpuVendor::Unknown => Backend::Spirv,
        }
    }

    pub fn llvm_triple(&self) -> String {
        match self.vendor {
            GpuVendor::Nvidia => nvidia::llvm_triple().to_owned(),
            GpuVendor::Amd => amd::llvm_triple().to_owned(),
            GpuVendor::Unknown => "spirv64-unknown-unknown".into(),
        }
    }

    pub fn llvm_cpu(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    pub fn llvm_features(&self) -> String {
        self.features
            .iter()
            .map(|feature| {
                if feature.starts_with('+') || feature.starts_with('-') {
                    feature.clone()
                } else {
                    format!("+{feature}")
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn is_nvidia(&self) -> bool {
        self.vendor == GpuVendor::Nvidia
    }

    pub fn is_amd(&self) -> bool {
        self.vendor == GpuVendor::Amd
    }
}
