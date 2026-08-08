use crate::{cpu::CpuTarget, gpu::GpuTarget};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetKind {
    Cpu,
    Gpu,
    Xla,
    Spirv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Native,
    Llvm,
    Xla,
    Nvidia,
    Amd,
    Spirv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    X86_64,
    Aarch64,
    Riscv64,
    Wasm32,
    Wasm64,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatingSystem {
    Linux,
    Windows,
    MacOs,
    FreeBsd,
    Android,
    Ios,
    Wasi,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Cpu(CpuTarget),
    Gpu(GpuTarget),
    Xla {
        platform: Option<String>,
        device_ordinal: Option<usize>,
    },
    Spirv {
        environment: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetError {
    Empty,
    UnknownTarget(String),
    InvalidDeviceOrdinal(String),
    InvalidGpuTarget(String),
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "target specification is empty"),
            Self::UnknownTarget(value) => write!(f, "unknown target `{value}`"),
            Self::InvalidDeviceOrdinal(value) => {
                write!(f, "invalid device ordinal `{value}`")
            }
            Self::InvalidGpuTarget(value) => write!(f, "invalid GPU target `{value}`"),
        }
    }
}

impl std::error::Error for TargetError {}

impl Target {
    pub fn native() -> Self {
        Self::Cpu(CpuTarget::detect())
    }

    pub fn parse(specification: &str) -> Result<Self, TargetError> {
        specification.parse()
    }

    pub fn kind(&self) -> TargetKind {
        match self {
            Self::Cpu(_) => TargetKind::Cpu,
            Self::Gpu(_) => TargetKind::Gpu,
            Self::Xla { .. } => TargetKind::Xla,
            Self::Spirv { .. } => TargetKind::Spirv,
        }
    }

    pub fn backend(&self) -> Backend {
        match self {
            Self::Cpu(cpu) => cpu.backend,
            Self::Gpu(gpu) => gpu.backend(),
            Self::Xla { .. } => Backend::Xla,
            Self::Spirv { .. } => Backend::Spirv,
        }
    }

    pub fn is_accelerator(&self) -> bool {
        matches!(self, Self::Gpu(_) | Self::Xla { .. } | Self::Spirv { .. })
    }

    pub fn llvm_triple(&self) -> Option<String> {
        match self {
            Self::Cpu(cpu) => Some(cpu.llvm_triple.clone()),
            Self::Gpu(gpu) => Some(gpu.llvm_triple()),
            Self::Xla { .. } | Self::Spirv { .. } => None,
        }
    }
}

impl FromStr for Target {
    type Err = TargetError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() {
            return Err(TargetError::Empty);
        }

        let (head, argument) = value
            .split_once(':')
            .map_or((value, None), |(head, argument)| (head, Some(argument)));

        match head.to_ascii_lowercase().as_str() {
            "native" | "cpu" => Ok(Self::native()),

            "llvm" => {
                let mut cpu = CpuTarget::detect();
                cpu.backend = Backend::Llvm;
                Ok(Self::Cpu(cpu))
            }

            "xla" => {
                let (platform, ordinal) = parse_xla_argument(argument)?;
                Ok(Self::Xla {
                    platform,
                    device_ordinal: ordinal,
                })
            }

            "nvidia" | "cuda" | "nvptx" => {
                let architecture = argument.map(str::to_owned);
                Ok(Self::Gpu(GpuTarget::nvidia(architecture)?))
            }

            "amd" | "rocm" | "amdgpu" => {
                let architecture = argument.map(str::to_owned);
                Ok(Self::Gpu(GpuTarget::amd(architecture)?))
            }

            "spirv" | "vulkan" => Ok(Self::Spirv {
                environment: argument.map(str::to_owned),
            }),

            _ => Err(TargetError::UnknownTarget(value.to_owned())),
        }
    }
}

fn parse_xla_argument(
    argument: Option<&str>,
) -> Result<(Option<String>, Option<usize>), TargetError> {
    let Some(argument) = argument.filter(|value| !value.is_empty()) else {
        return Ok((None, None));
    };

    if let Some((platform, ordinal)) = argument.split_once('@') {
        let ordinal = ordinal
            .parse::<usize>()
            .map_err(|_| TargetError::InvalidDeviceOrdinal(ordinal.to_owned()))?;
        return Ok((Some(platform.to_owned()), Some(ordinal)));
    }

    if let Ok(ordinal) = argument.parse::<usize>() {
        return Ok((None, Some(ordinal)));
    }

    Ok((Some(argument.to_owned()), None))
}

pub const fn host_architecture() -> Architecture {
    if cfg!(target_arch = "x86_64") {
        Architecture::X86_64
    } else if cfg!(target_arch = "aarch64") {
        Architecture::Aarch64
    } else if cfg!(target_arch = "riscv64") {
        Architecture::Riscv64
    } else if cfg!(target_arch = "wasm32") {
        Architecture::Wasm32
    } else if cfg!(target_arch = "wasm64") {
        Architecture::Wasm64
    } else {
        Architecture::Other
    }
}

pub const fn host_operating_system() -> OperatingSystem {
    if cfg!(target_os = "linux") {
        OperatingSystem::Linux
    } else if cfg!(target_os = "windows") {
        OperatingSystem::Windows
    } else if cfg!(target_os = "macos") {
        OperatingSystem::MacOs
    } else if cfg!(target_os = "freebsd") {
        OperatingSystem::FreeBsd
    } else if cfg!(target_os = "android") {
        OperatingSystem::Android
    } else if cfg!(target_os = "ios") {
        OperatingSystem::Ios
    } else if cfg!(target_os = "wasi") {
        OperatingSystem::Wasi
    } else {
        OperatingSystem::Other
    }
}
