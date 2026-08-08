use std::{fmt, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverTarget {
    Native,
    Llvm,
    Xla { platform: Option<String>, device_ordinal: Option<usize> },
    Nvidia { architecture: Option<String>, device_ordinal: Option<usize> },
    Amd { architecture: Option<String>, device_ordinal: Option<usize> },
    Spirv { environment: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFamily { Native, Llvm, Xla, Nvidia, Amd, Spirv }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetParseError(pub String);

impl fmt::Display for TargetParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for TargetParseError {}

impl Default for DriverTarget {
    fn default() -> Self { Self::Native }
}

impl DriverTarget {
    pub fn family(&self) -> BackendFamily {
        match self {
            Self::Native => BackendFamily::Native,
            Self::Llvm => BackendFamily::Llvm,
            Self::Xla { .. } => BackendFamily::Xla,
            Self::Nvidia { .. } => BackendFamily::Nvidia,
            Self::Amd { .. } => BackendFamily::Amd,
            Self::Spirv { .. } => BackendFamily::Spirv,
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Nvidia { .. } | Self::Amd { .. } | Self::Spirv { .. })
    }

    pub fn is_xla(&self) -> bool { matches!(self, Self::Xla { .. }) }

    pub fn device_ordinal(&self) -> Option<usize> {
        match self {
            Self::Xla { device_ordinal, .. }
            | Self::Nvidia { device_ordinal, .. }
            | Self::Amd { device_ordinal, .. } => *device_ordinal,
            _ => None,
        }
    }

    pub fn architecture(&self) -> Option<&str> {
        match self {
            Self::Nvidia { architecture, .. } | Self::Amd { architecture, .. } => architecture.as_deref(),
            _ => None,
        }
    }

    pub fn platform_target(&self) -> Result<severian_platform::Target, TargetParseError> {
        let specification = match self {
            Self::Native => "native".to_string(),
            Self::Llvm => "llvm".to_string(),
            Self::Xla { platform, device_ordinal } => match (platform, device_ordinal) {
                (Some(platform), Some(device)) => format!("xla:{platform}@{device}"),
                (Some(platform), None) => format!("xla:{platform}"),
                (None, Some(device)) => format!("xla:{device}"),
                (None, None) => "xla".into(),
            },
            Self::Nvidia { architecture, .. } => architecture
                .as_ref().map(|a| format!("cuda:{a}")).unwrap_or_else(|| "cuda".into()),
            Self::Amd { architecture, .. } => architecture
                .as_ref().map(|a| format!("rocm:{a}")).unwrap_or_else(|| "rocm".into()),
            Self::Spirv { environment } => environment
                .as_ref().map(|e| format!("spirv:{e}")).unwrap_or_else(|| "spirv".into()),
        };

        severian_platform::resolve_target(&specification)
            .map_err(|error| TargetParseError(error.to_string()))
    }
}

impl FromStr for DriverTarget {
    type Err = TargetParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() { return Err(TargetParseError("target cannot be empty".into())); }

        let (head, tail) = value
            .split_once(':')
            .map_or((value, None), |(head, tail)| (head, Some(tail)));

        match head.to_ascii_lowercase().as_str() {
            "native" | "cpu" => Ok(Self::Native),
            "llvm" => Ok(Self::Llvm),
            "xla" => {
                let (platform, device_ordinal) = parse_platform_device(tail)?;
                Ok(Self::Xla { platform, device_ordinal })
            }
            "cuda" | "nvidia" | "nvptx" => {
                let (architecture, device_ordinal) = parse_arch_device(tail)?;
                Ok(Self::Nvidia { architecture, device_ordinal })
            }
            "rocm" | "amd" | "amdgpu" => {
                let (architecture, device_ordinal) = parse_arch_device(tail)?;
                Ok(Self::Amd { architecture, device_ordinal })
            }
            "spirv" | "vulkan" => Ok(Self::Spirv { environment: tail.map(str::to_owned) }),
            _ => Err(TargetParseError(format!("unknown target `{value}`"))),
        }
    }
}

fn parse_platform_device(value: Option<&str>) -> Result<(Option<String>, Option<usize>), TargetParseError> {
    let Some(value) = value.filter(|v| !v.is_empty()) else { return Ok((None, None)); };
    if let Some((platform, device)) = value.split_once('@') {
        let device = device.parse::<usize>()
            .map_err(|_| TargetParseError(format!("invalid device ordinal `{device}`")))?;
        return Ok((Some(platform.to_owned()), Some(device)));
    }
    if let Ok(device) = value.parse::<usize>() { return Ok((None, Some(device))); }
    Ok((Some(value.to_owned()), None))
}

fn parse_arch_device(value: Option<&str>) -> Result<(Option<String>, Option<usize>), TargetParseError> {
    let Some(value) = value.filter(|v| !v.is_empty()) else { return Ok((None, None)); };
    if let Some((architecture, device)) = value.split_once('@') {
        let device = device.parse::<usize>()
            .map_err(|_| TargetParseError(format!("invalid device ordinal `{device}`")))?;
        return Ok((Some(architecture.to_owned()), Some(device)));
    }
    Ok((Some(value.to_owned()), None))
}
