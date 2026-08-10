use std::{fmt, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverTarget {
    Native,
    Xla { platform: Option<String>, device_ordinal: Option<usize> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFamily { Native, Xla }

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
            Self::Xla { .. } => BackendFamily::Xla,
        }
    }

    pub fn is_xla(&self) -> bool { matches!(self, Self::Xla { .. }) }

    pub fn device_ordinal(&self) -> Option<usize> {
        match self {
            Self::Xla { device_ordinal, .. } => *device_ordinal,
            _ => None,
        }
    }

    pub fn platform_target(&self) -> Result<severian_platform::Target, TargetParseError> {
        let specification = match self {
            Self::Native => "native".to_string(),
            Self::Xla { platform, device_ordinal } => match (platform, device_ordinal) {
                (Some(platform), Some(device)) => format!("xla:{platform}@{device}"),
                (Some(platform), None) => format!("xla:{platform}"),
                (None, Some(device)) => format!("xla:{device}"),
                (None, None) => "xla".into(),
            },
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
            "xla" => {
                let (platform, device_ordinal) = parse_platform_device(tail)?;
                Ok(Self::Xla { platform, device_ordinal })
            }
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
