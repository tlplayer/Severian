use crate::{Result, XlaError};
use std::{fmt, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableHloFormat {
    Text,
    MlirBytecode,
    PortableArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableHloVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl StableHloVersion {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim().trim_start_matches('v');
        let mut parts = value.split('.');

        let major = parts
            .next()
            .ok_or_else(|| XlaError::InvalidStableHlo("missing version major".into()))?
            .parse()
            .map_err(|_| XlaError::InvalidStableHlo(format!("invalid version: {value}")))?;
        let minor = parts
            .next()
            .ok_or_else(|| XlaError::InvalidStableHlo("missing version minor".into()))?
            .parse()
            .map_err(|_| XlaError::InvalidStableHlo(format!("invalid version: {value}")))?;
        let patch = parts
            .next()
            .unwrap_or("0")
            .parse()
            .map_err(|_| XlaError::InvalidStableHlo(format!("invalid version: {value}")))?;

        Ok(Self {
            major,
            minor,
            patch,
        })
    }
}

impl fmt::Display for StableHloVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone)]
pub struct StableHloModule {
    bytes: Vec<u8>,
    format: StableHloFormat,
    target_version: Option<StableHloVersion>,
    source_name: Option<PathBuf>,
}

impl StableHloModule {
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            bytes: text.into().into_bytes(),
            format: StableHloFormat::Text,
            target_version: None,
            source_name: None,
        }
    }

    pub fn from_bytes(bytes: Vec<u8>, format: StableHloFormat) -> Self {
        Self {
            bytes,
            format,
            target_version: None,
            source_name: None,
        }
    }

    pub fn with_target_version(mut self, version: StableHloVersion) -> Self {
        self.target_version = Some(version);
        self
    }

    pub fn with_source_name(mut self, path: impl Into<PathBuf>) -> Self {
        self.source_name = Some(path.into());
        self
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn format(&self) -> StableHloFormat {
        self.format
    }

    pub fn target_version(&self) -> Option<&StableHloVersion> {
        self.target_version.as_ref()
    }

    pub fn source_name(&self) -> Option<&PathBuf> {
        self.source_name.as_ref()
    }

    pub fn text(&self) -> Result<&str> {
        if self.format != StableHloFormat::Text {
            return Err(XlaError::InvalidStableHlo(
                "StableHLO module is not textual MLIR".into(),
            ));
        }

        std::str::from_utf8(&self.bytes)
            .map_err(|error| XlaError::InvalidStableHlo(error.to_string()))
    }

    pub fn validate_basic(&self) -> Result<()> {
        if self.bytes.is_empty() {
            return Err(XlaError::InvalidStableHlo(
                "module contains no data".into(),
            ));
        }

        if self.format == StableHloFormat::Text {
            let text = self.text()?;
            if !text.contains("stablehlo.") && !text.contains("module") {
                return Err(XlaError::InvalidStableHlo(
                    "text does not appear to contain StableHLO/MLIR".into(),
                ));
            }
        }

        Ok(())
    }
}
