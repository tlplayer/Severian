//! Severian ↔ OpenXLA integration.
//!
//! The crate boundary is intentionally small:
//! - Severian lowering produces StableHLO.
//! - `stablehlo` owns serialization/import/export.
//! - `pipeline` owns compile policy.
//! - `pjrt` owns device, buffer and executable runtime abstractions.
//! - `client` exposes the end-to-end interface used by the compiler/runtime.
//!
//! Raw PJRT C ABI details remain isolated inside crate-private `pjrt` modules.

pub mod client;
pub mod executable_cache;
pub mod pipeline;
pub mod pjrt;
mod runtime;
pub mod safetensors;
pub mod stablehlo;
mod tokenizer;

pub use client::XlaClient;
pub use executable_cache::ExecutableCache;
pub use pipeline::{CompileOptions, XlaPipeline};
pub use pjrt::{Buffer, Device, HostBuffer, LoadedExecutable, PjrtClient, PjrtPlugin};
pub use safetensors::{SafeTensorDType, SafeTensorEntry, SafeTensorStore, SafeTensorValidation};
pub use stablehlo::{StableHloFormat, StableHloModule};

use std::{fmt, io};

#[derive(Debug)]
pub enum XlaError {
    Io(io::Error),
    InvalidStableHlo(String),
    StableHloTool(String),
    PluginLoad(String),
    Pjrt(String),
    Compilation(String),
    Unsupported(String),
}

impl fmt::Display for XlaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::InvalidStableHlo(message) => write!(f, "invalid StableHLO: {message}"),
            Self::StableHloTool(message) => write!(f, "StableHLO tool error: {message}"),
            Self::PluginLoad(message) => write!(f, "PJRT plugin load error: {message}"),
            Self::Pjrt(message) => write!(f, "PJRT error: {message}"),
            Self::Compilation(message) => write!(f, "XLA compilation error: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported XLA feature: {message}"),
        }
    }
}

impl std::error::Error for XlaError {}

impl From<io::Error> for XlaError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, XlaError>;
