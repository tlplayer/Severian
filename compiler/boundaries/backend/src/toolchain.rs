use crate::BackendError;
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tool {
    MlirOpt,
    MlirTranslate,
    Clang,
    Opt,
    Lld,
    Ptxas,
    NvidiaSmi,
    AmdGpuArch,
    RocmInfo,
    SpirvOpt,
    SpirvVal,
    Nm,
}

impl Tool {
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::MlirOpt => "mlir-opt",
            Self::MlirTranslate => "mlir-translate",
            Self::Clang => "clang",
            Self::Opt => "opt",
            Self::Lld => "ld.lld",
            Self::Ptxas => "ptxas",
            Self::NvidiaSmi => "nvidia-smi",
            Self::AmdGpuArch => "amdgpu-arch",
            Self::RocmInfo => "rocminfo",
            Self::SpirvOpt => "spirv-opt",
            Self::SpirvVal => "spirv-val",
            Self::Nm => "nm",
        }
    }

    pub fn candidates(self) -> &'static [&'static str] {
        match self {
            Self::MlirOpt => &["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"],
            Self::MlirTranslate => &[
                "mlir-translate",
                "mlir-translate-21",
                "/usr/lib/llvm-21/bin/mlir-translate",
            ],
            Self::Clang => &["clang", "clang-21", "/usr/bin/clang-21"],
            Self::Opt => &["opt", "opt-21", "/usr/lib/llvm-21/bin/opt"],
            Self::Lld => &[
                "ld.lld",
                "/usr/lib/llvm-21/bin/ld.lld",
                "/opt/rocm/llvm/bin/ld.lld",
            ],
            Self::Ptxas => &["ptxas", "/usr/local/cuda/bin/ptxas", "/opt/cuda/bin/ptxas"],
            Self::NvidiaSmi => &["nvidia-smi", "/usr/bin/nvidia-smi"],
            Self::AmdGpuArch => &[
                "amdgpu-arch",
                "/opt/rocm/llvm/bin/amdgpu-arch",
                "/usr/lib/llvm-21/bin/amdgpu-arch",
            ],
            Self::RocmInfo => &["rocminfo", "/opt/rocm/bin/rocminfo"],
            Self::SpirvOpt => &["spirv-opt"],
            Self::SpirvVal => &["spirv-val"],
            Self::Nm => &["llvm-nm", "nm", "/usr/bin/nm"],
        }
    }
}

pub fn find_tool(tool: Tool) -> Option<PathBuf> {
    find_executable(tool.candidates())
}

pub fn find_required_tool(tool: Tool) -> Result<PathBuf, BackendError> {
    find_tool(tool).ok_or_else(|| {
        BackendError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("required tool `{}` was not found", tool.display_name()),
        ))
    })
}

pub fn find_executable(candidates: &[&str]) -> Option<PathBuf> {
    for candidate in candidates {
        let path = Path::new(candidate);

        if path.components().count() > 1 && path.is_file() {
            return Some(path.to_owned());
        }

        if let Some(paths) = std::env::var_os("PATH") {
            for directory in std::env::split_paths(&paths) {
                let executable = directory.join(candidate);
                if executable.is_file() {
                    return Some(executable);
                }
            }
        }
    }

    None
}

pub fn run_tool(tool: &Path, arguments: &[OsString]) -> Result<(), BackendError> {
    let output = Command::new(tool).args(arguments).output()?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let detail = if stderr.is_empty() { stdout } else { stderr };

    Err(BackendError(std::io::Error::other(format!(
        "{} failed: {detail}",
        tool.display()
    ))))
}

pub fn tool_version(tool: &Path) -> Option<String> {
    Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            let bytes = if output.stdout.is_empty() {
                &output.stderr
            } else {
                &output.stdout
            };

            String::from_utf8_lossy(bytes)
                .lines()
                .next()
                .unwrap_or("unknown version")
                .trim()
                .to_owned()
        })
}

pub struct TemporaryFiles {
    prefix: PathBuf,
    keep: bool,
}

impl TemporaryFiles {
    pub fn new(label: &str) -> Self {
        let prefix = std::env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time must follow the Unix epoch")
                .as_nanos()
        ));

        Self {
            prefix,
            keep: std::env::var_os("SEVERIAN_KEEP_NATIVE_TEMPS").is_some(),
        }
    }

    pub fn path(&self, suffix: &str) -> PathBuf {
        PathBuf::from(format!("{}.{}", self.prefix.display(), suffix))
    }

    pub fn directory(&self, suffix: &str) -> PathBuf {
        PathBuf::from(format!("{}.{}", self.prefix.display(), suffix))
    }
}

impl Drop for TemporaryFiles {
    fn drop(&mut self) {
        if self.keep {
            return;
        }

        let parent = self.prefix.parent().unwrap_or_else(|| Path::new("."));
        let Some(prefix_name) = self.prefix.file_name().and_then(|name| name.to_str()) else {
            return;
        };

        let Ok(entries) = std::fs::read_dir(parent) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let matches = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix_name));

            if matches {
                if path.is_dir() {
                    let _ = std::fs::remove_dir_all(path);
                } else {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
    }
}
