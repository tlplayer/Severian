use crate::BackendError;
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone)]
pub struct CoverageBackendOptions {
    pub enabled: bool,
    pub profile_runtime: Option<PathBuf>,
    pub profile_pattern: Option<String>,
}

impl Default for CoverageBackendOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            profile_runtime: None,
            profile_pattern: None,
        }
    }
}

impl CoverageBackendOptions {
    pub fn link_arguments(&self) -> Vec<OsString> {
        if !self.enabled {
            return Vec::new();
        }

        self.profile_runtime
            .as_ref()
            .map(|runtime| vec![runtime.as_os_str().to_owned()])
            .unwrap_or_default()
    }

    pub fn environment(&self) -> Vec<(String, String)> {
        self.profile_pattern
            .as_ref()
            .map(|pattern| vec![("LLVM_PROFILE_FILE".into(), pattern.clone())])
            .unwrap_or_default()
    }
}

pub fn discover_profile_runtime(clang: &Path) -> Result<PathBuf, BackendError> {
    let output = Command::new(clang)
        .arg("--print-file-name=libclang_rt.profile-x86_64.a")
        .output()?;

    if !output.status.success() {
        return Err(BackendError(std::io::Error::other(format!(
            "{} could not locate the LLVM profile runtime: {}",
            clang.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ))));
    }

    let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    if !path.is_file() {
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Clang reported profile runtime {}, but it does not exist",
                path.display()
            ),
        )));
    }

    Ok(path)
}

pub fn verify_instrumented_binary(binary: &Path) -> Result<(), BackendError> {
    let output = Command::new("nm").arg(binary).output()?;

    if !output.status.success() {
        return Err(BackendError(std::io::Error::other(
            "nm failed while verifying coverage instrumentation",
        )));
    }

    let symbols = String::from_utf8_lossy(&output.stdout);
    if !symbols.contains("__llvm_profile") {
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "binary does not contain LLVM profile instrumentation symbols",
        )));
    }

    Ok(())
}
