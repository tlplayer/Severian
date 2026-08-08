use crate::target::DriverTarget;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel { O0, O1, O2, O3 }

impl Default for OptimizationLevel {
    fn default() -> Self { Self::O2 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitKind { Executable, Mlir, LlvmIr, StableHlo, Object, SharedLibrary }

impl Default for EmitKind {
    fn default() -> Self { Self::Executable }
}

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub target: DriverTarget,
    pub optimization: OptimizationLevel,
    pub emit: EmitKind,
    pub output: Option<PathBuf>,
    pub keep_intermediates: bool,
    pub verify: bool,
    pub run_generic_passes: bool,
    pub run_xla_passes: bool,
    pub run_iree_passes: bool,
    pub debug_symbols: bool,
    pub strip: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            target: DriverTarget::Native,
            optimization: OptimizationLevel::O2,
            emit: EmitKind::Executable,
            output: None,
            keep_intermediates: false,
            verify: true,
            run_generic_passes: true,
            run_xla_passes: true,
            run_iree_passes: true,
            debug_symbols: false,
            strip: false,
        }
    }
}

impl CompileOptions {
    pub fn native() -> Self { Self::default() }

    pub fn xla() -> Self {
        Self {
            target: DriverTarget::Xla { platform: None, device_ordinal: None },
            ..Self::default()
        }
    }

    pub fn rocm(chip: impl Into<String>) -> Self {
        Self {
            target: DriverTarget::Amd {
                architecture: Some(chip.into()),
                device_ordinal: None,
            },
            ..Self::default()
        }
    }

    pub fn cuda(architecture: impl Into<String>) -> Self {
        Self {
            target: DriverTarget::Nvidia {
                architecture: Some(architecture.into()),
                device_ordinal: None,
            },
            ..Self::default()
        }
    }

    pub fn output_path(&self, source: Option<&std::path::Path>) -> PathBuf {
        if let Some(output) = &self.output { return output.clone(); }

        let stem = source
            .and_then(std::path::Path::file_stem)
            .and_then(|stem| stem.to_str())
            .unwrap_or("a");

        let extension = match self.emit {
            EmitKind::Executable => if cfg!(windows) { "exe" } else { "" },
            EmitKind::Mlir | EmitKind::StableHlo => "mlir",
            EmitKind::LlvmIr => "ll",
            EmitKind::Object => "o",
            EmitKind::SharedLibrary => {
                if cfg!(target_os = "macos") { "dylib" }
                else if cfg!(windows) { "dll" }
                else { "so" }
            }
        };

        if extension.is_empty() { PathBuf::from(stem) }
        else { PathBuf::from(format!("{stem}.{extension}")) }
    }
}
