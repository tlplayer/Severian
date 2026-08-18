use crate::{
    toolchain::{find_required_tool, run_tool, Tool},
    BackendError,
};
use std::{ffi::OsString, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugEmission {
    None,
    LineTables,
    Full,
}

#[derive(Debug, Clone)]
pub struct DebugBackendOptions {
    pub emission: DebugEmission,
    pub strip: bool,
}

impl Default for DebugBackendOptions {
    fn default() -> Self {
        Self {
            emission: DebugEmission::LineTables,
            strip: false,
        }
    }
}

impl DebugBackendOptions {
    pub fn clang_arguments(&self) -> Vec<OsString> {
        let mut arguments = Vec::new();

        match self.emission {
            DebugEmission::None => {}
            DebugEmission::LineTables => arguments.push("-gline-tables-only".into()),
            DebugEmission::Full => arguments.push("-g".into()),
        }

        if self.strip {
            arguments.push("-s".into());
        }

        arguments
    }

    pub fn enabled(&self) -> bool {
        self.emission != DebugEmission::None
    }
}

pub fn materialize_debug_scopes(input: &Path, output: &Path) -> Result<(), BackendError> {
    let mlir_opt = find_required_tool(Tool::MlirOpt)?;

    run_tool(
        &mlir_opt,
        &[
            input.as_os_str().to_owned(),
            "--llvm-di-scope-for-llvm-func".into(),
            "-o".into(),
            output.as_os_str().to_owned(),
        ],
    )
}
