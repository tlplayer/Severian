use crate::{
    toolchain::{find_required_tool, run_tool, Tool},
    BackendError,
};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct NativeLinkOptions {
    pub optimization: u8,
    pub math: bool,
    pub pthread: bool,
    pub sqlite: bool,
    pub libraries: Vec<PathBuf>,
    pub library_directories: Vec<PathBuf>,
    pub rpaths: Vec<PathBuf>,
    pub additional_arguments: Vec<OsString>,
}

impl Default for NativeLinkOptions {
    fn default() -> Self {
        Self {
            optimization: 3,
            math: true,
            pthread: false,
            sqlite: false,
            libraries: Vec::new(),
            library_directories: Vec::new(),
            rpaths: Vec::new(),
            additional_arguments: Vec::new(),
        }
    }
}

pub fn link_native_executable(
    llvm_ir: &Path,
    bridge_source: Option<&Path>,
    output: &Path,
    options: &NativeLinkOptions,
) -> Result<(), BackendError> {
    let clang = find_required_tool(Tool::Clang)?;
    let mut arguments = vec![llvm_ir.as_os_str().to_owned()];

    if let Some(bridge_source) = bridge_source {
        arguments.push(bridge_source.as_os_str().to_owned());
    }

    arguments.push(format!("-O{}", options.optimization.min(3)).into());
    arguments.push("-o".into());
    arguments.push(output.as_os_str().to_owned());

    if options.math {
        arguments.push("-lm".into());
    }
    if options.pthread {
        arguments.push("-pthread".into());
    }
    if options.sqlite {
        arguments.push("-lsqlite3".into());
    }

    for directory in &options.library_directories {
        arguments.push(format!("-L{}", directory.display()).into());
    }

    for library in &options.libraries {
        arguments.push(library.as_os_str().to_owned());
    }

    for rpath in &options.rpaths {
        arguments.push(format!("-Wl,-rpath,{}", rpath.display()).into());
    }

    arguments.extend(options.additional_arguments.iter().cloned());
    run_tool(&clang, &arguments)
}

pub fn link_shared_library(
    inputs: &[PathBuf],
    output: &Path,
    options: &NativeLinkOptions,
) -> Result<(), BackendError> {
    let clang = find_required_tool(Tool::Clang)?;
    let mut arguments = inputs
        .iter()
        .map(|path| path.as_os_str().to_owned())
        .collect::<Vec<_>>();

    arguments.extend([
        "-shared".into(),
        "-fPIC".into(),
        format!("-O{}", options.optimization.min(3)).into(),
        "-o".into(),
        output.as_os_str().to_owned(),
    ]);

    for directory in &options.library_directories {
        arguments.push(format!("-L{}", directory.display()).into());
    }
    for library in &options.libraries {
        arguments.push(library.as_os_str().to_owned());
    }
    for rpath in &options.rpaths {
        arguments.push(format!("-Wl,-rpath,{}", rpath.display()).into());
    }

    arguments.extend(options.additional_arguments.iter().cloned());
    run_tool(&clang, &arguments)
}
