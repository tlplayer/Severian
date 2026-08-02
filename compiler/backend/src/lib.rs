#![forbid(unsafe_code)]

use severian_hir::Program;
use severian_mlir::Module;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub struct BackendError(std::io::Error);

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for BackendError {}

impl From<std::io::Error> for BackendError {
    fn from(error: std::io::Error) -> Self {
        Self(error)
    }
}

/// Lowers verified MLIR through the host LLVM toolchain and links the native
/// platform provider required by the program.
pub fn compile_native(
    program: &Program,
    module: &Module,
    output: &Path,
) -> Result<(), BackendError> {
    let prefix = std::env::temp_dir().join(format!(
        "severian-compile-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos()
    ));
    let source_mlir = prefix.with_extension("mlir");
    let checked_mlir = prefix.with_extension("checked.mlir");
    let llvm_ir = prefix.with_extension("ll");
    let platform_source = prefix.with_extension("platform.c");

    let result = (|| {
        std::fs::write(&source_mlir, module.as_str())?;
        run_tool(
            find_tool(&["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"])
                .ok_or_else(|| missing_tool("mlir-opt"))?,
            &[
                source_mlir.as_path(),
                Path::new("--convert-linalg-to-loops"),
                Path::new("--lower-affine"),
                Path::new("--convert-scf-to-cf"),
                Path::new("--convert-cf-to-llvm"),
                Path::new("--convert-arith-to-llvm"),
                Path::new("--finalize-memref-to-llvm"),
                Path::new("--convert-func-to-llvm"),
                Path::new("--reconcile-unrealized-casts"),
                Path::new("-o"),
                checked_mlir.as_path(),
            ],
        )?;
        run_tool(
            find_tool(&[
                "mlir-translate",
                "mlir-translate-21",
                "/usr/lib/llvm-21/bin/mlir-translate",
            ])
            .ok_or_else(|| missing_tool("mlir-translate"))?,
            &[
                Path::new("--mlir-to-llvmir"),
                checked_mlir.as_path(),
                Path::new("-o"),
                llvm_ir.as_path(),
            ],
        )?;
        let clang = find_tool(&["clang", "clang-21", "/usr/bin/clang-21"])
            .ok_or_else(|| missing_tool("clang"))?;
        let uses_database = program.functions.iter().any(|function| {
            function
                .native_symbol
                .as_deref()
                .is_some_and(|symbol| symbol.starts_with("__sev_database_"))
        });
        let native_bridge = severian_lowering::native_bridge_source(program);
        if native_bridge.is_empty() {
            run_tool(
                clang,
                &[llvm_ir.as_path(), Path::new("-o"), output, Path::new("-lm")],
            )
        } else {
            std::fs::write(&platform_source, native_bridge)?;
            let mut arguments = vec![
                llvm_ir.as_path(),
                platform_source.as_path(),
                Path::new("-o"),
                output,
                Path::new("-lm"),
                Path::new("-pthread"),
            ];
            if uses_database {
                arguments.push(Path::new("-lsqlite3"));
            }
            run_tool(clang, &arguments)
        }
    })();

    if std::env::var_os("SEVERIAN_KEEP_NATIVE_TEMPS").is_none() {
        for temporary in [&source_mlir, &checked_mlir, &llvm_ir, &platform_source] {
            let _ = std::fs::remove_file(temporary);
        }
    }
    result
}

fn find_tool(candidates: &[&str]) -> Option<PathBuf> {
    for candidate in candidates {
        let path = Path::new(candidate);
        if path.components().count() > 1 && path.is_file() {
            return Some(path.into());
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

fn run_tool(tool: PathBuf, args: &[&Path]) -> Result<(), BackendError> {
    let output = Command::new(&tool).args(args).output()?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    Err(BackendError(std::io::Error::other(format!(
        "{} failed: {stderr}",
        tool.display()
    ))))
}

fn missing_tool(name: &str) -> BackendError {
    BackendError(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("required tool `{name}` was not found"),
    ))
}
