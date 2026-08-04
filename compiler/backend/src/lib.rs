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

/// Outlines parallel linalg kernels and lowers their device bodies to AMD's
/// ROCDL dialect. The result is target-specific MLIR suitable for inspection
/// or a later ROCm object/link step.
pub fn lower_to_rocdl(module: &Module, chip: &str) -> Result<Module, BackendError> {
    if !is_amd_gpu_chip(chip) {
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid AMD GPU architecture `{chip}`; expected a name such as `gfx1100`"),
        )));
    }
    if !module.as_str().contains("severian_parallel = \"gpu\"") {
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ROCm lowering requires a `with gpu:` execution region",
        )));
    }

    let prefix = temporary_prefix("severian-rocdl");
    let source = prefix.with_extension("mlir");
    let lowered = prefix.with_extension("rocdl.mlir");
    let target = format!("--rocdl-attach-target=chip={chip}");
    let result = (|| {
        std::fs::write(&source, module.as_str())?;
        run_tool(
            find_tool(&["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"])
                .ok_or_else(|| missing_tool("mlir-opt"))?,
            &[
                source.as_path(),
                Path::new("--convert-linalg-to-parallel-loops"),
                Path::new("--gpu-map-parallel-loops"),
                Path::new("--convert-parallel-loops-to-gpu"),
                Path::new("--gpu-kernel-outlining"),
                Path::new(&target),
                Path::new("--lower-affine"),
                Path::new("--convert-scf-to-cf"),
                Path::new("--convert-index-to-llvm"),
                Path::new("--convert-arith-to-llvm"),
                Path::new("--finalize-memref-to-llvm"),
                Path::new("--convert-func-to-llvm"),
                Path::new("--convert-gpu-to-rocdl"),
                Path::new("--reconcile-unrealized-casts"),
                Path::new("-o"),
                lowered.as_path(),
            ],
        )?;
        let text = std::fs::read_to_string(&lowered)?;
        if !text.contains("rocdl.target") || !text.contains("gpu.launch_func") {
            return Err(BackendError(std::io::Error::other(
                "the GPU region contains no ranked parallel linalg kernel that can be outlined",
            )));
        }
        Ok(Module::new(text))
    })();
    if std::env::var_os("SEVERIAN_KEEP_NATIVE_TEMPS").is_none() {
        let _ = std::fs::remove_file(source);
        let _ = std::fs::remove_file(lowered);
    }
    result
}

/// Detects the architecture reported by a locally installed ROCm toolchain.
/// `SEVERIAN_AMDGPU_CHIP` is useful on build hosts without GPU device access.
pub fn detect_amd_gpu_chip() -> Option<String> {
    if let Ok(chip) = std::env::var("SEVERIAN_AMDGPU_CHIP") {
        if is_amd_gpu_chip(chip.trim()) {
            return Some(chip.trim().to_owned());
        }
    }
    if let Some(tool) = find_tool(&[
        "amdgpu-arch",
        "/opt/rocm/llvm/bin/amdgpu-arch",
        "/usr/lib/llvm-21/bin/amdgpu-arch",
    ]) {
        if let Ok(output) = Command::new(tool).output() {
            if output.status.success() {
                if let Some(chip) = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(str::trim)
                    .find(|line| is_amd_gpu_chip(line))
                {
                    return Some(chip.to_owned());
                }
            }
        }
    }
    if let Some(tool) = find_tool(&["rocminfo", "/opt/rocm/bin/rocminfo"]) {
        if let Ok(output) = Command::new(tool).output() {
            if output.status.success() {
                if let Some(chip) = String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .map(|word| {
                        word.trim_matches(|character: char| !character.is_ascii_alphanumeric())
                    })
                    .find(|word| is_amd_gpu_chip(word))
                {
                    return Some(chip.to_owned());
                }
            }
        }
    }
    None
}

fn is_amd_gpu_chip(chip: &str) -> bool {
    chip.strip_prefix("gfx").is_some_and(|suffix| {
        suffix.len() >= 3 && suffix.chars().all(|c| c.is_ascii_alphanumeric())
    })
}

fn temporary_prefix(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time must follow the Unix epoch")
            .as_nanos()
    ))
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
