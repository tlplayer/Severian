#![forbid(unsafe_code)]

pub mod coverage;
pub mod debug;
pub mod linker;
pub mod llvm;
pub mod native;
pub mod toolchain;

pub use native::{NativeCompileOptions, NativeSanitizer};

use severian_hir::Program;
use severian_mlir::Module;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolStatus {
    pub name: &'static str,
    pub required: bool,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
    pub compatible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolchainReport {
    pub tools: Vec<ToolStatus>,
    pub sqlite_available: bool,
    pub rocm_hip_library: Option<PathBuf>,
    pub amd_gpu_chip: Option<String>,
}

impl ToolchainReport {
    pub fn native_ready(&self) -> bool {
        self.tools
            .iter()
            .filter(|tool| tool.required)
            .all(|tool| tool.path.is_some() && tool.compatible)
    }
}

/// Inspects the same programs and libraries used by native and ROCm lowering.
/// MLIR 21 is the supported dialect/tool version; optional capabilities do not
/// make the host-native toolchain unhealthy.
pub fn inspect_toolchain() -> ToolchainReport {
    let tools = vec![
        inspect_tool(
            "mlir-opt",
            true,
            &["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"],
            Some(21),
        ),
        inspect_tool(
            "mlir-translate",
            true,
            &[
                "mlir-translate",
                "mlir-translate-21",
                "/usr/lib/llvm-21/bin/mlir-translate",
            ],
            Some(21),
        ),
        inspect_tool(
            "clang",
            true,
            &["clang", "clang-21", "/usr/bin/clang-21"],
            Some(21),
        ),
    ];
    let sqlite_available = find_tool(&["pkg-config"]).is_some_and(|tool| {
        Command::new(tool)
            .args(["--exists", "sqlite3"])
            .status()
            .is_ok_and(|status| status.success())
    });
    ToolchainReport {
        tools,
        sqlite_available,
        rocm_hip_library: find_hip_library(),
        amd_gpu_chip: detect_amd_gpu_chip(),
    }
}

fn inspect_tool(
    name: &'static str,
    required: bool,
    candidates: &[&str],
    supported_major: Option<u32>,
) -> ToolStatus {
    let path = find_tool(candidates);
    let version = path.as_ref().and_then(|path| {
        Command::new(path)
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
    });
    let compatible = supported_major.is_none_or(|major| {
        version
            .as_deref()
            .is_some_and(|version| version_mentions_major(version, major))
    });
    ToolStatus {
        name,
        required,
        path,
        version,
        compatible,
    }
}

fn version_mentions_major(version: &str, major: u32) -> bool {
    version
        .split(|character: char| !character.is_ascii_digit())
        .find_map(|component| component.parse::<u32>().ok())
        == Some(major)
}

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
    native::compile_native(program, module, output, None, &NativeCompileOptions::default())
}

pub fn compile_native_with_xla_runtime(
    program: &Program,
    module: &Module,
    output: &Path,
    xla_runtime: &Path,
) -> Result<(), BackendError> {
    native::compile_native(
        program,
        module,
        output,
        Some(xla_runtime),
        &NativeCompileOptions::default(),
    )
}

pub fn compile_native_with_options(
    program: &Program,
    module: &Module,
    output: &Path,
    options: &NativeCompileOptions,
) -> Result<(), BackendError> {
    native::compile_native(program, module, output, None, options)
}

/// Compiles GPU execution regions to an AMD code object, embeds it in the host
/// executable, and links the MLIR GPU runtime ABI to HIP.
pub fn compile_rocm(
    program: &Program,
    module: &Module,
    output: &Path,
    chip: &str,
) -> Result<(), BackendError> {
    if !is_amd_gpu_chip(chip) {
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid AMD GPU architecture `{chip}`; expected a name such as `gfx1101`"),
        )));
    }
    if !module.as_str().contains("severian_parallel = \"gpu\"") {
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ROCm compilation requires a `with gpu:` execution region",
        )));
    }

    let prefix = temporary_prefix("severian-rocm-compile");
    let source = prefix.with_extension("mlir");
    let device = prefix.with_extension("device.mlir");
    let binary = prefix.with_extension("binary.mlir");
    let host_gpu = prefix.with_extension("host-gpu.mlir");
    let lowered = prefix.with_extension("llvm.mlir");
    let llvm_ir = prefix.with_extension("ll");
    let platform_source = prefix.with_extension("platform.c");
    let toolkit = prepare_rocm_toolkit(&prefix)?;
    let hip_library = find_hip_library().ok_or_else(|| {
        BackendError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "ROCm HIP runtime was not found; set SEVERIAN_ROCM_LIB to libamdhip64.so",
        ))
    })?;
    let hip_directory = hip_library.parent().expect("HIP library has a parent");
    let mlir_opt = find_tool(&["mlir-opt", "mlir-opt-21", "/usr/lib/llvm-21/bin/mlir-opt"])
        .ok_or_else(|| missing_tool("mlir-opt"))?;
    let translate = find_tool(&[
        "mlir-translate",
        "mlir-translate-21",
        "/usr/lib/llvm-21/bin/mlir-translate",
    ])
    .ok_or_else(|| missing_tool("mlir-translate"))?;
    let clang = find_tool(&["clang", "clang-21", "/usr/bin/clang-21"])
        .ok_or_else(|| missing_tool("clang"))?;
    let target = PathBuf::from(format!("--rocdl-attach-target=chip={chip}"));
    let binary_pass = PathBuf::from(format!(
        "--gpu-module-to-binary=toolkit={}",
        toolkit.display()
    ));
    let rpath = PathBuf::from(format!("-Wl,-rpath,{}", hip_directory.display()));

    let result = (|| {
        std::fs::write(&source, module.as_str())?;
        run_tool(
            mlir_opt.clone(),
            &[
                source.as_path(),
                Path::new("--convert-linalg-to-parallel-loops"),
                Path::new("--gpu-map-parallel-loops"),
                Path::new("--convert-parallel-loops-to-gpu"),
                Path::new("--gpu-kernel-outlining"),
                target.as_path(),
                Path::new("--lower-affine"),
                Path::new("--convert-scf-to-cf"),
                Path::new("--convert-index-to-llvm"),
                Path::new("--convert-math-to-rocdl"),
                Path::new("--convert-arith-to-llvm"),
                Path::new("--convert-gpu-to-rocdl=runtime=HIP"),
                Path::new("--reconcile-unrealized-casts"),
                Path::new("-o"),
                device.as_path(),
            ],
        )?;
        run_tool(
            mlir_opt.clone(),
            &[
                device.as_path(),
                binary_pass.as_path(),
                Path::new("-o"),
                binary.as_path(),
            ],
        )?;
        run_tool(
            mlir_opt.clone(),
            &[
                binary.as_path(),
                Path::new("--gpu-to-llvm"),
                Path::new("-o"),
                host_gpu.as_path(),
            ],
        )?;
        run_tool(
            mlir_opt,
            &[
                host_gpu.as_path(),
                Path::new("--lower-affine"),
                Path::new("--convert-scf-to-cf"),
                Path::new("--convert-index-to-llvm"),
                Path::new("--convert-math-to-llvm"),
                Path::new("--convert-arith-to-llvm"),
                Path::new("--finalize-memref-to-llvm"),
                Path::new("--convert-func-to-llvm"),
                Path::new("--reconcile-unrealized-casts"),
                Path::new("-o"),
                lowered.as_path(),
            ],
        )?;
        run_tool(
            translate,
            &[
                Path::new("--mlir-to-llvmir"),
                lowered.as_path(),
                Path::new("-o"),
                llvm_ir.as_path(),
            ],
        )?;
        let bridge = severian_lowering::rocm_bridge_source(program)
            .map_err(|error| BackendError(std::io::Error::other(error)))?;
        std::fs::write(&platform_source, bridge)?;
        run_tool(
            clang,
            &[
                llvm_ir.as_path(),
                platform_source.as_path(),
                hip_library.as_path(),
                rpath.as_path(),
                Path::new("-o"),
                output,
                Path::new("-lm"),
                Path::new("-pthread"),
            ],
        )
    })();

    if std::env::var_os("SEVERIAN_KEEP_NATIVE_TEMPS").is_none() {
        for temporary in [
            &source,
            &device,
            &binary,
            &host_gpu,
            &lowered,
            &llvm_ir,
            &platform_source,
        ] {
            let _ = std::fs::remove_file(temporary);
        }
        if toolkit.starts_with(&prefix) {
            let _ = std::fs::remove_dir_all(toolkit);
        }
    }
    result
}

/// Outlines parallel linalg kernels and lowers their device bodies to AMD's
/// ROCDL dialect. This inspection form is also the device-lowering prefix used
/// by `compile_rocm` before code-object serialization and HIP linking.
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
                Path::new("--convert-math-to-rocdl"),
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
    if let Some(tool) = find_tool(&["lspci", "/usr/bin/lspci"]) {
        if let Ok(output) = Command::new(tool).arg("-nn").output() {
            if output.status.success() {
                let devices = String::from_utf8_lossy(&output.stdout);
                for (needle, chip) in [
                    ("Navi 31", "gfx1100"),
                    ("[1002:744c]", "gfx1100"),
                    ("Navi 32", "gfx1101"),
                    ("[1002:747e]", "gfx1101"),
                    ("Navi 33", "gfx1102"),
                    ("[1002:7480]", "gfx1102"),
                ] {
                    if devices.contains(needle) {
                        return Some(chip.to_owned());
                    }
                }
            }
        }
    }
    None
}

fn prepare_rocm_toolkit(prefix: &Path) -> Result<PathBuf, BackendError> {
    if let Some(path) = std::env::var_os("SEVERIAN_ROCM_TOOLKIT").map(PathBuf::from) {
        if path.join("llvm/bin/ld.lld").is_file() {
            return Ok(path);
        }
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "SEVERIAN_ROCM_TOOLKIT={} has no llvm/bin/ld.lld",
                path.display()
            ),
        )));
    }
    let standard = PathBuf::from("/opt/rocm");
    if standard.join("llvm/bin/ld.lld").is_file() {
        return Ok(standard);
    }
    let lld = find_tool(&[
        "ld.lld",
        "/usr/lib/llvm-21/bin/ld.lld",
        "/usr/lib/llvm-21/bin/lld",
    ])
    .ok_or_else(|| missing_tool("ld.lld"))?;
    let toolkit = prefix.with_extension("rocm-toolkit");
    let llvm_bin = toolkit.join("llvm/bin");
    std::fs::create_dir_all(&llvm_bin)?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(lld, llvm_bin.join("ld.lld"))?;
    #[cfg(not(unix))]
    std::fs::copy(lld, llvm_bin.join("ld.lld"))?;
    Ok(toolkit)
}

fn find_hip_library() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SEVERIAN_ROCM_LIB").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    [
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib64/libamdhip64.so",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
        "/usr/local/lib/ollama/rocm_v7_2/libamdhip64.so.7",
    ]
    .iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
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

#[cfg(test)]
mod tests {
    use super::version_mentions_major;

    #[test]
    fn recognizes_tool_version_major_without_matching_minor_numbers() {
        assert!(version_mentions_major("Ubuntu LLVM version 21.1.8", 21));
        assert!(version_mentions_major("clang version 21.0.0", 21));
        assert!(!version_mentions_major("LLVM version 20.1.21", 21));
    }
}
