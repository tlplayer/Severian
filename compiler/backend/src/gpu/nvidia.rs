use crate::{
    linker::{link_native_executable, NativeLinkOptions},
    llvm::{llvm_lowering_passes, translate_llvm_dialect_to_ir, LlvmLoweringOptions},
    toolchain::{find_required_tool, run_tool, TemporaryFiles, Tool},
    BackendError,
};
use severian_hir::Program;
use severian_mlir::Module;
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

pub fn compile_cuda(
    program: &Program,
    module: &Module,
    output: &Path,
    architecture: &str,
) -> Result<(), BackendError> {
    validate_architecture(architecture)?;

    if !module.as_str().contains("severian_parallel = \"gpu\"") {
        return Err(invalid_input(
            "CUDA compilation requires a `with gpu:` execution region",
        ));
    }

    let temporary = TemporaryFiles::new("severian-cuda");
    let source = temporary.path("source.mlir");
    let nvvm = temporary.path("nvvm.mlir");
    let binary = temporary.path("binary.mlir");
    let host_gpu = temporary.path("host-gpu.mlir");
    let lowered = temporary.path("llvm.mlir");
    let llvm_ir = temporary.path("module.ll");
    let bridge_source = temporary.path("runtime.c");

    std::fs::write(&source, module.as_str())?;
    lower_device_to_nvvm(&source, &nvvm, architecture)?;

    let mlir_opt = find_required_tool(Tool::MlirOpt)?;

    run_tool(
        &mlir_opt,
        &[
            nvvm.as_os_str().to_owned(),
            "--gpu-module-to-binary".into(),
            "-o".into(),
            binary.as_os_str().to_owned(),
        ],
    )?;

    run_tool(
        &mlir_opt,
        &[
            binary.as_os_str().to_owned(),
            "--gpu-to-llvm".into(),
            "-o".into(),
            host_gpu.as_os_str().to_owned(),
        ],
    )?;

    let mut arguments = vec![host_gpu.as_os_str().to_owned()];
    arguments.extend(
        llvm_lowering_passes(&LlvmLoweringOptions::host_after_gpu())
            .into_iter()
            .map(OsString::from),
    );
    arguments.extend(["-o".into(), lowered.as_os_str().to_owned()]);
    run_tool(&mlir_opt, &arguments)?;

    translate_llvm_dialect_to_ir(&lowered, &llvm_ir)?;

    // Until lowering exposes a CUDA-specific managed-memory bridge, use the
    // regular runtime bridge. MLIR's GPU runtime handles kernel loading and
    // launches independently.
    let bridge = severian_lowering::native_bridge_source(program);
    let bridge_path = if bridge.is_empty() {
        None
    } else {
        std::fs::write(&bridge_source, bridge)?;
        Some(bridge_source.as_path())
    };

    let cuda_runtime = find_cuda_runtime_library();

    let mut link = NativeLinkOptions {
        math: true,
        pthread: bridge_path.is_some(),
        ..NativeLinkOptions::default()
    };

    if let Some(runtime) = cuda_runtime {
        if let Some(directory) = runtime.parent() {
            link.rpaths.push(directory.to_owned());
        }
        link.libraries.push(runtime);
    }

    link_native_executable(&llvm_ir, bridge_path, output, &link)
}

pub fn lower_to_nvvm(
    module: &Module,
    architecture: &str,
) -> Result<Module, BackendError> {
    validate_architecture(architecture)?;

    if !module.as_str().contains("severian_parallel = \"gpu\"") {
        return Err(invalid_input(
            "NVVM lowering requires a `with gpu:` execution region",
        ));
    }

    let temporary = TemporaryFiles::new("severian-nvvm");
    let source = temporary.path("source.mlir");
    let lowered = temporary.path("nvvm.mlir");

    std::fs::write(&source, module.as_str())?;
    lower_device_to_nvvm(&source, &lowered, architecture)?;

    Ok(Module::new(std::fs::read_to_string(lowered)?))
}

fn lower_device_to_nvvm(
    source: &Path,
    output: &Path,
    architecture: &str,
) -> Result<(), BackendError> {
    let mlir_opt = find_required_tool(Tool::MlirOpt)?;

    run_tool(
        &mlir_opt,
        &[
            source.as_os_str().to_owned(),
            "--convert-linalg-to-parallel-loops".into(),
            "--gpu-map-parallel-loops".into(),
            "--convert-parallel-loops-to-gpu".into(),
            "--gpu-kernel-outlining".into(),
            format!("--nvvm-attach-target=chip={architecture} O=3").into(),
            "--lower-affine".into(),
            "--convert-scf-to-cf".into(),
            "--convert-index-to-llvm".into(),
            "--convert-math-to-llvm".into(),
            "--convert-arith-to-llvm".into(),
            "--convert-gpu-to-nvvm".into(),
            "--reconcile-unrealized-casts".into(),
            "-o".into(),
            output.as_os_str().to_owned(),
        ],
    )
}

pub fn detect_nvidia_gpu_architecture() -> Option<String> {
    if let Ok(architecture) = std::env::var("SEVERIAN_NVIDIA_ARCH") {
        let architecture = normalize_architecture(architecture.trim())?;
        return Some(architecture);
    }

    let tool = crate::toolchain::find_tool(Tool::NvidiaSmi)?;
    let output = Command::new(tool)
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .find_map(normalize_architecture)
}

pub fn normalize_architecture(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();

    if valid_architecture(&value) {
        return Some(value);
    }

    let digits = value.replace('.', "");
    if !digits.is_empty() && digits.chars().all(|character| character.is_ascii_digit()) {
        let architecture = format!("sm_{digits}");
        if valid_architecture(&architecture) {
            return Some(architecture);
        }
    }

    None
}

fn validate_architecture(value: &str) -> Result<(), BackendError> {
    if valid_architecture(value) {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "invalid NVIDIA GPU architecture `{value}`; expected sm_*"
        )))
    }
}

fn valid_architecture(value: &str) -> bool {
    let Some(value) = value.strip_prefix("sm_") else {
        return false;
    };
    let value = value.strip_suffix('a').unwrap_or(value);
    !value.is_empty() && value.chars().all(|character| character.is_ascii_digit())
}

fn find_cuda_runtime_library() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SEVERIAN_CUDA_RUNTIME").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }

    [
        "/usr/local/cuda/lib64/libcudart.so",
        "/opt/cuda/lib64/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcudart.so",
    ]
    .iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

fn invalid_input(message: impl Into<String>) -> BackendError {
    BackendError(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}
