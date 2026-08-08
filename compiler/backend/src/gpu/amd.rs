use crate::{
    linker::{link_native_executable, NativeLinkOptions},
    llvm::{translate_llvm_dialect_to_ir, LlvmLoweringOptions},
    toolchain::{find_executable, find_required_tool, run_tool, TemporaryFiles, Tool},
    BackendError,
};
use severian_hir::Program;
use severian_mlir::Module;
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

pub fn compile_rocm(
    program: &Program,
    module: &Module,
    output: &Path,
    chip: &str,
) -> Result<(), BackendError> {
    validate_chip(chip)?;

    if !module.as_str().contains("severian_parallel = \"gpu\"") {
        return Err(invalid_input(
            "ROCm compilation requires a `with gpu:` execution region",
        ));
    }

    let temporary = TemporaryFiles::new("severian-rocm");
    let source = temporary.path("source.mlir");
    let device = temporary.path("device.mlir");
    let binary = temporary.path("binary.mlir");
    let host_gpu = temporary.path("host-gpu.mlir");
    let lowered = temporary.path("llvm.mlir");
    let llvm_ir = temporary.path("module.ll");
    let bridge_source = temporary.path("runtime.c");

    let toolkit = prepare_rocm_toolkit(&temporary)?;
    let hip_library = find_hip_library().ok_or_else(|| {
        BackendError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "ROCm HIP runtime was not found; set SEVERIAN_ROCM_LIB to libamdhip64.so",
        ))
    })?;
    let hip_directory = hip_library
        .parent()
        .expect("HIP library must have a parent")
        .to_owned();

    std::fs::write(&source, module.as_str())?;

    lower_device_to_rocdl(&source, &device, chip)?;

    let mlir_opt = find_required_tool(Tool::MlirOpt)?;
    run_tool(
        &mlir_opt,
        &[
            device.as_os_str().to_owned(),
            format!("--gpu-module-to-binary=toolkit={}", toolkit.display()).into(),
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

    let options = LlvmLoweringOptions::host_after_gpu();
    let mut passes = crate::llvm::llvm_lowering_passes(&options);
    let mut arguments = vec![host_gpu.as_os_str().to_owned()];
    arguments.extend(passes.drain(..).map(OsString::from));
    arguments.extend(["-o".into(), lowered.as_os_str().to_owned()]);
    run_tool(&mlir_opt, &arguments)?;

    translate_llvm_dialect_to_ir(&lowered, &llvm_ir)?;

    std::fs::write(
        &bridge_source,
        severian_lowering::rocm_bridge_source(program),
    )?;

    link_native_executable(
        &llvm_ir,
        Some(&bridge_source),
        output,
        &NativeLinkOptions {
            math: true,
            pthread: true,
            libraries: vec![hip_library],
            rpaths: vec![hip_directory],
            ..NativeLinkOptions::default()
        },
    )
}

pub fn lower_to_rocdl(module: &Module, chip: &str) -> Result<Module, BackendError> {
    validate_chip(chip)?;

    if !module.as_str().contains("severian_parallel = \"gpu\"") {
        return Err(invalid_input(
            "ROCm lowering requires a `with gpu:` execution region",
        ));
    }

    let temporary = TemporaryFiles::new("severian-rocdl");
    let source = temporary.path("source.mlir");
    let lowered = temporary.path("rocdl.mlir");

    std::fs::write(&source, module.as_str())?;
    lower_device_to_rocdl(&source, &lowered, chip)?;

    let text = std::fs::read_to_string(&lowered)?;
    Ok(Module::new(text))
}

fn lower_device_to_rocdl(
    source: &Path,
    output: &Path,
    chip: &str,
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
            format!("--rocdl-attach-target=chip={chip}").into(),
            "--lower-affine".into(),
            "--convert-scf-to-cf".into(),
            "--convert-index-to-llvm".into(),
            "--convert-math-to-rocdl".into(),
            "--convert-arith-to-llvm".into(),
            "--convert-gpu-to-rocdl=runtime=HIP".into(),
            "--reconcile-unrealized-casts".into(),
            "-o".into(),
            output.as_os_str().to_owned(),
        ],
    )
}

pub fn detect_amd_gpu_chip() -> Option<String> {
    if let Ok(chip) = std::env::var("SEVERIAN_AMDGPU_CHIP") {
        let chip = chip.trim();
        if valid_chip(chip) {
            return Some(chip.to_owned());
        }
    }

    if let Some(tool) = crate::toolchain::find_tool(Tool::AmdGpuArch) {
        if let Ok(output) = Command::new(tool).output() {
            if output.status.success() {
                if let Some(chip) = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .map(str::trim)
                    .find(|chip| valid_chip(chip))
                {
                    return Some(chip.to_owned());
                }
            }
        }
    }

    if let Some(tool) = crate::toolchain::find_tool(Tool::RocmInfo) {
        if let Ok(output) = Command::new(tool).output() {
            if output.status.success() {
                if let Some(chip) = String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .map(|word| {
                        word.trim_matches(|character: char| !character.is_ascii_alphanumeric())
                    })
                    .find(|word| valid_chip(word))
                {
                    return Some(chip.to_owned());
                }
            }
        }
    }

    None
}

fn prepare_rocm_toolkit(
    temporary: &TemporaryFiles,
) -> Result<PathBuf, BackendError> {
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

    let lld = find_required_tool(Tool::Lld)?;
    let toolkit = temporary.directory("rocm-toolkit");
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
    ]
    .iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}

fn validate_chip(chip: &str) -> Result<(), BackendError> {
    if valid_chip(chip) {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "invalid AMD GPU architecture `{chip}`; expected gfx*"
        )))
    }
}

fn valid_chip(chip: &str) -> bool {
    chip.strip_prefix("gfx").is_some_and(|suffix| {
        suffix.len() >= 3 && suffix.chars().all(|character| character.is_ascii_alphanumeric())
    })
}

fn invalid_input(message: impl Into<String>) -> BackendError {
    BackendError(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}
