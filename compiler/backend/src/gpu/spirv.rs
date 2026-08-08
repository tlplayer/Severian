use crate::{
    toolchain::{find_required_tool, run_tool, TemporaryFiles, Tool},
    BackendError,
};
use severian_mlir::Module;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpirvTarget {
    pub version: String,
    pub client_api: String,
    pub capabilities: Vec<String>,
}

impl Default for SpirvTarget {
    fn default() -> Self {
        Self {
            version: "v1.3".into(),
            client_api: "vulkan".into(),
            capabilities: vec!["Shader".into()],
        }
    }
}

/// Lowers GPU-dialect kernels to SPIR-V dialect MLIR.
///
/// Binary serialization is intentionally separate: Vulkan, OpenCL and custom
/// SPIR-V consumers have different packaging/linking requirements.
pub fn lower_to_spirv(
    module: &Module,
    target: &SpirvTarget,
) -> Result<Module, BackendError> {
    if !module.as_str().contains("severian_parallel = \"gpu\"") {
        return Err(BackendError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SPIR-V lowering requires a `with gpu:` execution region",
        )));
    }

    let temporary = TemporaryFiles::new("severian-spirv");
    let source = temporary.path("source.mlir");
    let lowered = temporary.path("spirv.mlir");

    std::fs::write(&source, module.as_str())?;

    let mlir_opt = find_required_tool(Tool::MlirOpt)?;

    let capabilities = if target.capabilities.is_empty() {
        String::new()
    } else {
        target.capabilities.join(",")
    };

    let target_environment = format!(
        "--spirv-update-vce=spirv-version={} client-api={} capabilities={}",
        target.version, target.client_api, capabilities
    );

    run_tool(
        &mlir_opt,
        &[
            source.as_os_str().to_owned(),
            "--convert-linalg-to-parallel-loops".into(),
            "--gpu-map-parallel-loops".into(),
            "--convert-parallel-loops-to-gpu".into(),
            "--gpu-kernel-outlining".into(),
            "--set-spirv-abi-attrs".into(),
            "--convert-gpu-to-spirv".into(),
            target_environment.into(),
            "--canonicalize".into(),
            "-o".into(),
            lowered.as_os_str().to_owned(),
        ],
    )?;

    Ok(Module::new(std::fs::read_to_string(lowered)?))
}

pub fn validate_spirv_binary(path: &Path) -> Result<(), BackendError> {
    let validator = find_required_tool(Tool::SpirvVal)?;
    run_tool(&validator, &[path.as_os_str().to_owned()])
}

pub fn optimize_spirv_binary(
    input: &Path,
    output: &Path,
) -> Result<(), BackendError> {
    let optimizer = find_required_tool(Tool::SpirvOpt)?;
    run_tool(
        &optimizer,
        &[
            "-O".into(),
            input.as_os_str().to_owned(),
            "-o".into(),
            output.as_os_str().to_owned(),
        ],
    )
}
