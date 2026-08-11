use crate::CompileError;
use severian_hir::{FunctionId, Program};
use severian_xla::{
    CompileOptions, Device, LoadedExecutable, PjrtClient, PjrtPlugin, StableHloModule, XlaClient,
};

#[derive(Debug, Clone)]
pub struct XlaKernelArtifact {
    pub function: FunctionId,
    pub stablehlo: StableHloModule,
}

pub fn collect_xla_kernels(program: &Program) -> Result<Vec<XlaKernelArtifact>, CompileError> {
    program
        .functions
        .iter()
        .filter(|function| {
            function
                .decorators
                .iter()
                .any(|decorator| decorator.package == "tensor")
        })
        .map(|function| {
            let module = severian_lowering::stablehlo::lower_entry(program, function.id)
                .map_err(|error| CompileError::Execution(error.to_string()))?;
            Ok(XlaKernelArtifact {
                function: function.id,
                stablehlo: StableHloModule::from_text(module.as_str()),
            })
        })
        .collect()
}

pub struct XlaExecutionContext {
    pub client: XlaClient,
    pub device: Device,
}

impl XlaExecutionContext {
    pub fn rocm() -> Result<Self, CompileError> {
        let plugin =
            PjrtPlugin::load_rocm().map_err(|error| CompileError::Execution(error.to_string()))?;
        let pjrt =
            PjrtClient::new(plugin).map_err(|error| CompileError::Execution(error.to_string()))?;
        let device = pjrt
            .amd_gpu_device()
            .map_err(|error| CompileError::Execution(error.to_string()))?;
        Ok(Self {
            client: XlaClient::new(pjrt),
            device,
        })
    }

    pub fn compile_entry(
        &self,
        program: &Program,
        function: FunctionId,
    ) -> Result<LoadedExecutable, CompileError> {
        let module = severian_lowering::stablehlo::lower_entry(program, function)
            .map_err(|error| CompileError::Execution(error.to_string()))?;
        self.client
            .compile(
                &StableHloModule::from_text(module.as_str()),
                &CompileOptions::default(),
            )
            .map_err(|error| CompileError::Execution(error.to_string()))
    }
}
