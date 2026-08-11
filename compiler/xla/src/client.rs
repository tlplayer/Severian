use crate::{
    pipeline::{CompileOptions, XlaPipeline},
    pjrt::{
        Buffer, Device, HostBuffer, LoadedExecutable, PjrtClient,
    },
    stablehlo::StableHloModule,
    Result,
};

/// High-level XLA client used by Severian.
///
/// This type intentionally hides whether execution is backed by the XLA CPU
/// plugin, XLA GPU plugin, TPU, or another PJRT implementation.
pub struct XlaClient {
    pjrt: PjrtClient,
    pipeline: XlaPipeline,
}

impl XlaClient {
    pub fn new(pjrt: PjrtClient) -> Self {
        Self {
            pjrt,
            pipeline: XlaPipeline::default(),
        }
    }

    pub fn with_pipeline(pjrt: PjrtClient, pipeline: XlaPipeline) -> Self {
        Self { pjrt, pipeline }
    }

    pub fn pjrt(&self) -> &PjrtClient {
        &self.pjrt
    }

    pub fn pipeline(&self) -> &XlaPipeline {
        &self.pipeline
    }

    pub fn devices(&self) -> Result<Vec<Device>> {
        self.pjrt.devices()
    }

    pub fn amd_gpu_device(&self) -> Result<Device> {
        self.pjrt.amd_gpu_device()
    }

    pub fn default_device(&self) -> Result<Device> {
        self.pjrt.default_device()
    }

    pub fn compile(
        &self,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<LoadedExecutable> {
        self.pipeline.compile(&self.pjrt, module, options)
    }

    pub fn upload(&self, host: HostBuffer) -> Result<Buffer> {
        self.pjrt.buffer_from_host(host, None)
    }

    pub fn upload_to(&self, host: HostBuffer, device: &Device) -> Result<Buffer> {
        self.pjrt.buffer_from_host(host, Some(device))
    }

    pub fn execute(
        &self,
        executable: &LoadedExecutable,
        arguments: &[&Buffer],
        device: &Device,
    ) -> Result<Vec<Buffer>> {
        executable.execute(arguments, device)
    }

    /// Compiles one StableHLO module, uploads all arguments to the selected
    /// AMD GPU, executes it there, and returns owned GPU-resident outputs.
    /// This is the single production path shared by direct backend tests and
    /// StableHLO emitted from Severian lowering.
    pub fn compile_and_execute_amd(
        &self,
        module: &StableHloModule,
        arguments: impl IntoIterator<Item = HostBuffer>,
        options: &CompileOptions,
    ) -> Result<Vec<Buffer>> {
        let device = self.amd_gpu_device()?;
        let executable = self.compile(module, options)?;
        let buffers = arguments
            .into_iter()
            .map(|argument| self.upload_to(argument, &device))
            .collect::<Result<Vec<_>>>()?;
        let borrowed = buffers.iter().collect::<Vec<_>>();
        executable.execute(&borrowed, &device)
    }
}
