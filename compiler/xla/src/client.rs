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
}
