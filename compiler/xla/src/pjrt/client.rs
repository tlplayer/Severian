use super::{
    buffer::{Buffer, HostBuffer},
    compile::RawClient,
    device::Device,
    executable::LoadedExecutable,
    plugin::RawPjrtPlugin,
};
use crate::{pipeline::CompileOptions, stablehlo::StableHloModule, Result};
use std::{ffi::c_void, path::Path, sync::Arc};

/// Loaded and initialized PJRT plugin.
pub struct PjrtPlugin {
    raw: RawPjrtPlugin,
}

impl PjrtPlugin {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { raw: RawPjrtPlugin::load(path)? })
    }

    pub fn path(&self) -> &Path { self.raw.path() }

    pub fn raw_api(&self) -> *const c_void {
        (self.raw.api() as *const super::api::PJRT_Api).cast()
    }
}

/// Owned PJRT client. Devices are borrowed from this client's PJRT lifetime;
/// buffers and loaded executables own their raw handles and destroy them on
/// drop through the plugin function table.
#[derive(Clone)]
pub struct PjrtClient {
    raw: Arc<RawClient>,
}

impl PjrtClient {
    pub fn new(plugin: PjrtPlugin) -> Result<Self> {
        Ok(Self { raw: Arc::new(RawClient::create(plugin.raw)?) })
    }

    pub fn platform_name(&self) -> Result<String> { self.raw.platform_name() }

    pub fn devices(&self) -> Result<Vec<Device>> {
        self.raw.devices()?
            .into_iter()
            .map(|device| Device::from_raw(device, Arc::clone(&self.raw)))
            .collect()
    }

    pub fn default_device(&self) -> Result<Device> {
        Device::from_raw(self.raw.default_device()?, Arc::clone(&self.raw))
    }

    pub fn compile(
        &self,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<LoadedExecutable> {
        self.raw.compile(module, options).map(LoadedExecutable::from_raw)
    }

    pub fn buffer_from_host(
        &self,
        host: HostBuffer,
        device: Option<&Device>,
    ) -> Result<Buffer> {
        let selected = match device {
            Some(device) => device.raw(),
            None => self.raw.default_device()?,
        };
        self.raw
            .upload_host_buffer(&host, selected.raw())
            .map(Buffer::from_raw)
    }
}
