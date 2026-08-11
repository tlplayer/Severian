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

    /// Loads the ROCm PJRT plugin selected for Severian.
    ///
    /// Packaged XLA distributions do not use one universal install prefix, so
    /// `SEVERIAN_ROCM_PJRT_PLUGIN` is the authoritative override. The two
    /// conventional ROCm system locations are tried when it is not set.
    pub fn load_rocm() -> Result<Self> {
        if let Some(path) = std::env::var_os("SEVERIAN_ROCM_PJRT_PLUGIN") {
            return Self::load(path);
        }

        for path in [
            "/opt/rocm/lib/libpjrt_rocm.so",
            "/opt/rocm/lib/libpjrt_plugin_rocm.so",
        ] {
            if Path::new(path).is_file() {
                return Self::load(path);
            }
        }

        Err(crate::XlaError::PluginLoad(
            "ROCm PJRT plugin not found; set SEVERIAN_ROCM_PJRT_PLUGIN to the library exporting GetPjrtApi"
                .into(),
        ))
    }

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

    pub fn addressable_devices(&self) -> Result<Vec<Device>> {
        self.raw.addressable_devices()?
            .into_iter()
            .map(|device| Device::from_raw(device, Arc::clone(&self.raw)))
            .collect()
    }

    pub fn amd_gpu_device(&self) -> Result<Device> {
        self.addressable_devices()?
            .into_iter()
            .find(|device| device.kind == super::device::DeviceKind::AmdGpu)
            .ok_or_else(|| crate::XlaError::Pjrt(
                "PJRT client has no addressable AMD GPU".into(),
            ))
    }

    pub fn default_device(&self) -> Result<Device> {
        Device::from_raw(self.raw.default_device()?, Arc::clone(&self.raw))
    }

    pub fn compile(
        &self,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<LoadedExecutable> {
        self.raw
            .compile(module, options)
            .map(|executable| LoadedExecutable::from_raw(executable, Arc::clone(&self.raw)))
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
            .map(|buffer| Buffer::from_raw(buffer, Arc::clone(&self.raw)))
    }
}
