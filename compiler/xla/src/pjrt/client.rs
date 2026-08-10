use super::{
    buffer::{Buffer, BufferId, HostBuffer},
    device::{Device, DeviceId},
    executable::{ExecuteOptions, ExecutionResult, LoadedExecutable},
    plugin::RawPjrtPlugin,
};
use crate::{
    pipeline::CompileOptions,
    stablehlo::StableHloModule,
    Result, XlaError,
};
use std::{ffi::c_void, path::Path, sync::Arc};

/// Backend interface implemented by the raw PJRT C API bridge.
///
/// The rest of Severian is intentionally independent of PJRT's C struct
/// versioning. Codex can replace/extend the raw bridge without touching the
/// compiler-facing APIs.
pub trait PjrtBackend: Send + Sync {
    fn platform_name(&self) -> Result<String>;
    fn devices(&self) -> Result<Vec<Device>>;
    fn default_device(&self) -> Result<Device>;

    fn compile(
        &self,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<LoadedExecutable>;

    fn buffer_from_host(
        &self,
        host: HostBuffer,
        device: Option<DeviceId>,
    ) -> Result<Buffer>;

    fn execute(
        &self,
        executable: &LoadedExecutable,
        inputs: &[Buffer],
        options: &ExecuteOptions,
    ) -> Result<ExecutionResult>;

    fn delete_buffer(&self, id: BufferId) -> Result<()>;
}

/// Stable Rust wrapper used by Severian.
#[derive(Clone)]
pub struct PjrtClient {
    backend: Arc<dyn PjrtBackend>,
}

impl PjrtClient {
    pub fn new(backend: impl PjrtBackend + 'static) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    pub fn platform_name(&self) -> Result<String> {
        self.backend.platform_name()
    }

    pub fn devices(&self) -> Result<Vec<Device>> {
        self.backend.devices()
    }

    pub fn default_device(&self) -> Result<Device> {
        self.backend.default_device()
    }

    pub fn compile(
        &self,
        module: &StableHloModule,
        options: &CompileOptions,
    ) -> Result<LoadedExecutable> {
        self.backend.compile(module, options)
    }

    pub fn buffer_from_host(
        &self,
        host: HostBuffer,
        device: Option<DeviceId>,
    ) -> Result<Buffer> {
        self.backend.buffer_from_host(host, device)
    }

    pub fn execute(
        &self,
        executable: &LoadedExecutable,
        inputs: &[Buffer],
        options: &ExecuteOptions,
    ) -> Result<ExecutionResult> {
        self.backend.execute(executable, inputs, options)
    }

    pub fn delete_buffer(&self, id: BufferId) -> Result<()> {
        self.backend.delete_buffer(id)
    }
}

/// Loaded PJRT plugin handle.
///
/// PJRT plugins expose `GetPjrtApi`, returning a versioned `PJRT_Api*`.
/// The raw ABI module owns loading, version checks, and initialization. This
/// wrapper is the single compiler-facing plugin/backend boundary.
pub struct PjrtPlugin {
    raw: RawPjrtPlugin,
}

impl PjrtPlugin {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { raw: RawPjrtPlugin::load(path)? })
    }

    pub fn path(&self) -> &Path {
        self.raw.path()
    }

    pub fn raw_api(&self) -> *const c_void {
        (self.raw.api() as *const super::api::PJRT_Api).cast()
    }
}

impl PjrtBackend for PjrtPlugin {
    fn platform_name(&self) -> Result<String> {
        Err(raw_bridge_required("platform_name"))
    }

    fn devices(&self) -> Result<Vec<Device>> {
        Err(raw_bridge_required("devices"))
    }

    fn default_device(&self) -> Result<Device> {
        Err(raw_bridge_required("default_device"))
    }

    fn compile(
        &self,
        _module: &StableHloModule,
        _options: &CompileOptions,
    ) -> Result<LoadedExecutable> {
        Err(raw_bridge_required("compile"))
    }

    fn buffer_from_host(
        &self,
        _host: HostBuffer,
        _device: Option<DeviceId>,
    ) -> Result<Buffer> {
        Err(raw_bridge_required("buffer_from_host"))
    }

    fn execute(
        &self,
        _executable: &LoadedExecutable,
        _inputs: &[Buffer],
        _options: &ExecuteOptions,
    ) -> Result<ExecutionResult> {
        Err(raw_bridge_required("execute"))
    }

    fn delete_buffer(&self, _id: BufferId) -> Result<()> {
        Err(raw_bridge_required("delete_buffer"))
    }
}

fn raw_bridge_required(operation: &str) -> XlaError {
    XlaError::Unsupported(format!(
        "PJRT raw C API supports `{operation}`, but its owned-handle adapter is not implemented"
    ))
}
