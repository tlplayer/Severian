use super::{
    buffer::{Buffer, BufferId, HostBuffer},
    device::{Device, DeviceId},
    executable::{ExecuteOptions, ExecutionResult, LoadedExecutable},
};
use crate::{
    pipeline::CompileOptions,
    stablehlo::StableHloModule,
    Result, XlaError,
};
use std::{
    ffi::{c_char, c_void, CString},
    path::{Path, PathBuf},
    sync::Arc,
};

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
/// This type performs only plugin loading and symbol resolution. The exact
/// function-table layout belongs in a generated/raw ABI module once Severian
/// pins an XLA/PJRT API revision.
pub struct PjrtPlugin {
    path: PathBuf,
    library: NativeLibrary,
    api: *const c_void,
}

unsafe impl Send for PjrtPlugin {}
unsafe impl Sync for PjrtPlugin {}

impl PjrtPlugin {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let library = NativeLibrary::open(&path)?;

        type GetPjrtApi = unsafe extern "C" fn() -> *const c_void;
        let get_api: GetPjrtApi = unsafe { library.symbol(b"GetPjrtApi\0")? };
        let api = unsafe { get_api() };

        if api.is_null() {
            return Err(XlaError::PluginLoad(format!(
                "{} returned a null PJRT_Api pointer",
                path.display()
            )));
        }

        Ok(Self { path, library, api })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn raw_api(&self) -> *const c_void {
        self.api
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
        "PJRT plugin loaded, but raw C-API bridge for `{operation}` has not been pinned/generated yet"
    ))
}

#[cfg(unix)]
struct NativeLibrary {
    handle: *mut c_void,
}

#[cfg(unix)]
unsafe impl Send for NativeLibrary {}
#[cfg(unix)]
unsafe impl Sync for NativeLibrary {}

#[cfg(unix)]
impl NativeLibrary {
    fn open(path: &Path) -> Result<Self> {
        let path = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
            XlaError::PluginLoad(format!(
                "plugin path contains an interior NUL: {}",
                path.display()
            ))
        })?;

        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            return Err(XlaError::PluginLoad(dl_error()));
        }

        Ok(Self { handle })
    }

    unsafe fn symbol<T: Copy>(&self, name: &[u8]) -> Result<T> {
        let name = std::ffi::CStr::from_bytes_with_nul(name)
            .map_err(|error| XlaError::PluginLoad(error.to_string()))?;

        dlerror();
        let symbol = dlsym(self.handle, name.as_ptr());
        let error = dlerror();

        if !error.is_null() {
            return Err(XlaError::PluginLoad(
                std::ffi::CStr::from_ptr(error)
                    .to_string_lossy()
                    .into_owned(),
            ));
        }

        if symbol.is_null() {
            return Err(XlaError::PluginLoad(format!(
                "symbol {} resolved to null",
                name.to_string_lossy()
            )));
        }

        Ok(std::mem::transmute_copy(&symbol))
    }
}

#[cfg(unix)]
impl Drop for NativeLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

#[cfg(unix)]
const RTLD_LOCAL: i32 = 0;
#[cfg(unix)]
const RTLD_NOW: i32 = 2;

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const c_char;
}

#[cfg(unix)]
fn dl_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dynamic loader error".into()
    } else {
        unsafe { std::ffi::CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(not(unix))]
struct NativeLibrary;

#[cfg(not(unix))]
impl NativeLibrary {
    fn open(path: &Path) -> Result<Self> {
        Err(XlaError::PluginLoad(format!(
            "native PJRT plugin loading is not implemented on this platform: {}",
            path.display()
        )))
    }

    unsafe fn symbol<T: Copy>(&self, _name: &[u8]) -> Result<T> {
        Err(XlaError::PluginLoad(
            "native PJRT symbol loading is not implemented on this platform".into(),
        ))
    }
}
