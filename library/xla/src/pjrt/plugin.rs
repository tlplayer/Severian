use super::{api, error};
use crate::{Result, XlaError};
use std::{
    ffi::{c_char, c_void, CString},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};

pub struct RawPjrtPlugin {
    inner: Arc<PluginInner>,
}

struct PluginInner {
    path: PathBuf,
    _library: NativeLibrary,
    api: NonNull<api::PJRT_Api>,
}

unsafe impl Send for PluginInner {}
unsafe impl Sync for PluginInner {}

impl Clone for RawPjrtPlugin {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl RawPjrtPlugin {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let library = NativeLibrary::open(&path)?;

        let get_api: api::GetPjrtApi = unsafe { library.symbol(b"GetPjrtApi\0")? };
        let raw_api = unsafe { get_api() };
        let api = NonNull::new(raw_api.cast_mut())
            .ok_or_else(|| error::invalid_raw_pointer("PJRT_Api"))?;

        let plugin = Self {
            inner: Arc::new(PluginInner {
                path,
                _library: library,
                api,
            }),
        };

        plugin.check_version()?;
        plugin.initialize()?;
        Ok(plugin)
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    pub fn api(&self) -> &api::PJRT_Api {
        unsafe { self.inner.api.as_ref() }
    }

    pub fn version(&self) -> api::PJRT_Api_Version {
        self.api().pjrt_api_version
    }

    fn check_version(&self) -> Result<()> {
        let version = self.version();

        if version.major_version != api::PJRT_API_MAJOR {
            return Err(XlaError::PluginLoad(format!(
                "PJRT API major mismatch: Severian expects {}, plugin reports {}",
                api::PJRT_API_MAJOR,
                version.major_version
            )));
        }

        // Severian only reads a prefix of the function table. A newer minor is
        // acceptable as long as the major stays compatible.
        if version.minor_version < api::PJRT_API_MINOR {
            return Err(XlaError::PluginLoad(format!(
                "PJRT plugin is too old: Severian bridge is pinned to 0.{}, plugin reports 0.{}",
                api::PJRT_API_MINOR,
                version.minor_version
            )));
        }

        Ok(())
    }

    fn initialize(&self) -> Result<()> {
        let mut args = api::PJRT_Plugin_Initialize_Args {
            struct_size: api::struct_size::<api::PJRT_Plugin_Initialize_Args>(),
            extension_start: api::null_extension(),
        };

        let error = unsafe { (self.api().PJRT_Plugin_Initialize)(&mut args) };
        unsafe { error::check(self.api(), error) }
    }
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
        use std::os::unix::ffi::OsStrExt;

        let path_c = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            XlaError::PluginLoad(format!(
                "plugin path contains an interior NUL: {}",
                path.display()
            ))
        })?;

        let handle = unsafe { dlopen(path_c.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
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
        let loader_error = dlerror();

        if !loader_error.is_null() {
            return Err(XlaError::PluginLoad(
                std::ffi::CStr::from_ptr(loader_error)
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
            "raw PJRT dynamic loading is not implemented on this platform: {}",
            path.display()
        )))
    }

    unsafe fn symbol<T: Copy>(&self, _name: &[u8]) -> Result<T> {
        Err(XlaError::PluginLoad(
            "raw PJRT symbol loading is not implemented on this platform".into(),
        ))
    }
}
