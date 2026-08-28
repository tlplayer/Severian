//! Runtime-loaded native donor bridge.
//!
//! Keeping the bridge dynamically loaded lets ordinary CPU-only Severian
//! builds remain usable. GPU compilation itself is entirely native: Rust ->
//! versioned C ABI -> Triton/MLIR/LLVM C++.

use crate::{
    with_abi_request, AbiCompileFn, AbiCompiledKernel, AbiDestroyKernelFn, AbiStatus, BridgeError,
    CompileOptions, CompiledKernel, FusionGraph, FusionRegion, KernelSpecialization,
    LaunchMetadata, TritonCompiler, ABI_VERSION,
};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem::MaybeUninit;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::sync::Arc;

const RTLD_NOW: c_int = 2;
const DEFAULT_LIBRARY_NAME: &str = "libseverian_triton_bridge.so";

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

struct DynamicLibrary(*mut c_void);

// `dlopen` handles may be used concurrently. The bridge has no per-handle
// mutable state and protects LLVM's process-global initialization internally.
unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { dlclose(self.0) };
        }
    }
}

#[derive(Clone)]
pub struct NativeTritonCompiler {
    _library: Arc<DynamicLibrary>,
    compile: AbiCompileFn,
    destroy: AbiDestroyKernelFn,
}

impl std::fmt::Debug for NativeTritonCompiler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeTritonCompiler")
            .finish_non_exhaustive()
    }
}

impl NativeTritonCompiler {
    /// Loads the bridge named by `SEVERIAN_TRITON_BRIDGE_LIBRARY`, falling
    /// back to the platform loader's search path.
    pub fn load() -> Result<Self, BridgeError> {
        let path = std::env::var_os("SEVERIAN_TRITON_BRIDGE_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_LIBRARY_NAME));
        Self::load_from(path)
    }

    pub fn load_from(path: impl AsRef<Path>) -> Result<Self, BridgeError> {
        let bytes = path.as_ref().as_os_str().as_bytes();
        let path = CString::new(bytes).map_err(|_| {
            BridgeError::NativeUnavailable("library path contains a NUL byte".into())
        })?;
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(BridgeError::NativeUnavailable(dl_error()));
        }
        let library = Arc::new(DynamicLibrary(handle));
        let compile = unsafe { load_symbol::<AbiCompileFn>(handle, c"sev_triton_compile")? };
        let destroy =
            unsafe { load_symbol::<AbiDestroyKernelFn>(handle, c"sev_triton_destroy_kernel")? };
        Ok(Self {
            _library: library,
            compile,
            destroy,
        })
    }
}

impl TritonCompiler for NativeTritonCompiler {
    fn compile(
        &self,
        graph: &FusionGraph,
        region: &FusionRegion,
        specialization: &KernelSpecialization,
        options: &CompileOptions,
    ) -> Result<CompiledKernel, BridgeError> {
        with_abi_request(graph, region, specialization, options, |request| {
            let mut raw = MaybeUninit::<AbiCompiledKernel>::uninit();
            let status = unsafe { (self.compile)(request, raw.as_mut_ptr()) };
            // ABI v5 requires the bridge to initialize output on every path.
            let mut raw = unsafe { raw.assume_init() };
            let result = decode_result(status, &raw);
            unsafe { (self.destroy)(&mut raw) };
            result
        })?
    }
}

fn decode_result(
    status: AbiStatus,
    raw: &AbiCompiledKernel,
) -> Result<CompiledKernel, BridgeError> {
    let diagnostics = copy_bytes(raw.diagnostics.data, raw.diagnostics.len);
    let diagnostic = String::from_utf8_lossy(&diagnostics).into_owned();
    if status != AbiStatus::Ok {
        return Err(match status {
            AbiStatus::UnsupportedTarget => BridgeError::UnsupportedTarget,
            AbiStatus::ParseFailure => BridgeError::InvalidTtir(diagnostic),
            AbiStatus::InvalidArgument
            | AbiStatus::PassFailure
            | AbiStatus::CodegenFailure
            | AbiStatus::InternalFailure
            | AbiStatus::Ok => BridgeError::DonorCompiler(diagnostic),
        });
    }
    if raw.abi_version != ABI_VERSION {
        return Err(BridgeError::AbiMismatch {
            expected: ABI_VERSION,
            found: raw.abi_version,
        });
    }
    let entry = copy_bytes(raw.entry_point.data, raw.entry_point.len);
    let entry_point = String::from_utf8(entry).map_err(|_| {
        BridgeError::DonorCompiler("bridge returned a non-UTF-8 entry point".into())
    })?;
    Ok(CompiledKernel {
        format: raw.format,
        entry_point,
        code: copy_bytes(raw.code.data, raw.code.len),
        launch: LaunchMetadata {
            grid: [raw.launch.grid_x, raw.launch.grid_y, raw.launch.grid_z],
            num_warps: raw.launch.num_warps,
            warp_size: raw.launch.warp_size,
            num_ctas: raw.launch.num_ctas,
            shared_memory_bytes: raw.launch.shared_memory_bytes,
        },
    })
}

fn copy_bytes(data: *const u8, len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    if data.is_null() {
        return Vec::new();
    }
    unsafe { slice::from_raw_parts(data, len) }.to_vec()
}

unsafe fn load_symbol<T: Copy>(handle: *mut c_void, name: &CStr) -> Result<T, BridgeError> {
    // Clear a previous loader error before calling `dlsym`.
    unsafe { dlerror() };
    let symbol = unsafe { dlsym(handle, name.as_ptr()) };
    if symbol.is_null() {
        return Err(BridgeError::NativeUnavailable(dl_error()));
    }
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    Ok(unsafe { ptr::read((&symbol as *const *mut c_void).cast::<T>()) })
}

fn dl_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "dynamic loader returned no diagnostic".into()
    } else {
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_bridge_is_a_structured_error() {
        let error =
            NativeTritonCompiler::load_from("/definitely/not/a/triton/bridge.so").unwrap_err();
        assert!(matches!(error, BridgeError::NativeUnavailable(_)));
    }
}
