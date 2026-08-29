//! Runtime-loaded native donor bridge.
//!
//! Keeping the bridge dynamically loaded lets ordinary CPU-only Severian
//! builds remain usable. GPU compilation itself is entirely native: Rust ->
//! versioned C ABI -> Triton/MLIR/LLVM C++.

use crate::{
    with_abi_request, AbiCompileFn, AbiCompiledKernel, AbiDestroyKernelFn, AbiStatus, BridgeError,
    CompileOptions, CompileTarget, CompiledKernel, FusionGraph, FusionRegion, KernelFormat,
    KernelSpecialization, LaunchMetadata, TritonCompiler, ABI_VERSION, DONOR_REVISION,
};
use severian_runtime::gpu::{
    CompilerOptions as RuntimeCompilerOptions, GpuCompiler, GridPolicy, KernelArtifact,
    KernelBinaryFormat, KernelCompileRequest, LaunchRequirements,
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
    /// Loads the explicitly configured bridge, then a bridge staged beside
    /// the current executable, then the platform loader's search path.
    pub fn load() -> Result<Self, BridgeError> {
        if let Some(path) = std::env::var_os("SEVERIAN_TRITON_BRIDGE_LIBRARY") {
            return Self::load_from(PathBuf::from(path));
        }
        let mut candidates = Vec::new();
        if let Ok(executable) = std::env::current_exe() {
            if let Some(directory) = executable.parent() {
                candidates.push(directory.join(DEFAULT_LIBRARY_NAME));
            }
        }
        candidates.push(PathBuf::from(DEFAULT_LIBRARY_NAME));
        let mut diagnostics = Vec::new();
        for candidate in candidates {
            match Self::load_from(&candidate) {
                Ok(compiler) => return Ok(compiler),
                Err(error) => diagnostics.push(format!("{}: {error}", candidate.display())),
            }
        }
        Err(BridgeError::NativeUnavailable(diagnostics.join("; ")))
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
        format: decode_format(raw.format)?,
        entry_point,
        code: copy_bytes(raw.code.data, raw.code.len),
        launch: LaunchMetadata {
            grid: [raw.launch.grid_x, raw.launch.grid_y, raw.launch.grid_z],
            num_warps: raw.launch.num_warps,
            warp_size: raw.launch.warp_size,
            num_ctas: raw.launch.num_ctas,
            shared_memory_bytes: raw.launch.shared_memory_bytes,
            global_scratch_bytes_per_program: raw.launch.global_scratch_bytes_per_program,
            global_scratch_alignment: raw.launch.global_scratch_alignment,
            profile_scratch_bytes_per_program: raw.launch.profile_scratch_bytes_per_program,
            profile_scratch_alignment: raw.launch.profile_scratch_alignment,
        },
    })
}

fn decode_format(format: u32) -> Result<KernelFormat, BridgeError> {
    match format {
        1 => Ok(KernelFormat::LlvmIr),
        2 => Ok(KernelFormat::AmdGcN),
        3 => Ok(KernelFormat::Hsaco),
        4 => Ok(KernelFormat::Ptx),
        5 => Ok(KernelFormat::Cubin),
        _ => Err(BridgeError::DonorCompiler(format!(
            "bridge returned unknown kernel format {format}"
        ))),
    }
}

impl GpuCompiler for NativeTritonCompiler {
    fn donor_revision(&self) -> &str {
        DONOR_REVISION
    }

    fn compile(&self, request: &KernelCompileRequest<'_>) -> Result<KernelArtifact, String> {
        let options = triton_options(request.options)?;
        let compiled = TritonCompiler::compile(
            self,
            request.graph,
            request.region,
            request.specialization,
            &options,
        )
        .map_err(|error| error.to_string())?;
        let output = request
            .region
            .outputs
            .first()
            .copied()
            .ok_or_else(|| "fusion region has no output for launch grid".to_owned())?;
        let threads = compiled
            .launch
            .num_warps
            .checked_mul(compiled.launch.warp_size)
            .ok_or_else(|| "GPU block size overflows u32".to_owned())?;
        Ok(KernelArtifact {
            format: runtime_format(compiled.format),
            entry_point: compiled.entry_point,
            code: compiled.code,
            launch: LaunchRequirements {
                grid: GridPolicy::Linear {
                    output,
                    elements_per_program: 256,
                },
                block: [threads, 1, 1],
                num_warps: compiled.launch.num_warps,
                warp_size: compiled.launch.warp_size,
                num_ctas: compiled.launch.num_ctas,
                shared_memory_bytes: compiled.launch.shared_memory_bytes,
                global_scratch_bytes_per_program: compiled.launch.global_scratch_bytes_per_program,
                global_scratch_alignment: compiled.launch.global_scratch_alignment,
                profile_scratch_bytes_per_program: compiled
                    .launch
                    .profile_scratch_bytes_per_program,
                profile_scratch_alignment: compiled.launch.profile_scratch_alignment,
            },
        })
    }
}

fn triton_options(options: &RuntimeCompilerOptions) -> Result<CompileOptions, String> {
    let target = match options.target {
        severian_fusion::GpuTarget::Amd => CompileTarget::AmdGpu,
        severian_fusion::GpuTarget::Nvidia => CompileTarget::NvidiaGpu,
    };
    Ok(CompileOptions {
        target,
        architecture: options.architecture.clone(),
        num_warps: options.num_warps,
        warp_size: options.warp_size,
        num_ctas: options.num_ctas,
        num_stages: options.num_stages,
        emit: triton_format(options.emit)?,
        debug: options.debug,
    })
}

fn triton_format(format: KernelBinaryFormat) -> Result<KernelFormat, String> {
    Ok(match format {
        KernelBinaryFormat::LlvmIr => KernelFormat::LlvmIr,
        KernelBinaryFormat::AmdGcN => KernelFormat::AmdGcN,
        KernelBinaryFormat::Hsaco => KernelFormat::Hsaco,
        KernelBinaryFormat::Ptx => KernelFormat::Ptx,
        KernelBinaryFormat::Cubin => KernelFormat::Cubin,
    })
}

fn runtime_format(format: KernelFormat) -> KernelBinaryFormat {
    match format {
        KernelFormat::LlvmIr => KernelBinaryFormat::LlvmIr,
        KernelFormat::AmdGcN => KernelBinaryFormat::AmdGcN,
        KernelFormat::Hsaco => KernelBinaryFormat::Hsaco,
        KernelFormat::Ptx => KernelBinaryFormat::Ptx,
        KernelFormat::Cubin => KernelBinaryFormat::Cubin,
    }
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
