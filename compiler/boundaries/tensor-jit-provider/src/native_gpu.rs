use severian_fusion::GpuTarget;
use severian_runtime::gpu::{
    BufferId, DeviceId, DeviceInfo, EventId, GpuDriver, KernelArtifact, KernelBinaryFormat,
    KernelId, LaunchCommand,
};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::mem;
use std::ptr;

const RTLD_NOW: c_int = 2;
const HIP_SUCCESS: i32 = 0;
const CUDA_SUCCESS: i32 = 0;

type HipInit = unsafe extern "C" fn(u32) -> i32;
type HipGetDeviceCount = unsafe extern "C" fn(*mut i32) -> i32;
type HipSetDevice = unsafe extern "C" fn(i32) -> i32;
type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type HipFree = unsafe extern "C" fn(*mut c_void) -> i32;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
type HipModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
type HipModuleGetFunction =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32;
type HipModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
type HipModuleLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type HipDeviceSynchronize = unsafe extern "C" fn() -> i32;

type CuInit = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxCreate = unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32;
type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> i32;
type CuCtxDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> i32;
type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> i32;
type CuLaunchKernel = unsafe extern "C" fn(
    *mut c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type CuCtxSynchronize = unsafe extern "C" fn() -> i32;

struct Library(*mut c_void);

unsafe impl Send for Library {}

impl Library {
    fn open(names: &[&str]) -> Result<Self, String> {
        for name in names {
            let name = CString::new(*name).expect("driver library names contain no NUL");
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if !handle.is_null() {
                return Ok(Self(handle));
            }
        }
        Err(loader_error())
    }

    unsafe fn symbol(&self, name: &'static CStr) -> Result<*mut c_void, String> {
        let value = unsafe { dlsym(self.0, name.as_ptr()) };
        if value.is_null() {
            Err(format!(
                "GPU driver symbol {} is unavailable: {}",
                name.to_string_lossy(),
                loader_error()
            ))
        } else {
            Ok(value)
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { dlclose(self.0) };
        }
    }
}

struct Hip {
    _library: Library,
    set_device: HipSetDevice,
    malloc: HipMalloc,
    free: HipFree,
    memcpy: HipMemcpy,
    module_load_data: HipModuleLoadData,
    module_get_function: HipModuleGetFunction,
    module_unload: HipModuleUnload,
    module_launch_kernel: HipModuleLaunchKernel,
    synchronize: HipDeviceSynchronize,
}

struct Cuda {
    _library: Library,
    context: *mut c_void,
    set_current: CuCtxSetCurrent,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    module_load_data: CuModuleLoadData,
    module_get_function: CuModuleGetFunction,
    module_unload: CuModuleUnload,
    launch_kernel: CuLaunchKernel,
    synchronize: CuCtxSynchronize,
    destroy_context: CuCtxDestroy,
}

unsafe impl Send for Cuda {}

enum Backend {
    Hip(Hip),
    Cuda(Cuda),
}

struct LoadedKernel {
    module: *mut c_void,
    function: *mut c_void,
}

pub struct NativeGpuDriver {
    backend: Backend,
    target: GpuTarget,
    architecture: String,
    next_buffer: u64,
    next_kernel: u64,
    next_event: u64,
    buffers: BTreeMap<BufferId, u64>,
    kernels: BTreeMap<KernelId, LoadedKernel>,
}

unsafe impl Send for NativeGpuDriver {}

impl NativeGpuDriver {
    pub fn load(target: GpuTarget, architecture: String) -> Result<Self, String> {
        let backend = match target {
            GpuTarget::Amd => Backend::Hip(load_hip()?),
            GpuTarget::Nvidia => Backend::Cuda(load_cuda()?),
        };
        Ok(Self {
            backend,
            target,
            architecture,
            next_buffer: 0,
            next_kernel: 0,
            next_event: 0,
            buffers: BTreeMap::new(),
            kernels: BTreeMap::new(),
        })
    }

    fn activate(&self) -> Result<(), String> {
        match &self.backend {
            Backend::Hip(hip) => {
                status("hipSetDevice", unsafe { (hip.set_device)(0) }, HIP_SUCCESS)
            }
            Backend::Cuda(cuda) => status(
                "cuCtxSetCurrent",
                unsafe { (cuda.set_current)(cuda.context) },
                CUDA_SUCCESS,
            ),
        }
    }

    fn raw_buffer(&self, buffer: BufferId) -> Result<u64, String> {
        self.buffers
            .get(&buffer)
            .copied()
            .ok_or_else(|| format!("unknown GPU buffer {}", buffer.0))
    }
}

impl GpuDriver for NativeGpuDriver {
    fn discover_devices(&self) -> Result<Vec<DeviceInfo>, String> {
        self.activate()?;
        Ok(vec![DeviceInfo {
            id: DeviceId(0),
            target: self.target,
            name: match self.target {
                GpuTarget::Amd => "HIP device 0".into(),
                GpuTarget::Nvidia => "CUDA device 0".into(),
            },
            architecture: self.architecture.clone(),
            total_memory_bytes: 0,
            max_shared_memory_bytes: 0,
            warp_size: match self.target {
                GpuTarget::Amd => {
                    severian_triton::AmdTargetFeatures::new(&self.architecture).warp_size()
                }
                GpuTarget::Nvidia => 32,
            },
        }])
    }

    fn allocate(
        &mut self,
        _device: DeviceId,
        bytes: u64,
        _alignment: u64,
    ) -> Result<BufferId, String> {
        self.activate()?;
        let bytes =
            usize::try_from(bytes.max(1)).map_err(|_| "GPU allocation exceeds usize".to_owned())?;
        let raw = match &self.backend {
            Backend::Hip(hip) => {
                let mut raw = ptr::null_mut();
                status(
                    "hipMalloc",
                    unsafe { (hip.malloc)(&mut raw, bytes) },
                    HIP_SUCCESS,
                )?;
                raw as usize as u64
            }
            Backend::Cuda(cuda) => {
                let mut raw = 0;
                status(
                    "cuMemAlloc",
                    unsafe { (cuda.mem_alloc)(&mut raw, bytes) },
                    CUDA_SUCCESS,
                )?;
                raw
            }
        };
        let id = BufferId(self.next_buffer);
        self.next_buffer += 1;
        self.buffers.insert(id, raw);
        Ok(id)
    }

    fn deallocate(&mut self, buffer: BufferId) -> Result<(), String> {
        self.activate()?;
        let raw = self
            .buffers
            .remove(&buffer)
            .ok_or_else(|| format!("unknown GPU buffer {}", buffer.0))?;
        match &self.backend {
            Backend::Hip(hip) => status(
                "hipFree",
                unsafe { (hip.free)(raw as usize as *mut c_void) },
                HIP_SUCCESS,
            ),
            Backend::Cuda(cuda) => {
                status("cuMemFree", unsafe { (cuda.mem_free)(raw) }, CUDA_SUCCESS)
            }
        }
    }

    fn upload(&mut self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), String> {
        self.activate()?;
        let raw = self
            .raw_buffer(buffer)?
            .checked_add(offset)
            .ok_or_else(|| "GPU upload address overflow".to_owned())?;
        match &self.backend {
            Backend::Hip(hip) => status(
                "hipMemcpyHtoD",
                unsafe {
                    (hip.memcpy)(
                        raw as usize as *mut c_void,
                        data.as_ptr().cast(),
                        data.len(),
                        1,
                    )
                },
                HIP_SUCCESS,
            ),
            Backend::Cuda(cuda) => status(
                "cuMemcpyHtoD",
                unsafe { (cuda.memcpy_htod)(raw, data.as_ptr().cast(), data.len()) },
                CUDA_SUCCESS,
            ),
        }
    }

    fn download(&mut self, buffer: BufferId, offset: u64, data: &mut [u8]) -> Result<(), String> {
        self.activate()?;
        let raw = self
            .raw_buffer(buffer)?
            .checked_add(offset)
            .ok_or_else(|| "GPU download address overflow".to_owned())?;
        match &self.backend {
            Backend::Hip(hip) => status(
                "hipMemcpyDtoH",
                unsafe {
                    (hip.memcpy)(
                        data.as_mut_ptr().cast(),
                        raw as usize as *const c_void,
                        data.len(),
                        2,
                    )
                },
                HIP_SUCCESS,
            ),
            Backend::Cuda(cuda) => status(
                "cuMemcpyDtoH",
                unsafe { (cuda.memcpy_dtoh)(data.as_mut_ptr().cast(), raw, data.len()) },
                CUDA_SUCCESS,
            ),
        }
    }

    fn device_address(&self, buffer: BufferId) -> Result<u64, String> {
        self.raw_buffer(buffer)
    }

    fn load_kernel(
        &mut self,
        _device: DeviceId,
        artifact: &KernelArtifact,
    ) -> Result<KernelId, String> {
        self.activate()?;
        let entry = CString::new(artifact.entry_point.as_bytes())
            .map_err(|_| "kernel entry point contains NUL".to_owned())?;
        let (module, function) = match &self.backend {
            Backend::Hip(hip) => {
                if artifact.format != KernelBinaryFormat::Hsaco {
                    return Err(format!(
                        "HIP requires HSACO, received {:?}",
                        artifact.format
                    ));
                }
                let mut module = ptr::null_mut();
                status(
                    "hipModuleLoadData",
                    unsafe { (hip.module_load_data)(&mut module, artifact.code.as_ptr().cast()) },
                    HIP_SUCCESS,
                )?;
                let mut function = ptr::null_mut();
                if let Err(error) = status(
                    "hipModuleGetFunction",
                    unsafe { (hip.module_get_function)(&mut function, module, entry.as_ptr()) },
                    HIP_SUCCESS,
                ) {
                    unsafe { (hip.module_unload)(module) };
                    return Err(error);
                }
                (module, function)
            }
            Backend::Cuda(cuda) => {
                if !matches!(
                    artifact.format,
                    KernelBinaryFormat::Ptx | KernelBinaryFormat::Cubin
                ) {
                    return Err(format!(
                        "CUDA requires PTX or CUBIN, received {:?}",
                        artifact.format
                    ));
                }
                let mut image = artifact.code.clone();
                if artifact.format == KernelBinaryFormat::Ptx && !image.ends_with(&[0]) {
                    image.push(0);
                }
                let mut module = ptr::null_mut();
                status(
                    "cuModuleLoadData",
                    unsafe { (cuda.module_load_data)(&mut module, image.as_ptr().cast()) },
                    CUDA_SUCCESS,
                )?;
                let mut function = ptr::null_mut();
                if let Err(error) = status(
                    "cuModuleGetFunction",
                    unsafe { (cuda.module_get_function)(&mut function, module, entry.as_ptr()) },
                    CUDA_SUCCESS,
                ) {
                    unsafe { (cuda.module_unload)(module) };
                    return Err(error);
                }
                (module, function)
            }
        };
        let id = KernelId(self.next_kernel);
        self.next_kernel += 1;
        self.kernels.insert(id, LoadedKernel { module, function });
        Ok(id)
    }

    fn unload_kernel(&mut self, kernel: KernelId) -> Result<(), String> {
        self.activate()?;
        let kernel = self
            .kernels
            .remove(&kernel)
            .ok_or_else(|| format!("unknown GPU kernel {}", kernel.0))?;
        match &self.backend {
            Backend::Hip(hip) => status(
                "hipModuleUnload",
                unsafe { (hip.module_unload)(kernel.module) },
                HIP_SUCCESS,
            ),
            Backend::Cuda(cuda) => status(
                "cuModuleUnload",
                unsafe { (cuda.module_unload)(kernel.module) },
                CUDA_SUCCESS,
            ),
        }
    }

    fn launch(&mut self, command: &LaunchCommand) -> Result<EventId, String> {
        self.activate()?;
        let function = self
            .kernels
            .get(&command.kernel)
            .ok_or_else(|| format!("unknown GPU kernel {}", command.kernel.0))?
            .function;
        let grid = command
            .grid
            .map(|value| u32::try_from(value).map_err(|_| "GPU grid exceeds driver ABI".to_owned()))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let shared = u32::try_from(command.shared_memory_bytes)
            .map_err(|_| "GPU shared-memory request exceeds driver ABI".to_owned())?;
        let mut parameters = command
            .arguments
            .offsets
            .iter()
            .map(|offset| unsafe { command.arguments.storage.as_ptr().add(*offset) as *mut c_void })
            .collect::<Vec<_>>();
        let result = match &self.backend {
            Backend::Hip(hip) => unsafe {
                (hip.module_launch_kernel)(
                    function,
                    grid[0],
                    grid[1],
                    grid[2],
                    command.block[0],
                    command.block[1],
                    command.block[2],
                    shared,
                    ptr::null_mut(),
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            Backend::Cuda(cuda) => unsafe {
                (cuda.launch_kernel)(
                    function,
                    grid[0],
                    grid[1],
                    grid[2],
                    command.block[0],
                    command.block[1],
                    command.block[2],
                    shared,
                    ptr::null_mut(),
                    parameters.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
        };
        status("GPU kernel launch", result, 0)?;
        let event = EventId(self.next_event);
        self.next_event += 1;
        Ok(event)
    }

    fn wait(&mut self, _events: &[EventId]) -> Result<(), String> {
        self.activate()?;
        match &self.backend {
            Backend::Hip(hip) => status(
                "hipDeviceSynchronize",
                unsafe { (hip.synchronize)() },
                HIP_SUCCESS,
            ),
            Backend::Cuda(cuda) => status(
                "cuCtxSynchronize",
                unsafe { (cuda.synchronize)() },
                CUDA_SUCCESS,
            ),
        }
    }
}

impl Drop for NativeGpuDriver {
    fn drop(&mut self) {
        let _ = self.activate();
        for (_, kernel) in mem::take(&mut self.kernels) {
            match &self.backend {
                Backend::Hip(hip) => unsafe {
                    (hip.module_unload)(kernel.module);
                },
                Backend::Cuda(cuda) => unsafe {
                    (cuda.module_unload)(kernel.module);
                },
            }
        }
        for (_, buffer) in mem::take(&mut self.buffers) {
            match &self.backend {
                Backend::Hip(hip) => unsafe {
                    (hip.free)(buffer as usize as *mut c_void);
                },
                Backend::Cuda(cuda) => unsafe {
                    (cuda.mem_free)(buffer);
                },
            }
        }
        if let Backend::Cuda(cuda) = &self.backend {
            unsafe {
                (cuda.destroy_context)(cuda.context);
            }
        }
    }
}

fn load_hip() -> Result<Hip, String> {
    let library = Library::open(&["libamdhip64.so", "libamdhip64.so.7"])?;
    unsafe {
        let init: HipInit = mem::transmute(library.symbol(c"hipInit")?);
        status("hipInit", init(0), HIP_SUCCESS)?;
        let get_count: HipGetDeviceCount = mem::transmute(library.symbol(c"hipGetDeviceCount")?);
        let mut count = 0;
        status("hipGetDeviceCount", get_count(&mut count), HIP_SUCCESS)?;
        if count == 0 {
            return Err("HIP reported no devices".into());
        }
        Ok(Hip {
            set_device: mem::transmute(library.symbol(c"hipSetDevice")?),
            malloc: mem::transmute(library.symbol(c"hipMalloc")?),
            free: mem::transmute(library.symbol(c"hipFree")?),
            memcpy: mem::transmute(library.symbol(c"hipMemcpy")?),
            module_load_data: mem::transmute(library.symbol(c"hipModuleLoadData")?),
            module_get_function: mem::transmute(library.symbol(c"hipModuleGetFunction")?),
            module_unload: mem::transmute(library.symbol(c"hipModuleUnload")?),
            module_launch_kernel: mem::transmute(library.symbol(c"hipModuleLaunchKernel")?),
            synchronize: mem::transmute(library.symbol(c"hipDeviceSynchronize")?),
            _library: library,
        })
    }
}

fn load_cuda() -> Result<Cuda, String> {
    let library = Library::open(&["libcuda.so.1", "libcuda.so"])?;
    unsafe {
        let init: CuInit = mem::transmute(library.symbol(c"cuInit")?);
        status("cuInit", init(0), CUDA_SUCCESS)?;
        let get_count: CuDeviceGetCount = mem::transmute(library.symbol(c"cuDeviceGetCount")?);
        let mut count = 0;
        status("cuDeviceGetCount", get_count(&mut count), CUDA_SUCCESS)?;
        if count == 0 {
            return Err("CUDA reported no devices".into());
        }
        let device_get: CuDeviceGet = mem::transmute(library.symbol(c"cuDeviceGet")?);
        let mut device = 0;
        status("cuDeviceGet", device_get(&mut device, 0), CUDA_SUCCESS)?;
        let create: CuCtxCreate = mem::transmute(library.symbol(c"cuCtxCreate_v2")?);
        let mut context = ptr::null_mut();
        status("cuCtxCreate", create(&mut context, 0, device), CUDA_SUCCESS)?;
        Ok(Cuda {
            context,
            set_current: mem::transmute(library.symbol(c"cuCtxSetCurrent")?),
            mem_alloc: mem::transmute(library.symbol(c"cuMemAlloc_v2")?),
            mem_free: mem::transmute(library.symbol(c"cuMemFree_v2")?),
            memcpy_htod: mem::transmute(library.symbol(c"cuMemcpyHtoD_v2")?),
            memcpy_dtoh: mem::transmute(library.symbol(c"cuMemcpyDtoH_v2")?),
            module_load_data: mem::transmute(library.symbol(c"cuModuleLoadData")?),
            module_get_function: mem::transmute(library.symbol(c"cuModuleGetFunction")?),
            module_unload: mem::transmute(library.symbol(c"cuModuleUnload")?),
            launch_kernel: mem::transmute(library.symbol(c"cuLaunchKernel")?),
            synchronize: mem::transmute(library.symbol(c"cuCtxSynchronize")?),
            destroy_context: mem::transmute(library.symbol(c"cuCtxDestroy_v2")?),
            _library: library,
        })
    }
}

fn status(operation: &str, found: i32, expected: i32) -> Result<(), String> {
    if found == expected {
        Ok(())
    } else {
        Err(format!("{operation} failed with driver status {found}"))
    }
}

fn loader_error() -> String {
    let value = unsafe { dlerror() };
    if value.is_null() {
        "dynamic loader returned no diagnostic".into()
    } else {
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned()
    }
}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}
