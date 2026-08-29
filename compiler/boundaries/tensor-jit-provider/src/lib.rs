mod native_gpu;

use severian_fusion::{
    ElementKind, FusionGraph, FusionPlan, GpuTarget, KernelSpecialization, NodeId, NodeKind,
    OperandRole, RuntimeOperand,
};
use severian_mlir::{LoweredFloatFormat, LoweredType, MlirArtifact};
use severian_runtime::gpu::{
    specialize_storage_views, CompilerOptions as GpuCompilerOptions, DeviceId, GpuRuntime,
    HostStorageInput, KernelCache, StorageSpecializationBinding,
};
use severian_runtime::tensor_jit::{TensorJitProgram, TensorJitTarget};
use severian_runtime::{
    StorageElementKind, StorageElementRepresentationAbi, StorageFloatFormat, StorageOwnership,
    StorageView, StorageViewAbi, STORAGE_VIEW_ABI_MAGIC, STORAGE_VIEW_ABI_VERSION,
    STORAGE_VIEW_CONTIGUOUS,
};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt::Write as _;
use std::fs;
use std::mem;
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub const PROVIDER_ABI_VERSION: u32 = 1;
const ABI_VERSION: u32 = PROVIDER_ABI_VERSION;
const REGION_MAGIC: u64 = 0x5356_544a_4954_4142;
const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 1;
const COMPILE_FAILED: i32 = 3;

const VALUE_STORAGE: u32 = 1;
const VALUE_POINTER: u32 = 2;
const VALUE_LIST_I64: u32 = 6;

const RTLD_NOW: c_int = 2;
static NEXT_LIBRARY: AtomicU64 = AtomicU64::new(0);

#[repr(C)]
#[derive(Clone, Copy)]
union ValuePayloadAbi {
    storage: *mut StorageViewAbi,
    pointer: *mut c_void,
    signed_integer: i64,
    unsigned_integer: u64,
    floating: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ValueAbi {
    abi_version: u32,
    byte_size: u32,
    kind: u32,
    bits: u32,
    value: ValuePayloadAbi,
}

#[repr(C)]
struct RegionAbi {
    magic: u64,
    abi_version: u32,
    byte_size: u32,
    graph_hash: [u64; 4],
    compiler_hash: [u64; 4],
    program: *const u8,
    program_size: u64,
    target: u32,
    input_count: u32,
    output_count: u32,
    reserved: u32,
}

type LaunchFn = unsafe extern "C" fn(
    *mut c_void,
    *const ValueAbi,
    u32,
    *mut ValueAbi,
    u32,
) -> i32;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type CompileFn = unsafe extern "C" fn(
    *mut c_void,
    *const RegionAbi,
    *const ValueAbi,
    u32,
    *mut CompiledAbi,
) -> i32;

#[repr(C)]
struct CompiledAbi {
    abi_version: u32,
    byte_size: u32,
    instance: *mut c_void,
    launch: Option<LaunchFn>,
    destroy: Option<DestroyFn>,
}

#[repr(C)]
pub struct ProviderAbi {
    abi_version: u32,
    byte_size: u32,
    compile: Option<CompileFn>,
    context: *mut c_void,
}

struct ProviderStatic(ProviderAbi);
unsafe impl Sync for ProviderStatic {}

static PROVIDER: ProviderStatic = ProviderStatic(ProviderAbi {
    abi_version: ABI_VERSION,
    byte_size: mem::size_of::<ProviderAbi>() as u32,
    compile: Some(compile_region),
    context: ptr::null_mut(),
});

enum CompiledInstance {
    Cpu {
        library: *mut c_void,
        launch: NativeLaunch,
    },
    Gpu(Mutex<GpuInstance>),
}

struct GpuInstance {
    runtime: GpuRuntime<native_gpu::NativeGpuDriver, severian_triton::NativeTritonCompiler>,
    program: TensorJitProgram,
    graph: FusionGraph,
    plan: FusionPlan,
    options: GpuCompilerOptions,
}

type NativeLaunch = unsafe extern "C" fn(*const ValueAbi, u32, *mut ValueAbi, u32) -> i32;

#[no_mangle]
pub extern "C" fn sev_tensor_jit_provider_v1() -> *const ProviderAbi {
    &PROVIDER.0
}

unsafe extern "C" fn compile_region(
    _context: *mut c_void,
    region: *const RegionAbi,
    inputs: *const ValueAbi,
    input_count: u32,
    compiled: *mut CompiledAbi,
) -> i32 {
    if region.is_null() || compiled.is_null() || (input_count != 0 && inputs.is_null()) {
        return INVALID_ARGUMENT;
    }
    unsafe {
        ptr::write_bytes(compiled, 0, 1);
    }
    let result = unsafe { compile_region_inner(&*region, inputs, input_count) };
    match result {
        Ok(instance) => {
            let instance = Box::into_raw(Box::new(instance)).cast::<c_void>();
            unsafe {
                *compiled = CompiledAbi {
                    abi_version: ABI_VERSION,
                    byte_size: mem::size_of::<CompiledAbi>() as u32,
                    instance,
                    launch: Some(launch_region),
                    destroy: Some(destroy_region),
                };
            }
            OK
        }
        Err(error) => {
            eprintln!("Severian Tensor-JIT compilation failed: {error}");
            COMPILE_FAILED
        }
    }
}

unsafe fn compile_region_inner(
    region: &RegionAbi,
    inputs: *const ValueAbi,
    input_count: u32,
) -> Result<CompiledInstance, String> {
    if region.magic != REGION_MAGIC
        || region.abi_version != ABI_VERSION
        || region.byte_size as usize != mem::size_of::<RegionAbi>()
        || region.input_count != input_count
        || region.program.is_null()
    {
        return Err("invalid versioned region descriptor".into());
    }
    let program_size = usize::try_from(region.program_size)
        .map_err(|_| "Tensor-JIT program is too large".to_owned())?;
    let encoded = unsafe { slice::from_raw_parts(region.program, program_size) };
    let mut program = TensorJitProgram::decode(encoded).map_err(|error| error.to_string())?;
    if region.target != target_number(program.target) {
        return Err("launcher target does not match serialized program".into());
    }
    let input_values = unsafe { slice::from_raw_parts(inputs, input_count as usize) };
    unsafe { hydrate_runtime_operands(&mut program, input_values) }?;
    let graph = program.graph().map_err(|error| error.to_string())?;
    let specialization = specialize_program(&program, &graph, input_values)?;
    match program.target {
        TensorJitTarget::Cpu => compile_cpu(&program, &graph, &specialization),
        TensorJitTarget::Amd | TensorJitTarget::Nvidia => {
            compile_gpu(&program, &graph, &specialization)
        }
    }
}

#[repr(C)]
struct RuntimeList {
    length: usize,
    capacity: usize,
    values: *const usize,
}

unsafe fn hydrate_runtime_operands(
    program: &mut TensorJitProgram,
    inputs: &[ValueAbi],
) -> Result<(), String> {
    for node in &mut program.nodes {
        for (operand_index, (input_node, role)) in node
            .inputs
            .iter()
            .zip(&node.operand_roles)
            .enumerate()
        {
            if *role == OperandRole::Data
                || node
                    .runtime_operands
                    .iter()
                    .any(|operand| usize::from(operand.input_index) == operand_index)
            {
                continue;
            }
            let Some(external_index) = program.inputs.iter().position(|input| input == input_node)
            else {
                continue;
            };
            let input = &inputs[external_index];
            if input.kind != VALUE_LIST_I64 {
                continue;
            }
            let raw = unsafe { input.value.pointer.cast::<RuntimeList>() };
            if raw.is_null() {
                return Err(format!(
                    "runtime shape operand {operand_index} for node {} is null",
                    node.id.0
                ));
            }
            let list = unsafe { &*raw };
            if list.length > list.capacity || (list.length != 0 && list.values.is_null()) {
                return Err(format!(
                    "runtime shape operand {operand_index} for node {} is invalid",
                    node.id.0
                ));
            }
            let values = unsafe { slice::from_raw_parts(list.values, list.length) }
                .iter()
                .map(|value| *value as i64)
                .collect();
            node.runtime_operands.push(RuntimeOperand {
                input_index: u16::try_from(operand_index)
                    .map_err(|_| "runtime operand index exceeds the ABI".to_owned())?,
                values,
            });
        }
    }
    Ok(())
}

fn target_number(target: TensorJitTarget) -> u32 {
    match target {
        TensorJitTarget::Cpu => 0,
        TensorJitTarget::Amd => 1,
        TensorJitTarget::Nvidia => 2,
    }
}

fn specialize_program(
    program: &TensorJitProgram,
    graph: &FusionGraph,
    inputs: &[ValueAbi],
) -> Result<KernelSpecialization, String> {
    if inputs.len() != program.inputs.len() {
        return Err("Tensor-JIT input count does not match the serialized graph".into());
    }
    let mut bindings = Vec::new();
    for (index, node_id) in program.inputs.iter().copied().enumerate() {
        let node = graph.node(node_id);
        if !logical_tensor_input(node) {
            continue;
        }
        let input = &inputs[index];
        let raw = unsafe {
            match input.kind {
                VALUE_STORAGE => input.value.storage,
                VALUE_POINTER => input.value.pointer.cast::<StorageViewAbi>(),
                _ => ptr::null_mut(),
            }
        };
        let view = unsafe { copy_storage_view(raw) }
            .map_err(|error| format!("input {index}: {error}"))?;
        bindings.push(StorageSpecializationBinding {
            node: node_id,
            view,
        });
    }
    let target = match program.target {
        TensorJitTarget::Nvidia => GpuTarget::Nvidia,
        TensorJitTarget::Cpu | TensorJitTarget::Amd => GpuTarget::Amd,
    };
    specialize_storage_views(graph, target, &bindings).map_err(|error| {
        let nodes = graph
            .nodes()
            .iter()
            .map(|node| {
                format!(
                    "{}:{:?}.{} inputs={:?} rank={:?} runtime={:?}",
                    node.id.0,
                    node.kind,
                    node.operation,
                    node.inputs,
                    node.shape.rank,
                    node.runtime_operands
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        format!("{error}; graph [{nodes}]")
    })
}

fn logical_tensor_input(node: &severian_fusion::FusionNode) -> bool {
    node.shape.element_kind != ElementKind::Opaque
        && !matches!(node.shape.rank, severian_fusion::Rank::Ranked(ref axes) if axes.is_empty())
}

unsafe fn copy_storage_view(raw: *const StorageViewAbi) -> Result<StorageView, String> {
    if raw.is_null() {
        return Err("tensor input has no StorageView descriptor".into());
    }
    let raw = unsafe { &*raw };
    if raw.magic != STORAGE_VIEW_ABI_MAGIC
        || raw.abi_version != STORAGE_VIEW_ABI_VERSION
        || raw.byte_size as usize != mem::size_of::<StorageViewAbi>()
    {
        return Err("StorageView ABI mismatch".into());
    }
    let rank = usize::try_from(raw.rank).map_err(|_| "StorageView rank is too large")?;
    if rank != 0 && (raw.dimensions.is_null() || raw.strides.is_null()) {
        return Err("StorageView shape or strides are missing".into());
    }
    let dimensions = unsafe { slice::from_raw_parts(raw.dimensions, rank) }
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| "StorageView has a negative dimension".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let strides = unsafe { slice::from_raw_parts(raw.strides, rank) }.to_vec();
    StorageView::new(
        raw.data as usize as u64,
        raw.byte_length,
        raw.element,
        dimensions,
        strides,
        raw.offset,
        StorageOwnership::Borrowed,
    )
    .map_err(|error| error.to_string())
}

fn compile_cpu(
    program: &TensorJitProgram,
    graph: &FusionGraph,
    specialization: &KernelSpecialization,
) -> Result<CompiledInstance, String> {
    let target = severian_target::TargetSpec::host();
    let artifact = severian_tensor_compiler::compile_specialized_fusion_cpu(
        graph,
        &program.inputs,
        &program.outputs,
        specialization,
        &target,
    )
    .map_err(|error| error.to_string())?;
    let directory = jit_directory()?;
    let ordinal = NEXT_LIBRARY.fetch_add(1, Ordering::Relaxed);
    let wrapper = directory.join(format!("launcher-{ordinal}.c"));
    let library = directory.join(format!("kernel-{ordinal}.so"));
    fs::write(&wrapper, render_cpu_launcher(program, graph, &artifact)?)
        .map_err(|error| error.to_string())?;
    severian_backend::emit_mlir_shared_library_with_linker_arguments(
        &artifact.module,
        &target.triple,
        &library,
        &[wrapper.to_string_lossy().into_owned()],
    )
    .map_err(|error| error.to_string())?;
    load_compiled_library(&library)
}

fn compile_gpu(
    program: &TensorJitProgram,
    graph: &FusionGraph,
    _specialization: &KernelSpecialization,
) -> Result<CompiledInstance, String> {
    let plan = severian_fusion::plan(graph, severian_fusion::DeviceModel::conservative_gpu());
    if plan.regions.is_empty() {
        return Err("Triton Tensor-JIT graph produced no fusion regions".into());
    }
    let compiler = severian_triton::NativeTritonCompiler::load().map_err(|error| error.to_string())?;
    let target = match program.target {
        TensorJitTarget::Amd => GpuTarget::Amd,
        TensorJitTarget::Nvidia => GpuTarget::Nvidia,
        TensorJitTarget::Cpu => return Err("CPU program reached the Triton provider".into()),
    };
    let options = GpuCompilerOptions {
        target,
        architecture: program.architecture.clone(),
        num_warps: 4,
        warp_size: match target {
            GpuTarget::Amd => {
                severian_triton::AmdTargetFeatures::new(&program.architecture).warp_size()
            }
            GpuTarget::Nvidia => 32,
        },
        num_ctas: 1,
        num_stages: 2,
        emit: if program.target == TensorJitTarget::Amd {
            severian_runtime::gpu::KernelBinaryFormat::Hsaco
        } else {
            severian_runtime::gpu::KernelBinaryFormat::Ptx
        },
        debug: false,
    };
    let driver = native_gpu::NativeGpuDriver::load(target, program.architecture.clone())?;
    let cache = KernelCache::persistent(jit_directory()?.join("gpu-kernels"));
    let runtime = GpuRuntime::new(driver, compiler, cache).map_err(|error| error.to_string())?;
    Ok(CompiledInstance::Gpu(Mutex::new(GpuInstance {
        runtime,
        program: program.clone(),
        graph: graph.clone(),
        plan,
        options,
    })))
}

fn jit_directory() -> Result<PathBuf, String> {
    let directory = std::env::temp_dir().join(format!("severian-tensor-jit-{}", std::process::id()));
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    Ok(directory)
}

fn load_compiled_library(path: &Path) -> Result<CompiledInstance, String> {
    let path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| "compiled library path contains a NUL byte".to_owned())?;
    let library = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
    if library.is_null() {
        return Err(dl_error());
    }
    let symbol = unsafe { dlsym(library, c"sev_tensor_jit_invoke".as_ptr()) };
    if symbol.is_null() {
        unsafe { dlclose(library) };
        return Err(dl_error());
    }
    let launch = unsafe { mem::transmute::<*mut c_void, NativeLaunch>(symbol) };
    Ok(CompiledInstance::Cpu { library, launch })
}

unsafe extern "C" fn launch_region(
    instance: *mut c_void,
    inputs: *const ValueAbi,
    input_count: u32,
    outputs: *mut ValueAbi,
    output_count: u32,
) -> i32 {
    if instance.is_null() {
        return INVALID_ARGUMENT;
    }
    let instance = unsafe { &*instance.cast::<CompiledInstance>() };
    match instance {
        CompiledInstance::Cpu { launch, .. } => unsafe {
            launch(inputs, input_count, outputs, output_count)
        },
        CompiledInstance::Gpu(instance) => {
            let Ok(mut instance) = instance.lock() else {
                return COMPILE_FAILED;
            };
            match unsafe {
                launch_gpu(&mut instance, inputs, input_count, outputs, output_count)
            } {
                Ok(()) => OK,
                Err(error) => {
                    eprintln!("Severian GPU Tensor-JIT launch failed: {error}");
                    COMPILE_FAILED
                }
            }
        }
    }
}

unsafe extern "C" fn destroy_region(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    let instance = unsafe { Box::from_raw(instance.cast::<CompiledInstance>()) };
    if let CompiledInstance::Cpu { library, .. } = *instance {
        unsafe { dlclose(library) };
    }
}

unsafe fn launch_gpu(
    instance: &mut GpuInstance,
    inputs: *const ValueAbi,
    input_count: u32,
    outputs: *mut ValueAbi,
    output_count: u32,
) -> Result<(), String> {
    if input_count as usize != instance.program.inputs.len()
        || output_count as usize != instance.program.outputs.len()
        || (input_count != 0 && inputs.is_null())
        || (output_count != 0 && outputs.is_null())
    {
        return Err("GPU launcher value counts do not match the fusion program".into());
    }
    let inputs = unsafe { slice::from_raw_parts(inputs, input_count as usize) };
    let mut views = Vec::new();
    for (index, node) in instance.program.inputs.iter().copied().enumerate() {
        if !logical_tensor_input(instance.graph.node(node)) {
            continue;
        }
        let raw = unsafe { value_storage(&inputs[index]) }?;
        let view = unsafe { copy_storage_view(raw) }
            .map_err(|error| format!("GPU input {index}: {error}"))?;
        if view.data == 0 && view.byte_length != 0 {
            return Err(format!("GPU input {index} has a null data pointer"));
        }
        views.push((node, view));
    }
    let host_inputs = views
        .iter()
        .map(|(node, view)| {
            let length = usize::try_from(view.byte_length)
                .map_err(|_| format!("GPU input {} byte length exceeds usize", node.0))?;
            let bytes = if length == 0 {
                &[][..]
            } else {
                unsafe { slice::from_raw_parts(view.data as usize as *const u8, length) }
            };
            Ok(HostStorageInput {
                node: *node,
                view: view.clone(),
                bytes,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let execution = instance
        .runtime
        .execute_storage_graph(
            DeviceId(0),
            &instance.graph,
            &instance.plan,
            &host_inputs,
            &instance.options,
        )
        .map_err(|error| error.to_string())?;
    instance
        .runtime
        .synchronize(&execution.execution)
        .map_err(|error| error.to_string())?;
    let outputs = unsafe { slice::from_raw_parts_mut(outputs, output_count as usize) };
    for (index, node) in instance.program.outputs.iter().copied().enumerate() {
        let descriptor = instance.graph.node(node);
        if !logical_tensor_input(descriptor) {
            return Err(format!(
                "GPU output {} ({:?}.{}) is not a tensor storage value",
                node.0, descriptor.kind, descriptor.operation
            ));
        }
        let dimensions = execution
            .specialization
            .shapes
            .iter()
            .find(|shape| shape.node == node)
            .ok_or_else(|| format!("GPU output {} has no specialized shape", node.0))?
            .dimensions
            .clone();
        let strides = execution
            .specialization
            .strides
            .iter()
            .find(|strides| strides.node == node)
            .ok_or_else(|| format!("GPU output {} has no specialized strides", node.0))?
            .strides
            .clone();
        let bytes = tensor_byte_length(node, &dimensions, descriptor.shape.element_bits)?;
        let mut host = vec![0u8; bytes];
        let buffer = execution
            .buffers
            .get(&node)
            .copied()
            .ok_or_else(|| format!("GPU output {} has no device buffer", node.0))?;
        instance
            .runtime
            .download(buffer, 0, &mut host)
            .map_err(|error| error.to_string())?;
        outputs[index] = leak_gpu_output(descriptor, host, dimensions, strides)?;
    }
    let mut buffers = execution.buffers.values().copied().collect::<Vec<_>>();
    buffers.sort_unstable();
    buffers.dedup();
    for buffer in buffers {
        instance
            .runtime
            .deallocate(buffer)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

unsafe fn value_storage(value: &ValueAbi) -> Result<*const StorageViewAbi, String> {
    match value.kind {
        VALUE_STORAGE => Ok(unsafe { value.value.storage }),
        VALUE_POINTER => Ok(unsafe { value.value.pointer.cast::<StorageViewAbi>() }),
        kind => Err(format!("GPU tensor input uses value kind {kind}")),
    }
}

fn tensor_byte_length(node: NodeId, dimensions: &[u64], bits: u16) -> Result<usize, String> {
    dimensions
        .iter()
        .try_fold(1u64, |elements, dimension| elements.checked_mul(*dimension))
        .and_then(|elements| elements.checked_mul(u64::from(bits.div_ceil(8))))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("tensor {} byte size overflows the host ABI", node.0))
}

fn leak_gpu_output(
    node: &severian_fusion::FusionNode,
    mut bytes: Vec<u8>,
    dimensions: Vec<u64>,
    strides: Vec<i64>,
) -> Result<ValueAbi, String> {
    let dimensions = dimensions
        .into_iter()
        .map(|dimension| i64::try_from(dimension).map_err(|_| "GPU dimension exceeds i64"))
        .collect::<Result<Vec<_>, _>>()?;
    let data = bytes.as_mut_ptr();
    let byte_length = bytes.len() as u64;
    mem::forget(bytes);
    let dimensions = Box::leak(dimensions.into_boxed_slice());
    let strides = Box::leak(strides.into_boxed_slice());
    let element = storage_element_representation(node.shape.element_kind, node.shape.element_bits)?;
    let view = Box::new(StorageViewAbi {
        magic: STORAGE_VIEW_ABI_MAGIC,
        abi_version: STORAGE_VIEW_ABI_VERSION,
        byte_size: mem::size_of::<StorageViewAbi>() as u32,
        flags: STORAGE_VIEW_CONTIGUOUS,
        data,
        byte_length,
        rank: dimensions.len() as u64,
        dimensions: dimensions.as_ptr(),
        strides: strides.as_ptr(),
        offset: 0,
        element,
        owner: data.cast(),
    });
    Ok(ValueAbi {
        abi_version: ABI_VERSION,
        byte_size: mem::size_of::<ValueAbi>() as u32,
        kind: VALUE_STORAGE,
        bits: u32::from(node.shape.element_bits),
        value: ValuePayloadAbi {
            storage: Box::into_raw(view),
        },
    })
}

fn storage_element_representation(
    kind: ElementKind,
    bits: u16,
) -> Result<StorageElementRepresentationAbi, String> {
    let (kind, float_format) = match kind {
        ElementKind::SignedInteger => (StorageElementKind::SignedInteger, StorageFloatFormat::None),
        ElementKind::UnsignedInteger | ElementKind::Boolean => {
            (StorageElementKind::UnsignedInteger, StorageFloatFormat::None)
        }
        ElementKind::IeeeFloat => (StorageElementKind::Float, StorageFloatFormat::Ieee),
        ElementKind::BrainFloat => (StorageElementKind::Float, StorageFloatFormat::BrainFloat),
        ElementKind::Float8E4M3Fn => {
            (StorageElementKind::Float, StorageFloatFormat::Float8E4M3Fn)
        }
        ElementKind::Float8E5M2 => {
            (StorageElementKind::Float, StorageFloatFormat::Float8E5M2)
        }
        ElementKind::Opaque => return Err("opaque GPU tensor has no storage representation".into()),
    };
    Ok(StorageElementRepresentationAbi {
        abi_version: STORAGE_VIEW_ABI_VERSION,
        byte_size: mem::size_of::<StorageElementRepresentationAbi>() as u32,
        kind,
        bits: u32::from(bits),
        float_format,
        reserved: 0,
    })
}

fn render_cpu_launcher(
    program: &TensorJitProgram,
    graph: &FusionGraph,
    artifact: &MlirArtifact,
) -> Result<String, String> {
    let header = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or_else(|| "provider repository root is unavailable".to_owned())?
        .join("compiler/runtime/native/tensor_jit.h");
    let mut source = format!(
        "#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n#include \"{}\"\nextern void *__sev_list_create(void);\nextern void __sev_list_push_i64(void *, int64_t);\nextern void __sev_list_push_f64(void *, double);\nstatic double __sev_jit_bf16_to_f64(uint16_t bits) {{ uint32_t word = ((uint32_t)bits) << 16; float value; memcpy(&value, &word, sizeof(value)); return (double)value; }}\n",
        header.display()
    );
    for node in graph.nodes() {
        writeln!(
            source,
            "/* node {} {:?}.{} inputs={:?} rank={:?} element={:?}/{} runtime={:?} */",
            node.id.0,
            node.kind,
            node.operation,
            node.inputs,
            node.shape.rank,
            node.shape.element_kind,
            node.shape.element_bits,
            node.runtime_operands
        )
        .unwrap();
    }
    let mut ranks = artifact
        .inputs
        .iter()
        .chain(&artifact.outputs)
        .filter_map(tensor_rank)
        .collect::<Vec<_>>();
    ranks.sort_unstable();
    ranks.dedup();
    for rank in ranks {
        if rank == 0 {
            writeln!(source, "typedef struct {{ void *allocated; void *aligned; int64_t offset; }} sev_memref_0;").unwrap();
        } else {
            writeln!(source, "typedef struct {{ void *allocated; void *aligned; int64_t offset; int64_t sizes[{rank}]; int64_t strides[{rank}]; }} sev_memref_{rank};").unwrap();
        }
    }
    let result_type = c_result_type(&mut source, &artifact.outputs)?;
    let mut prototype = Vec::new();
    if !artifact.outputs.is_empty() {
        prototype.push(format!("{result_type} *result"));
    }
    for (index, ty) in artifact.inputs.iter().enumerate() {
        prototype.push(match ty {
            LoweredType::Tensor { .. } => format!("{} *arg{index}", c_type(ty)?),
            _ => format!("{} arg{index}", c_type(ty)?),
        });
    }
    writeln!(source, "extern void _mlir_ciface_entry({});", prototype.join(", ")).unwrap();
    source.push_str("int32_t sev_tensor_jit_invoke(const sev_tensor_jit_value_abi *inputs, uint32_t input_count, sev_tensor_jit_value_abi *outputs, uint32_t output_count) {\n");
    writeln!(source, "  if (input_count != {} || output_count != {}) return {INVALID_ARGUMENT};", artifact.inputs.len(), artifact.outputs.len()).unwrap();
    for (index, (ty, node)) in artifact.inputs.iter().zip(&program.inputs).enumerate() {
        render_input(&mut source, index, ty, graph.node(*node))?;
    }
    if !artifact.outputs.is_empty() {
        writeln!(source, "  {result_type} result; memset(&result, 0, sizeof(result));").unwrap();
    }
    let mut call = Vec::new();
    if !artifact.outputs.is_empty() {
        call.push("&result".to_owned());
    }
    for (index, ty) in artifact.inputs.iter().enumerate() {
        call.push(if matches!(ty, LoweredType::Tensor { .. }) {
            format!("&arg{index}")
        } else {
            format!("arg{index}")
        });
    }
    writeln!(source, "  _mlir_ciface_entry({});", call.join(", ")).unwrap();
    for (index, (ty, node)) in artifact.outputs.iter().zip(&program.outputs).enumerate() {
        let output_node = graph.node(*node);
        let tensor_node = if output_node.kind == NodeKind::StorageView
            && matches!(output_node.operation.as_str(), "shape" | "strides" | "values")
        {
            graph.node(
                *output_node
                    .inputs
                    .first()
                    .ok_or_else(|| "storage metadata output has no input".to_owned())?,
            )
        } else {
            output_node
        };
        render_output(
            &mut source,
            index,
            ty,
            output_node,
            tensor_node,
            artifact.outputs.len(),
        )?;
    }
    source.push_str("  return 0;\n}\n");
    Ok(source)
}

fn tensor_rank(ty: &LoweredType) -> Option<usize> {
    match ty {
        LoweredType::Tensor {
            shape: severian_mlir::LoweredTensorShape::Ranked(dimensions),
            ..
        } => Some(dimensions.len()),
        _ => None,
    }
}

fn c_type(ty: &LoweredType) -> Result<String, String> {
    Ok(match ty {
        LoweredType::Tensor { .. } => format!("sev_memref_{}", tensor_rank(ty).ok_or("unranked JIT artifact")?),
        LoweredType::Bytes | LoweredType::String => "void *".into(),
        LoweredType::Boolean => "uint8_t".into(),
        LoweredType::Integer { bits: 1..=8, signed: true } => "int8_t".into(),
        LoweredType::Integer { bits: 1..=8, signed: false } => "uint8_t".into(),
        LoweredType::Integer { bits: 9..=16, signed: true } => "int16_t".into(),
        LoweredType::Integer { bits: 9..=16, signed: false } => "uint16_t".into(),
        LoweredType::Integer { bits: 17..=32, signed: true } => "int32_t".into(),
        LoweredType::Integer { bits: 17..=32, signed: false } => "uint32_t".into(),
        LoweredType::Integer { bits: 33..=64, signed: true } => "int64_t".into(),
        LoweredType::Integer { bits: 33..=64, signed: false } => "uint64_t".into(),
        LoweredType::Float { format: LoweredFloatFormat::Ieee(32) } => "float".into(),
        LoweredType::Float { format: LoweredFloatFormat::Ieee(64) } => "double".into(),
        unsupported => return Err(format!("unsupported CPU Tensor-JIT ABI type {unsupported:?}")),
    })
}

fn c_result_type(source: &mut String, outputs: &[LoweredType]) -> Result<String, String> {
    if outputs.len() == 1 {
        return c_type(&outputs[0]);
    }
    let fields = outputs
        .iter()
        .enumerate()
        .map(|(index, ty)| Ok(format!("{} field{index};", c_type(ty)?)))
        .collect::<Result<Vec<_>, String>>()?
        .join(" ");
    writeln!(source, "typedef struct {{ {fields} }} sev_jit_result;").unwrap();
    Ok("sev_jit_result".into())
}

fn render_input(
    source: &mut String,
    index: usize,
    ty: &LoweredType,
    node: &severian_fusion::FusionNode,
) -> Result<(), String> {
    if let Some(rank) = tensor_rank(ty) {
        writeln!(source, "  sev_jit_storage_view_abi *view{index} = inputs[{index}].kind == SEV_TENSOR_JIT_VALUE_STORAGE ? inputs[{index}].value.storage : (sev_jit_storage_view_abi *)inputs[{index}].value.pointer;").unwrap();
        writeln!(source, "  if (view{index} == NULL || view{index}->rank != {rank}) return {INVALID_ARGUMENT};").unwrap();
        writeln!(source, "  sev_memref_{rank} arg{index}; arg{index}.allocated = view{index}->owner != NULL ? view{index}->owner : (void *)view{index}->data; arg{index}.aligned = (void *)view{index}->data; arg{index}.offset = view{index}->offset;").unwrap();
        for axis in 0..rank {
            writeln!(source, "  arg{index}.sizes[{axis}] = view{index}->dimensions[{axis}]; arg{index}.strides[{axis}] = view{index}->strides[{axis}];").unwrap();
        }
        return Ok(());
    }
    let c = c_type(ty)?;
    let expression = match ty {
        LoweredType::Bytes | LoweredType::String => {
            if node.shape.element_kind == ElementKind::Opaque {
                format!("inputs[{index}].value.pointer")
            } else {
                format!("inputs[{index}].value.storage")
            }
        }
        LoweredType::Integer { signed: true, .. } => format!("inputs[{index}].value.signed_integer"),
        LoweredType::Integer { signed: false, .. } | LoweredType::Boolean => format!("inputs[{index}].value.unsigned_integer"),
        LoweredType::Float { .. } => format!("inputs[{index}].value.floating"),
        _ => return Err(format!("unsupported launcher input {ty:?}")),
    };
    writeln!(source, "  {c} arg{index} = ({c})({expression});").unwrap();
    Ok(())
}

fn render_output(
    source: &mut String,
    index: usize,
    ty: &LoweredType,
    node: &severian_fusion::FusionNode,
    tensor_node: &severian_fusion::FusionNode,
    count: usize,
) -> Result<(), String> {
    let value = if count == 1 {
        "result".to_owned()
    } else {
        format!("result.field{index}")
    };
    writeln!(source, "  outputs[{index}].abi_version = SEV_TENSOR_JIT_ABI_VERSION; outputs[{index}].byte_size = sizeof(sev_tensor_jit_value_abi);").unwrap();
    if node.kind == NodeKind::StorageView
        && matches!(node.operation.as_str(), "shape" | "strides" | "values")
    {
        let rank = tensor_rank(ty).ok_or_else(|| {
            format!("terminal storage metadata requires a ranked tensor, found {ty:?}")
        })?;
        writeln!(source, "  void *list{index} = __sev_list_create();").unwrap();
        if node.operation == "shape" {
            for axis in 0..rank {
                writeln!(source, "  __sev_list_push_i64(list{index}, {value}.sizes[{axis}]);").unwrap();
            }
        } else if node.operation == "strides" {
            for axis in 0..rank {
                writeln!(source, "  __sev_list_push_i64(list{index}, {value}.strides[{axis}]);").unwrap();
            }
        } else {
            writeln!(source, "  uint64_t value_count{index} = 1; for (uint32_t axis = 0; axis < {rank}; ++axis) value_count{index} *= (uint64_t){value}.sizes[axis];").unwrap();
            writeln!(source, "  for (uint64_t linear = 0; linear < value_count{index}; ++linear) {{ uint64_t remaining = linear; int64_t element_offset = {value}.offset;").unwrap();
            for axis in (0..rank).rev() {
                writeln!(source, "    element_offset += (int64_t)(remaining % (uint64_t){value}.sizes[{axis}]) * {value}.strides[{axis}]; remaining /= (uint64_t){value}.sizes[{axis}];").unwrap();
            }
            writeln!(
                source,
                "    __sev_list_push_f64(list{index}, {}); }}",
                c_element_as_f64(
                    tensor_node.shape.element_kind,
                    tensor_node.shape.element_bits,
                    &value,
                )?
            )
            .unwrap();
        }
        let kind = if matches!(node.operation.as_str(), "shape" | "strides") {
            "SEV_TENSOR_JIT_VALUE_LIST_I64"
        } else {
            "SEV_TENSOR_JIT_VALUE_POINTER"
        };
        writeln!(source, "  outputs[{index}].kind = {kind}; outputs[{index}].value.pointer = list{index};").unwrap();
        return Ok(());
    }
    if let Some(rank) = tensor_rank(ty) {
        let (kind, bits, format) = tensor_element(
            tensor_node.shape.element_kind,
            tensor_node.shape.element_bits,
        )?;
        writeln!(source, "  sev_jit_storage_view_abi *out{index} = (sev_jit_storage_view_abi *)calloc(1, sizeof(sev_jit_storage_view_abi) + {rank} * sizeof(int64_t) * 2); if (out{index} == NULL) return 5;").unwrap();
        writeln!(source, "  out{index}->magic = SEV_STORAGE_VIEW_ABI_MAGIC; out{index}->abi_version = SEV_STORAGE_VIEW_ABI_VERSION; out{index}->byte_size = sizeof(sev_jit_storage_view_abi); out{index}->data = (const uint8_t *){value}.aligned; out{index}->rank = {rank}; out{index}->offset = {value}.offset; out{index}->dimensions = (const int64_t *)(out{index} + 1); out{index}->strides = out{index}->dimensions + {rank}; out{index}->owner = {value}.allocated;").unwrap();
        writeln!(source, "  out{index}->element.abi_version = 1; out{index}->element.byte_size = sizeof(sev_jit_element_abi); out{index}->element.kind = {kind}; out{index}->element.bits = {bits}; out{index}->element.float_format = {format};").unwrap();
        writeln!(source, "  uint64_t elements{index} = 1;").unwrap();
        for axis in 0..rank {
            writeln!(source, "  ((int64_t *)out{index}->dimensions)[{axis}] = {value}.sizes[{axis}]; ((int64_t *)out{index}->strides)[{axis}] = {value}.strides[{axis}]; elements{index} *= (uint64_t){value}.sizes[{axis}];").unwrap();
        }
        writeln!(source, "  out{index}->byte_length = elements{index} * (({bits} + 7) / 8); outputs[{index}].kind = SEV_TENSOR_JIT_VALUE_STORAGE; outputs[{index}].bits = {bits}; outputs[{index}].value.storage = out{index};").unwrap();
        return Ok(());
    }
    match ty {
        LoweredType::Bytes | LoweredType::String => writeln!(source, "  outputs[{index}].kind = SEV_TENSOR_JIT_VALUE_POINTER; outputs[{index}].value.pointer = {value};").unwrap(),
        LoweredType::Integer { bits, signed: true } => writeln!(source, "  outputs[{index}].kind = SEV_TENSOR_JIT_VALUE_SIGNED; outputs[{index}].bits = {bits}; outputs[{index}].value.signed_integer = (int64_t){value};").unwrap(),
        LoweredType::Integer { bits, signed: false } => writeln!(source, "  outputs[{index}].kind = SEV_TENSOR_JIT_VALUE_UNSIGNED; outputs[{index}].bits = {bits}; outputs[{index}].value.unsigned_integer = (uint64_t){value};").unwrap(),
        LoweredType::Boolean => writeln!(source, "  outputs[{index}].kind = SEV_TENSOR_JIT_VALUE_UNSIGNED; outputs[{index}].bits = 1; outputs[{index}].value.unsigned_integer = (uint64_t){value};").unwrap(),
        LoweredType::Float { .. } => writeln!(source, "  outputs[{index}].kind = SEV_TENSOR_JIT_VALUE_FLOAT; outputs[{index}].bits = {}; outputs[{index}].value.floating = (double){value};", float_bits(ty)?).unwrap(),
        _ => return Err(format!("unsupported launcher output {ty:?}")),
    }
    Ok(())
}

fn c_element_as_f64(kind: ElementKind, bits: u16, value: &str) -> Result<String, String> {
    let indexed = |ty: &str| format!("(({ty} *){value}.aligned)[element_offset]");
    Ok(match (kind, bits) {
        (ElementKind::SignedInteger, 8) => format!("(double){}", indexed("int8_t")),
        (ElementKind::SignedInteger, 16) => format!("(double){}", indexed("int16_t")),
        (ElementKind::SignedInteger, 32) => format!("(double){}", indexed("int32_t")),
        (ElementKind::SignedInteger, 64) => format!("(double){}", indexed("int64_t")),
        (ElementKind::UnsignedInteger | ElementKind::Boolean, 1 | 8) => {
            format!("(double){}", indexed("uint8_t"))
        }
        (ElementKind::UnsignedInteger, 16) => format!("(double){}", indexed("uint16_t")),
        (ElementKind::UnsignedInteger, 32) => format!("(double){}", indexed("uint32_t")),
        (ElementKind::UnsignedInteger, 64) => format!("(double){}", indexed("uint64_t")),
        (ElementKind::IeeeFloat, 16) => format!("(double){}", indexed("_Float16")),
        (ElementKind::IeeeFloat, 32) => format!("(double){}", indexed("float")),
        (ElementKind::IeeeFloat, 64) => indexed("double"),
        (ElementKind::BrainFloat, 16) => {
            format!("__sev_jit_bf16_to_f64({})", indexed("uint16_t"))
        }
        _ => return Err(format!("values ABI does not support {kind:?}/{bits}")),
    })
}

fn tensor_element(kind: ElementKind, bits: u16) -> Result<(u32, u16, u32), String> {
    Ok(match kind {
        ElementKind::SignedInteger => (StorageElementKind::SignedInteger as u32, bits, StorageFloatFormat::None as u32),
        ElementKind::UnsignedInteger | ElementKind::Boolean => (StorageElementKind::UnsignedInteger as u32, bits, StorageFloatFormat::None as u32),
        ElementKind::IeeeFloat => (StorageElementKind::Float as u32, bits, StorageFloatFormat::Ieee as u32),
        ElementKind::BrainFloat => (StorageElementKind::Float as u32, bits, StorageFloatFormat::BrainFloat as u32),
        ElementKind::Float8E4M3Fn => (StorageElementKind::Float as u32, bits, StorageFloatFormat::Float8E4M3Fn as u32),
        ElementKind::Float8E5M2 => (StorageElementKind::Float as u32, bits, StorageFloatFormat::Float8E5M2 as u32),
        ElementKind::Opaque => return Err("opaque tensor output has no element ABI".into()),
    })
}

fn float_bits(ty: &LoweredType) -> Result<u16, String> {
    match ty {
        LoweredType::Float { format: LoweredFloatFormat::Ieee(bits) } => Ok(*bits),
        _ => Err(format!("unsupported scalar float {ty:?}")),
    }
}

fn dl_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "dynamic loader returned no diagnostic".into()
    } else {
        unsafe { CStr::from_ptr(error) }.to_string_lossy().into_owned()
    }
}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

const _: () = {
    assert!(mem::size_of::<ValueAbi>() == 24);
    assert!(mem::size_of::<RegionAbi>() == 112);
    assert!(mem::size_of::<CompiledAbi>() == 32);
};
