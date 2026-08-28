//! Severian-owned GPU execution services.
//!
//! Drivers implement one target-neutral contract. Triton is a compiler
//! provider, not the owner of devices, memory, launch ordering, or caching.

mod cache;

pub use cache::{CacheKey, KernelCache};

use severian_fusion::{
    Dimension, FusionGraph, FusionPlan, FusionRegion, GpuTarget, KernelSpecialization, NodeId,
    Rank, RegionId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionExecutionId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub target: GpuTarget,
    pub name: String,
    pub architecture: String,
    pub total_memory_bytes: u64,
    pub max_shared_memory_bytes: u64,
    pub warp_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBinaryFormat {
    LlvmIr,
    AmdGcN,
    Hsaco,
    Ptx,
    Cubin,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridPolicy {
    Fixed([u64; 3]),
    Linear {
        output: NodeId,
        elements_per_program: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchRequirements {
    pub grid: GridPolicy,
    pub block: [u32; 3],
    pub num_warps: u32,
    pub warp_size: u32,
    pub num_ctas: u32,
    pub shared_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifact {
    pub format: KernelBinaryFormat,
    pub entry_point: String,
    pub code: Vec<u8>,
    pub launch: LaunchRequirements,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerOptions {
    pub target: GpuTarget,
    pub architecture: String,
    pub num_warps: u32,
    pub warp_size: u32,
    pub num_ctas: u32,
    pub num_stages: u32,
    pub emit: KernelBinaryFormat,
    pub debug: bool,
}

pub struct KernelCompileRequest<'a> {
    pub graph: &'a FusionGraph,
    pub region: &'a FusionRegion,
    pub specialization: &'a KernelSpecialization,
    pub options: &'a CompilerOptions,
}

pub trait GpuCompiler: Send + Sync {
    fn donor_revision(&self) -> &str;

    fn compile(&self, request: &KernelCompileRequest<'_>) -> Result<KernelArtifact, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelArgument {
    Buffer { buffer: BufferId, byte_offset: u64 },
    Scalar { bytes: Vec<u8>, alignment: u8 },
}

impl KernelArgument {
    pub fn scalar<T: ScalarArgument>(value: T) -> Self {
        Self::Scalar {
            bytes: value.to_ne_bytes(),
            alignment: value.alignment(),
        }
    }
}

pub trait ScalarArgument: Copy {
    fn to_ne_bytes(self) -> Vec<u8>;
    fn alignment(self) -> u8;
}

macro_rules! scalar_argument {
    ($($type:ty),* $(,)?) => {
        $(
            impl ScalarArgument for $type {
                fn to_ne_bytes(self) -> Vec<u8> {
                    <$type>::to_ne_bytes(self).to_vec()
                }

                fn alignment(self) -> u8 {
                    std::mem::align_of::<$type>() as u8
                }
            }
        )*
    };
}

scalar_argument!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedArguments {
    /// Host-side values passed to CUDA/HIP's kernel-parameter ABI.
    pub storage: Vec<u8>,
    /// One offset per kernel argument into `storage`.
    pub offsets: Vec<usize>,
}

impl PackedArguments {
    pub fn value(&self, index: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(index)?;
        let end = self
            .offsets
            .get(index + 1)
            .copied()
            .unwrap_or(self.storage.len());
        self.storage.get(start..end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchCommand {
    pub kernel: KernelId,
    pub grid: [u64; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u64,
    pub arguments: PackedArguments,
    pub dependencies: Vec<EventId>,
}

/// Driver implementations may wrap CUDA, HIP, a remote executor, or a test
/// device. Severian retains ownership of scheduling and argument layout.
pub trait GpuDriver: Send {
    fn discover_devices(&self) -> Result<Vec<DeviceInfo>, String>;
    fn allocate(
        &mut self,
        device: DeviceId,
        bytes: u64,
        alignment: u64,
    ) -> Result<BufferId, String>;
    fn deallocate(&mut self, buffer: BufferId) -> Result<(), String>;
    fn upload(&mut self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), String>;
    fn download(&mut self, buffer: BufferId, offset: u64, data: &mut [u8]) -> Result<(), String>;
    fn device_address(&self, buffer: BufferId) -> Result<u64, String>;
    fn load_kernel(
        &mut self,
        device: DeviceId,
        artifact: &KernelArtifact,
    ) -> Result<KernelId, String>;
    fn unload_kernel(&mut self, kernel: KernelId) -> Result<(), String>;
    fn launch(&mut self, command: &LaunchCommand) -> Result<EventId, String>;
    fn wait(&mut self, events: &[EventId]) -> Result<(), String>;
}

#[derive(Debug)]
pub enum RuntimeError {
    Driver(String),
    Compiler(String),
    Cache(String),
    InvalidDevice(DeviceId),
    TargetMismatch,
    InvalidAlignment(u64),
    AddressOverflow,
    GridOverflow,
    MissingOutputShape(NodeId),
    DuplicateExecution(RegionExecutionId),
    UnknownDependency {
        execution: RegionExecutionId,
        dependency: RegionExecutionId,
    },
    CyclicRegionDependencies,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(formatter, "GPU driver failed: {error}"),
            Self::Compiler(error) => write!(formatter, "GPU compiler failed: {error}"),
            Self::Cache(error) => write!(formatter, "GPU kernel cache failed: {error}"),
            Self::InvalidDevice(device) => write!(formatter, "unknown GPU device {}", device.0),
            Self::TargetMismatch => formatter.write_str("GPU target does not match the device"),
            Self::InvalidAlignment(alignment) => {
                write!(
                    formatter,
                    "invalid GPU argument/buffer alignment {alignment}"
                )
            }
            Self::AddressOverflow => formatter.write_str("GPU buffer address overflow"),
            Self::GridOverflow => formatter.write_str("GPU launch grid overflow"),
            Self::MissingOutputShape(node) => {
                write!(formatter, "node {} has no concrete runtime shape", node.0)
            }
            Self::DuplicateExecution(execution) => {
                write!(formatter, "duplicate GPU execution id {}", execution.0)
            }
            Self::UnknownDependency {
                execution,
                dependency,
            } => write!(
                formatter,
                "GPU execution {} depends on unscheduled execution {}",
                execution.0, dependency.0
            ),
            Self::CyclicRegionDependencies => {
                formatter.write_str("fusion regions contain a cyclic GPU dependency")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionSchedule {
    pub region: RegionId,
    pub dependencies: Vec<RegionId>,
}

/// Derives execution dependencies from values crossing fusion-region
/// boundaries and returns a stable topological order.
pub fn schedule_fusion_regions(
    graph: &FusionGraph,
    plan: &FusionPlan,
) -> Result<Vec<RegionSchedule>, RuntimeError> {
    let mut dependencies = plan
        .regions
        .iter()
        .map(|region| (region.id, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    for region in &plan.regions {
        let region_dependencies = dependencies
            .get_mut(&region.id)
            .expect("every fusion region has a dependency set");
        for input in &region.inputs {
            if input.0 as usize >= graph.nodes().len() {
                continue;
            }
            if let Some(producer) = plan.node_regions.get(input.0 as usize).copied().flatten() {
                if producer != region.id {
                    region_dependencies.insert(producer);
                }
            }
        }
    }

    let mut scheduled = Vec::with_capacity(plan.regions.len());
    let mut emitted = BTreeSet::new();
    while scheduled.len() != plan.regions.len() {
        let next = dependencies
            .iter()
            .find(|(region, required)| {
                !emitted.contains(*region) && required.iter().all(|region| emitted.contains(region))
            })
            .map(|(region, required)| (*region, required.iter().copied().collect::<Vec<_>>()));
        let Some((region, required)) = next else {
            return Err(RuntimeError::CyclicRegionDependencies);
        };
        emitted.insert(region);
        scheduled.push(RegionSchedule {
            region,
            dependencies: required,
        });
    }
    Ok(scheduled)
}

impl std::error::Error for RuntimeError {}

pub struct RegionInvocation<'a> {
    pub id: RegionExecutionId,
    pub graph: &'a FusionGraph,
    pub region: &'a FusionRegion,
    pub specialization: &'a KernelSpecialization,
    pub options: &'a CompilerOptions,
    pub arguments: Vec<KernelArgument>,
    pub dependencies: Vec<RegionExecutionId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    pub events: BTreeMap<RegionExecutionId, EventId>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

pub struct GpuRuntime<D, C> {
    driver: D,
    compiler: C,
    devices: Vec<DeviceInfo>,
    cache: KernelCache,
    loaded: BTreeMap<(DeviceId, CacheKey), KernelId>,
}

impl<D: GpuDriver, C: GpuCompiler> GpuRuntime<D, C> {
    pub fn new(driver: D, compiler: C, cache: KernelCache) -> Result<Self, RuntimeError> {
        let devices = driver.discover_devices().map_err(RuntimeError::Driver)?;
        Ok(Self {
            driver,
            compiler,
            devices,
            cache,
            loaded: BTreeMap::new(),
        })
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    pub fn driver(&self) -> &D {
        &self.driver
    }

    pub fn driver_mut(&mut self) -> &mut D {
        &mut self.driver
    }

    pub fn allocate(
        &mut self,
        device: DeviceId,
        bytes: u64,
        alignment: u64,
    ) -> Result<BufferId, RuntimeError> {
        self.device(device)?;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(RuntimeError::InvalidAlignment(alignment));
        }
        self.driver
            .allocate(device, bytes, alignment)
            .map_err(RuntimeError::Driver)
    }

    pub fn deallocate(&mut self, buffer: BufferId) -> Result<(), RuntimeError> {
        self.driver.deallocate(buffer).map_err(RuntimeError::Driver)
    }

    pub fn upload(
        &mut self,
        buffer: BufferId,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RuntimeError> {
        self.driver
            .upload(buffer, offset, data)
            .map_err(RuntimeError::Driver)
    }

    pub fn download(
        &mut self,
        buffer: BufferId,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), RuntimeError> {
        self.driver
            .download(buffer, offset, data)
            .map_err(RuntimeError::Driver)
    }

    pub fn execute(
        &mut self,
        device: DeviceId,
        invocations: &[RegionInvocation<'_>],
    ) -> Result<ExecutionResult, RuntimeError> {
        let device_info = self.device(device)?.clone();
        let mut events = BTreeMap::new();
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        for invocation in invocations {
            if events.contains_key(&invocation.id) {
                return Err(RuntimeError::DuplicateExecution(invocation.id));
            }
            if invocation.options.target != device_info.target
                || invocation.specialization.target != device_info.target
                || invocation.options.architecture != device_info.architecture
            {
                return Err(RuntimeError::TargetMismatch);
            }
            let dependencies = invocation
                .dependencies
                .iter()
                .map(|dependency| {
                    events
                        .get(dependency)
                        .copied()
                        .ok_or(RuntimeError::UnknownDependency {
                            execution: invocation.id,
                            dependency: *dependency,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let key = CacheKey::for_kernel(
                invocation.graph,
                invocation.region,
                invocation.specialization,
                invocation.options,
                self.compiler.donor_revision(),
            );
            let artifact = if let Some(artifact) = self
                .cache
                .get(&key)
                .map_err(|error| RuntimeError::Cache(error.to_string()))?
            {
                cache_hits += 1;
                artifact
            } else {
                cache_misses += 1;
                let request = KernelCompileRequest {
                    graph: invocation.graph,
                    region: invocation.region,
                    specialization: invocation.specialization,
                    options: invocation.options,
                };
                let artifact = self
                    .compiler
                    .compile(&request)
                    .map_err(RuntimeError::Compiler)?;
                self.cache
                    .insert(key, artifact.clone())
                    .map_err(|error| RuntimeError::Cache(error.to_string()))?;
                artifact
            };
            let kernel = if let Some(kernel) = self.loaded.get(&(device, key)).copied() {
                kernel
            } else {
                let kernel = self
                    .driver
                    .load_kernel(device, &artifact)
                    .map_err(RuntimeError::Driver)?;
                self.loaded.insert((device, key), kernel);
                kernel
            };
            let arguments = pack_arguments(&self.driver, &invocation.arguments)?;
            let grid = calculate_grid(
                invocation.graph,
                invocation.specialization,
                &artifact.launch.grid,
            )?;
            let event = self
                .driver
                .launch(&LaunchCommand {
                    kernel,
                    grid,
                    block: artifact.launch.block,
                    shared_memory_bytes: artifact.launch.shared_memory_bytes,
                    arguments,
                    dependencies,
                })
                .map_err(RuntimeError::Driver)?;
            events.insert(invocation.id, event);
        }
        Ok(ExecutionResult {
            events,
            cache_hits,
            cache_misses,
        })
    }

    pub fn synchronize(&mut self, result: &ExecutionResult) -> Result<(), RuntimeError> {
        self.driver
            .wait(&result.events.values().copied().collect::<Vec<_>>())
            .map_err(RuntimeError::Driver)
    }

    pub fn unload_all(&mut self) -> Result<(), RuntimeError> {
        let kernels = std::mem::take(&mut self.loaded);
        for (_, kernel) in kernels {
            self.driver
                .unload_kernel(kernel)
                .map_err(RuntimeError::Driver)?;
        }
        Ok(())
    }

    fn device(&self, id: DeviceId) -> Result<&DeviceInfo, RuntimeError> {
        self.devices
            .iter()
            .find(|device| device.id == id)
            .ok_or(RuntimeError::InvalidDevice(id))
    }
}

pub fn pack_arguments(
    driver: &impl GpuDriver,
    arguments: &[KernelArgument],
) -> Result<PackedArguments, RuntimeError> {
    let mut storage = Vec::new();
    let mut offsets = Vec::with_capacity(arguments.len());
    for argument in arguments {
        let (bytes, alignment) = match argument {
            KernelArgument::Buffer {
                buffer,
                byte_offset,
            } => {
                let address = driver
                    .device_address(*buffer)
                    .map_err(RuntimeError::Driver)?
                    .checked_add(*byte_offset)
                    .ok_or(RuntimeError::AddressOverflow)?;
                (address.to_ne_bytes().to_vec(), 8usize)
            }
            KernelArgument::Scalar { bytes, alignment } => {
                let alignment = usize::from(*alignment);
                if alignment == 0 || !alignment.is_power_of_two() {
                    return Err(RuntimeError::InvalidAlignment(alignment as u64));
                }
                (bytes.clone(), alignment)
            }
        };
        let padding = (alignment - storage.len() % alignment) % alignment;
        storage.resize(storage.len() + padding, 0);
        offsets.push(storage.len());
        storage.extend_from_slice(&bytes);
    }
    Ok(PackedArguments { storage, offsets })
}

pub fn calculate_grid(
    graph: &FusionGraph,
    specialization: &KernelSpecialization,
    policy: &GridPolicy,
) -> Result<[u64; 3], RuntimeError> {
    match policy {
        GridPolicy::Fixed(grid) => Ok(*grid),
        GridPolicy::Linear {
            output,
            elements_per_program,
        } => {
            if *elements_per_program == 0 {
                return Err(RuntimeError::GridOverflow);
            }
            let elements = concrete_elements(graph, specialization, *output)?;
            let programs = elements
                .checked_add(elements_per_program - 1)
                .ok_or(RuntimeError::GridOverflow)?
                / elements_per_program;
            Ok([programs.max(1), 1, 1])
        }
    }
}

fn concrete_elements(
    graph: &FusionGraph,
    specialization: &KernelSpecialization,
    node: NodeId,
) -> Result<u64, RuntimeError> {
    let descriptor = graph.node(node);
    let runtime = specialization
        .shapes
        .iter()
        .find(|shape| shape.node == node)
        .map(|shape| shape.dimensions.as_slice());
    let dimensions = match &descriptor.shape.rank {
        Rank::Unranked => runtime
            .ok_or(RuntimeError::MissingOutputShape(node))?
            .to_vec(),
        Rank::Ranked(dimensions) => dimensions
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                Dimension::Known(value) => Ok(*value),
                Dimension::Dynamic => runtime
                    .and_then(|dimensions| dimensions.get(axis))
                    .copied()
                    .ok_or(RuntimeError::MissingOutputShape(node)),
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    dimensions
        .into_iter()
        .try_fold(1u64, |elements, dimension| {
            elements
                .checked_mul(dimension)
                .ok_or(RuntimeError::GridOverflow)
        })
}

#[cfg(test)]
mod tests;
