use super::*;
use severian_fusion::{
    plan, ContractionDimension, DeviceModel, ElementKind, FusionGraph, FusionNode,
    KernelSpecialization, Matmul, NodeKind, OperandRole, Rank, RuntimeOperand, RuntimeShape, Shape,
    StorageLayout,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Default)]
struct MockDriver {
    next_buffer: u64,
    next_kernel: u64,
    next_event: u64,
    buffers: BTreeMap<BufferId, Vec<u8>>,
    launches: Vec<LaunchCommand>,
    loaded: usize,
}

impl GpuDriver for MockDriver {
    fn discover_devices(&self) -> Result<Vec<DeviceInfo>, String> {
        Ok(vec![DeviceInfo {
            id: DeviceId(0),
            target: GpuTarget::Nvidia,
            name: "mock-nvidia".into(),
            architecture: "sm_80".into(),
            total_memory_bytes: 8 << 30,
            max_shared_memory_bytes: 64 << 10,
            warp_size: 32,
        }])
    }

    fn allocate(
        &mut self,
        _device: DeviceId,
        bytes: u64,
        _alignment: u64,
    ) -> Result<BufferId, String> {
        let id = BufferId(self.next_buffer);
        self.next_buffer += 1;
        self.buffers.insert(id, vec![0; bytes as usize]);
        Ok(id)
    }

    fn deallocate(&mut self, buffer: BufferId) -> Result<(), String> {
        self.buffers
            .remove(&buffer)
            .ok_or_else(|| "unknown buffer".to_owned())?;
        Ok(())
    }

    fn upload(&mut self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), String> {
        let storage = self
            .buffers
            .get_mut(&buffer)
            .ok_or_else(|| "unknown buffer".to_owned())?;
        let start = offset as usize;
        storage[start..start + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn download(&mut self, buffer: BufferId, offset: u64, data: &mut [u8]) -> Result<(), String> {
        let storage = self
            .buffers
            .get(&buffer)
            .ok_or_else(|| "unknown buffer".to_owned())?;
        let start = offset as usize;
        data.copy_from_slice(&storage[start..start + data.len()]);
        Ok(())
    }

    fn device_address(&self, buffer: BufferId) -> Result<u64, String> {
        self.buffers
            .contains_key(&buffer)
            .then_some(0x1000_0000 + buffer.0 * 0x10000)
            .ok_or_else(|| "unknown buffer".to_owned())
    }

    fn load_kernel(
        &mut self,
        _device: DeviceId,
        _artifact: &KernelArtifact,
    ) -> Result<KernelId, String> {
        let id = KernelId(self.next_kernel);
        self.next_kernel += 1;
        self.loaded += 1;
        Ok(id)
    }

    fn unload_kernel(&mut self, _kernel: KernelId) -> Result<(), String> {
        Ok(())
    }

    fn launch(&mut self, command: &LaunchCommand) -> Result<EventId, String> {
        self.launches.push(command.clone());
        let event = EventId(self.next_event);
        self.next_event += 1;
        Ok(event)
    }

    fn wait(&mut self, _events: &[EventId]) -> Result<(), String> {
        Ok(())
    }
}

struct MockCompiler {
    count: Arc<AtomicUsize>,
}

fn f32_storage_view(dimensions: Vec<u64>) -> crate::StorageView {
    let mut stride = 1i64;
    let mut strides = vec![0; dimensions.len()];
    for axis in (0..dimensions.len()).rev() {
        strides[axis] = stride;
        stride *= dimensions[axis] as i64;
    }
    crate::StorageView::new(
        0x1000,
        dimensions.iter().product::<u64>() * 4,
        crate::StorageElementRepresentationAbi {
            abi_version: crate::STORAGE_VIEW_ABI_VERSION,
            byte_size: core::mem::size_of::<crate::StorageElementRepresentationAbi>() as u32,
            kind: crate::StorageElementKind::Float,
            bits: 32,
            float_format: crate::StorageFloatFormat::Ieee,
            reserved: 0,
        },
        dimensions,
        strides,
        0,
        crate::StorageOwnership::Borrowed,
    )
    .unwrap()
}

fn i64_storage_view(dimensions: Vec<u64>) -> crate::StorageView {
    let mut stride = 1i64;
    let mut strides = vec![0; dimensions.len()];
    for axis in (0..dimensions.len()).rev() {
        strides[axis] = stride;
        stride *= dimensions[axis] as i64;
    }
    crate::StorageView::new(
        0x2000,
        dimensions.iter().product::<u64>() * 8,
        crate::StorageElementRepresentationAbi {
            abi_version: crate::STORAGE_VIEW_ABI_VERSION,
            byte_size: core::mem::size_of::<crate::StorageElementRepresentationAbi>() as u32,
            kind: crate::StorageElementKind::SignedInteger,
            bits: 64,
            float_format: crate::StorageFloatFormat::None,
            reserved: 0,
        },
        dimensions,
        strides,
        0,
        crate::StorageOwnership::Borrowed,
    )
    .unwrap()
}

#[test]
fn storage_view_metadata_creates_the_kernel_specialization_before_compilation() {
    let mut parameter = FusionNode::structural(
        0,
        NodeKind::Parameter,
        [],
        Shape {
            rank: Rank::Unranked,
            element_kind: ElementKind::BrainFloat,
            element_bits: 16,
        },
    );
    parameter.layout = StorageLayout::Runtime;
    let graph = FusionGraph::new(vec![parameter]).unwrap();
    let view = crate::StorageView::new(
        0x1000,
        2_097_152,
        crate::StorageElementRepresentationAbi {
            abi_version: crate::STORAGE_VIEW_ABI_VERSION,
            byte_size: core::mem::size_of::<crate::StorageElementRepresentationAbi>() as u32,
            kind: crate::StorageElementKind::Float,
            bits: 16,
            float_format: crate::StorageFloatFormat::BrainFloat,
            reserved: 0,
        },
        vec![1, 16, 512, 128],
        vec![1_048_576, 65_536, 128, 1],
        0,
        crate::StorageOwnership::Borrowed,
    )
    .unwrap();
    let prepared = prepare_storage_inputs(
        &graph,
        GpuTarget::Nvidia,
        &[StorageSpecializationBinding {
            node: NodeId(0),
            view,
        }],
    )
    .unwrap();
    let specialization = prepared.specialization;
    assert_eq!(
        specialization.shapes,
        vec![RuntimeShape {
            node: NodeId(0),
            dimensions: vec![1, 16, 512, 128],
        }]
    );
    assert_eq!(
        specialization.strides[0].strides,
        [1_048_576, 65_536, 128, 1]
    );
    assert_eq!(specialization.target, GpuTarget::Nvidia);
    let packed = pack_arguments(&MockDriver::default(), &prepared.arguments).unwrap();
    assert_eq!(packed.value(0), Some(0x1000u64.to_ne_bytes().as_slice()));
}

#[test]
fn storage_specialization_propagates_elementwise_result_shapes_and_strides() {
    let runtime = || {
        let mut node = FusionNode::structural(
            0,
            NodeKind::Parameter,
            [],
            Shape::unranked(ElementKind::IeeeFloat, 32),
        );
        node.layout = StorageLayout::Runtime;
        node
    };
    let left = runtime();
    let mut right = runtime();
    right.id = NodeId(1);
    let mut add = FusionNode::structural(
        2,
        NodeKind::Elementwise,
        [NodeId(0), NodeId(1)],
        Shape::unranked(ElementKind::IeeeFloat, 32),
    );
    add.operation = "add".into();
    add.layout = StorageLayout::Runtime;
    let graph = FusionGraph::new(vec![left, right, add]).unwrap();
    let specialization = specialize_storage_views(
        &graph,
        GpuTarget::Nvidia,
        &[
            StorageSpecializationBinding {
                node: NodeId(0),
                view: f32_storage_view(vec![2, 1, 4]),
            },
            StorageSpecializationBinding {
                node: NodeId(1),
                view: f32_storage_view(vec![1, 3, 4]),
            },
        ],
    )
    .unwrap();
    assert_eq!(
        specialization
            .shapes
            .iter()
            .find(|shape| shape.node == NodeId(2))
            .unwrap()
            .dimensions,
        [2, 3, 4]
    );
    let output = specialization
        .strides
        .iter()
        .find(|strides| strides.node == NodeId(2))
        .unwrap();
    assert_eq!(output.strides, [12, 4, 1]);
    assert_eq!(output.offset, 0);
}

#[test]
fn storage_specialization_propagates_rank_generic_batched_matmul() {
    let parameter = |id| {
        let mut node = FusionNode::structural(
            id,
            NodeKind::Parameter,
            [],
            Shape::unranked(ElementKind::IeeeFloat, 32),
        );
        node.layout = StorageLayout::Runtime;
        node
    };
    let mut matmul = FusionNode::structural(
        2,
        NodeKind::Contraction,
        [NodeId(0), NodeId(1)],
        Shape::unranked(ElementKind::IeeeFloat, 32),
    );
    matmul.operation = "matmul".into();
    matmul.layout = StorageLayout::Runtime;
    matmul.matmul = Some(Matmul {
        lhs_shape: Rank::Unranked,
        rhs_shape: Rank::Unranked,
        result_shape: Rank::Unranked,
        batch_dimensions: Vec::new(),
        contraction_dimensions: vec![ContractionDimension { lhs: 0, rhs: 0 }],
    });
    let graph = FusionGraph::new(vec![parameter(0), parameter(1), matmul]).unwrap();
    let specialization = specialize_storage_views(
        &graph,
        GpuTarget::Nvidia,
        &[
            StorageSpecializationBinding {
                node: NodeId(0),
                view: f32_storage_view(vec![2, 1, 4, 8]),
            },
            StorageSpecializationBinding {
                node: NodeId(1),
                view: f32_storage_view(vec![1, 3, 8, 16]),
            },
        ],
    )
    .unwrap();
    assert_eq!(
        specialization
            .shapes
            .iter()
            .find(|shape| shape.node == NodeId(2))
            .unwrap()
            .dimensions,
        [2, 3, 4, 16]
    );
}

#[test]
fn runtime_operand_data_specializes_every_shape_changing_structural_class() {
    let parameter = |id, kind, bits| {
        let mut node =
            FusionNode::structural(id, NodeKind::Parameter, [], Shape::unranked(kind, bits));
        node.layout = StorageLayout::Runtime;
        node
    };
    let constant = |id| {
        FusionNode::structural(
            id,
            NodeKind::Constant,
            [],
            Shape::typed([], ElementKind::Opaque, 64),
        )
    };
    let runtime_node =
        |id, kind, operation: &str, inputs: Vec<NodeId>, operands: Vec<RuntimeOperand>| {
            let mut node = FusionNode::structural(
                id,
                kind,
                inputs,
                Shape::unranked(ElementKind::IeeeFloat, 32),
            );
            node.operation = operation.into();
            node.runtime_operands = operands;
            for operand in &node.runtime_operands {
                node.operand_roles[usize::from(operand.input_index)] = OperandRole::RuntimeShape;
            }
            node.layout = StorageLayout::Runtime;
            node
        };
    let graph = FusionGraph::new(vec![
        parameter(0, ElementKind::IeeeFloat, 32),
        parameter(1, ElementKind::IeeeFloat, 32),
        parameter(2, ElementKind::SignedInteger, 64),
        constant(3),
        runtime_node(
            4,
            NodeKind::Reshape,
            "reshape",
            vec![NodeId(0), NodeId(3)],
            vec![RuntimeOperand {
                input_index: 1,
                values: vec![1, 4],
            }],
        ),
        constant(5),
        runtime_node(
            6,
            NodeKind::Permute,
            "axes",
            vec![NodeId(0), NodeId(5)],
            vec![RuntimeOperand {
                input_index: 1,
                values: vec![1, 0],
            }],
        ),
        constant(7),
        constant(8),
        constant(9),
        runtime_node(
            10,
            NodeKind::Slice,
            "slice",
            vec![NodeId(0), NodeId(7), NodeId(8), NodeId(9)],
            vec![
                RuntimeOperand {
                    input_index: 1,
                    values: vec![0, 1],
                },
                RuntimeOperand {
                    input_index: 2,
                    values: vec![2, 2],
                },
                RuntimeOperand {
                    input_index: 3,
                    values: vec![1, 1],
                },
            ],
        ),
        constant(11),
        runtime_node(
            12,
            NodeKind::Broadcast,
            "repeat",
            vec![NodeId(0), NodeId(11)],
            vec![RuntimeOperand {
                input_index: 1,
                values: vec![0, 2],
            }],
        ),
        runtime_node(
            13,
            NodeKind::Gather,
            "gather",
            vec![NodeId(0), NodeId(2)],
            Vec::new(),
        ),
        constant(14),
        runtime_node(
            15,
            NodeKind::Concatenate,
            "concatenate",
            vec![NodeId(0), NodeId(1), NodeId(14)],
            vec![RuntimeOperand {
                input_index: 2,
                values: vec![0],
            }],
        ),
        runtime_node(
            16,
            NodeKind::Broadcast,
            "like",
            vec![NodeId(0), NodeId(1)],
            Vec::new(),
        ),
    ])
    .unwrap();
    let specialization = specialize_storage_views(
        &graph,
        GpuTarget::Nvidia,
        &[
            StorageSpecializationBinding {
                node: NodeId(0),
                view: f32_storage_view(vec![2, 2]),
            },
            StorageSpecializationBinding {
                node: NodeId(1),
                view: f32_storage_view(vec![2, 2]),
            },
            StorageSpecializationBinding {
                node: NodeId(2),
                view: i64_storage_view(vec![2]),
            },
        ],
    )
    .unwrap();
    let shape = |node| {
        specialization
            .shapes
            .iter()
            .find(|shape| shape.node == NodeId(node))
            .unwrap()
            .dimensions
            .clone()
    };
    assert_eq!(shape(4), [1, 4]);
    assert_eq!(shape(6), [2, 2]);
    assert_eq!(shape(10), [2, 1]);
    assert_eq!(shape(12), [4, 2]);
    assert_eq!(shape(13), [2, 2]);
    assert_eq!(shape(15), [4, 2]);
    assert_eq!(shape(16), [2, 2]);
    let slice = specialization
        .strides
        .iter()
        .find(|strides| strides.node == NodeId(10))
        .unwrap();
    assert_eq!(slice.strides, [2, 1]);
    assert_eq!(slice.offset, 1);
}

impl GpuCompiler for MockCompiler {
    fn donor_revision(&self) -> &str {
        "donor-revision-a"
    }

    fn compile(&self, request: &KernelCompileRequest<'_>) -> Result<KernelArtifact, String> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(KernelArtifact {
            format: KernelBinaryFormat::Ptx,
            entry_point: format!("severian_region_{}", request.region.id.0),
            code: b"// mock ptx".to_vec(),
            launch: LaunchRequirements {
                grid: GridPolicy::Linear {
                    output: request.region.outputs[0],
                    elements_per_program: 256,
                },
                block: [128, 1, 1],
                num_warps: 4,
                warp_size: 32,
                num_ctas: 1,
                shared_memory_bytes: 1024,
            },
        })
    }
}

fn graph(element: ElementKind, bits: u16) -> FusionGraph {
    let shape = || Shape::typed([Dimension::Known(1024)], element, bits);
    let mut add = FusionNode::structural(2, NodeKind::Elementwise, [NodeId(0), NodeId(1)], shape());
    add.operation = "add".into();
    FusionGraph::new(vec![
        FusionNode::structural(0, NodeKind::Parameter, [], shape()),
        FusionNode::structural(1, NodeKind::Parameter, [], shape()),
        add,
    ])
    .unwrap()
}

fn options() -> CompilerOptions {
    CompilerOptions {
        target: GpuTarget::Nvidia,
        architecture: "sm_80".into(),
        num_warps: 4,
        warp_size: 32,
        num_ctas: 1,
        num_stages: 3,
        emit: KernelBinaryFormat::Ptx,
        debug: false,
    }
}

#[test]
fn runtime_owns_buffers_transfers_and_argument_packing() {
    let count = Arc::new(AtomicUsize::new(0));
    let mut runtime = GpuRuntime::new(
        MockDriver::default(),
        MockCompiler { count },
        KernelCache::memory(),
    )
    .unwrap();
    let buffer = runtime.allocate(DeviceId(0), 16, 256).unwrap();
    runtime.upload(buffer, 4, &[1, 2, 3, 4]).unwrap();
    let mut result = [0; 4];
    runtime.download(buffer, 4, &mut result).unwrap();
    assert_eq!(result, [1, 2, 3, 4]);

    let packed = pack_arguments(
        runtime.driver(),
        &[
            KernelArgument::Buffer {
                buffer,
                byte_offset: 12,
            },
            KernelArgument::scalar(7u32),
        ],
    )
    .unwrap();
    assert_eq!(packed.offsets, [0, 8]);
    assert_eq!(
        packed.value(0).unwrap(),
        &(0x1000_0000u64 + 12).to_ne_bytes()
    );
    assert_eq!(packed.value(1).unwrap(), &7u32.to_ne_bytes());
}

#[test]
fn schedules_dependencies_and_reuses_compiled_and_loaded_kernels() {
    let graph = graph(ElementKind::IeeeFloat, 32);
    let fusion = plan(&graph, DeviceModel::conservative_gpu());
    let region = &fusion.regions[0];
    let specialization = KernelSpecialization {
        shapes: Vec::new(),
        strides: Vec::new(),
        target: GpuTarget::Nvidia,
    };
    let options = options();
    let count = Arc::new(AtomicUsize::new(0));
    let mut runtime = GpuRuntime::new(
        MockDriver::default(),
        MockCompiler {
            count: Arc::clone(&count),
        },
        KernelCache::memory(),
    )
    .unwrap();
    let input = runtime.allocate(DeviceId(0), 4096, 256).unwrap();
    let output = runtime.allocate(DeviceId(0), 4096, 256).unwrap();
    let arguments = || {
        vec![
            KernelArgument::Buffer {
                buffer: input,
                byte_offset: 0,
            },
            KernelArgument::Buffer {
                buffer: input,
                byte_offset: 0,
            },
            KernelArgument::Buffer {
                buffer: output,
                byte_offset: 0,
            },
        ]
    };
    let result = runtime
        .execute(
            DeviceId(0),
            &[
                RegionInvocation {
                    id: RegionExecutionId(0),
                    graph: &graph,
                    region,
                    specialization: &specialization,
                    options: &options,
                    arguments: arguments(),
                    dependencies: Vec::new(),
                },
                RegionInvocation {
                    id: RegionExecutionId(1),
                    graph: &graph,
                    region,
                    specialization: &specialization,
                    options: &options,
                    arguments: arguments(),
                    dependencies: vec![RegionExecutionId(0)],
                },
            ],
        )
        .unwrap();
    assert_eq!(result.cache_misses, 1);
    assert_eq!(result.cache_hits, 1);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.driver().loaded, 1);
    assert_eq!(runtime.driver().launches[0].grid, [4, 1, 1]);
    assert!(runtime.driver().launches[0].dependencies.is_empty());
    assert_eq!(runtime.driver().launches[1].dependencies, [EventId(0)]);
}

#[test]
fn storage_view_to_specialization_cache_launcher_and_execution_is_one_path() {
    let graph = graph(ElementKind::IeeeFloat, 32);
    let fusion = plan(&graph, DeviceModel::conservative_gpu());
    let options = options();
    let count = Arc::new(AtomicUsize::new(0));
    let mut runtime = GpuRuntime::new(
        MockDriver::default(),
        MockCompiler {
            count: Arc::clone(&count),
        },
        KernelCache::memory(),
    )
    .unwrap();
    let left_bytes = vec![1u8; 4096];
    let right_bytes = vec![2u8; 4096];
    let inputs = [
        HostStorageInput {
            node: NodeId(0),
            view: f32_storage_view(vec![1024]),
            bytes: &left_bytes,
        },
        HostStorageInput {
            node: NodeId(1),
            view: f32_storage_view(vec![1024]),
            bytes: &right_bytes,
        },
    ];

    let first = runtime
        .execute_storage_graph(DeviceId(0), &graph, &fusion, &inputs, &options)
        .unwrap();
    assert_eq!(first.execution.cache_misses, 1);
    assert_eq!(first.execution.cache_hits, 0);
    assert_eq!(first.specialization.shapes.len(), 3);
    assert!(first.buffers.contains_key(&NodeId(2)));

    let second = runtime
        .execute_storage_graph(DeviceId(0), &graph, &fusion, &inputs, &options)
        .unwrap();
    assert_eq!(second.execution.cache_misses, 0);
    assert_eq!(second.execution.cache_hits, 1);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.driver().launches.len(), 2);
    assert_eq!(runtime.driver().launches[0].arguments.offsets.len(), 3);
}

#[test]
fn derives_cross_region_dependencies_from_graph_values() {
    let shape = || Shape::typed([Dimension::Known(64)], ElementKind::IeeeFloat, 32);
    let mut producer = FusionNode::structural(1, NodeKind::Elementwise, [NodeId(0)], shape());
    producer.operation = "exp".into();
    producer.has_side_effects = true;
    let mut consumer = FusionNode::structural(2, NodeKind::Elementwise, [NodeId(1)], shape());
    consumer.operation = "log".into();
    let graph = FusionGraph::new(vec![
        FusionNode::structural(0, NodeKind::Parameter, [], shape()),
        producer,
        consumer,
    ])
    .unwrap();
    let fusion = plan(&graph, DeviceModel::conservative_gpu());
    assert_eq!(fusion.regions.len(), 2);
    assert_eq!(
        schedule_fusion_regions(&graph, &fusion).unwrap(),
        [
            RegionSchedule {
                region: RegionId(0),
                dependencies: Vec::new(),
            },
            RegionSchedule {
                region: RegionId(1),
                dependencies: vec![RegionId(0)],
            },
        ]
    );
}

#[test]
fn cache_key_covers_representation_specialization_target_donor_and_options() {
    let f32_graph = graph(ElementKind::IeeeFloat, 32);
    let bf16_graph = graph(ElementKind::BrainFloat, 16);
    let f32_plan = plan(&f32_graph, DeviceModel::conservative_gpu());
    let bf16_plan = plan(&bf16_graph, DeviceModel::conservative_gpu());
    let specialization = KernelSpecialization {
        shapes: Vec::new(),
        strides: Vec::new(),
        target: GpuTarget::Nvidia,
    };
    let options = options();
    let base = CacheKey::for_kernel(
        &f32_graph,
        &f32_plan.regions[0],
        &specialization,
        &options,
        "revision-a",
    );
    assert_ne!(
        base,
        CacheKey::for_kernel(
            &bf16_graph,
            &bf16_plan.regions[0],
            &specialization,
            &options,
            "revision-a"
        )
    );
    let mut changed_options = options.clone();
    changed_options.num_stages = 7;
    assert_ne!(
        base,
        CacheKey::for_kernel(
            &f32_graph,
            &f32_plan.regions[0],
            &specialization,
            &changed_options,
            "revision-a"
        )
    );
    assert_ne!(
        base,
        CacheKey::for_kernel(
            &f32_graph,
            &f32_plan.regions[0],
            &specialization,
            &options,
            "revision-b"
        )
    );

    let shape = Shape::unranked(ElementKind::IeeeFloat, 32);
    let mut dynamic_op =
        FusionNode::structural(1, NodeKind::Elementwise, [NodeId(0)], shape.clone());
    dynamic_op.operation = "silu".into();
    let dynamic_graph = FusionGraph::new(vec![
        FusionNode::structural(0, NodeKind::Parameter, [], shape),
        dynamic_op,
    ])
    .unwrap();
    let dynamic_plan = plan(&dynamic_graph, DeviceModel::conservative_gpu());
    let specialized = |extent, stride| KernelSpecialization {
        shapes: vec![
            RuntimeShape {
                node: NodeId(0),
                dimensions: vec![extent],
            },
            RuntimeShape {
                node: NodeId(1),
                dimensions: vec![extent],
            },
        ],
        strides: vec![
            severian_fusion::RuntimeStrides {
                node: NodeId(0),
                strides: vec![stride],
                offset: 0,
            },
            severian_fusion::RuntimeStrides {
                node: NodeId(1),
                strides: vec![stride],
                offset: 0,
            },
        ],
        target: GpuTarget::Nvidia,
    };
    let dynamic = CacheKey::for_kernel(
        &dynamic_graph,
        &dynamic_plan.regions[0],
        &specialized(64, 1),
        &options,
        "revision-a",
    );
    assert_ne!(
        dynamic,
        CacheKey::for_kernel(
            &dynamic_graph,
            &dynamic_plan.regions[0],
            &specialized(128, 1),
            &options,
            "revision-a"
        )
    );
    assert_ne!(
        dynamic,
        CacheKey::for_kernel(
            &dynamic_graph,
            &dynamic_plan.regions[0],
            &specialized(64, 2),
            &options,
            "revision-a"
        )
    );
}

#[test]
fn unranked_grid_uses_runtime_specialization_without_ranked_symbols() {
    let shape = Shape::unranked(ElementKind::IeeeFloat, 32);
    let mut operation =
        FusionNode::structural(1, NodeKind::Elementwise, [NodeId(0)], shape.clone());
    operation.operation = "silu".into();
    let graph = FusionGraph::new(vec![
        FusionNode::structural(0, NodeKind::Parameter, [], shape),
        operation,
    ])
    .unwrap();
    let specialization = KernelSpecialization {
        shapes: vec![
            RuntimeShape {
                node: NodeId(0),
                dimensions: vec![3, 5],
            },
            RuntimeShape {
                node: NodeId(1),
                dimensions: vec![3, 5],
            },
        ],
        strides: Vec::new(),
        target: GpuTarget::Nvidia,
    };
    assert_eq!(
        calculate_grid(
            &graph,
            &specialization,
            &GridPolicy::Linear {
                output: NodeId(1),
                elements_per_program: 8,
            }
        )
        .unwrap(),
        [2, 1, 1]
    );
}

#[test]
fn persistent_cache_round_trips_launch_metadata_and_code() {
    let directory =
        std::env::temp_dir().join(format!("severian-gpu-cache-test-{}", std::process::id()));
    let graph = graph(ElementKind::IeeeFloat, 32);
    let fusion = plan(&graph, DeviceModel::conservative_gpu());
    let specialization = KernelSpecialization {
        shapes: Vec::new(),
        strides: Vec::new(),
        target: GpuTarget::Nvidia,
    };
    let key = CacheKey::for_kernel(
        &graph,
        &fusion.regions[0],
        &specialization,
        &options(),
        "revision-a",
    );
    let artifact = KernelArtifact {
        format: KernelBinaryFormat::Ptx,
        entry_point: "severian_region_0".into(),
        code: b"ptx".to_vec(),
        launch: LaunchRequirements {
            grid: GridPolicy::Fixed([7, 2, 1]),
            block: [128, 1, 1],
            num_warps: 4,
            warp_size: 32,
            num_ctas: 1,
            shared_memory_bytes: 4096,
        },
    };
    let mut cache = KernelCache::persistent(&directory);
    cache.insert(key, artifact.clone()).unwrap();
    drop(cache);
    let mut cache = KernelCache::persistent(&directory);
    assert_eq!(cache.get(&key).unwrap(), Some(artifact));
    std::fs::remove_dir_all(directory).unwrap();
}
