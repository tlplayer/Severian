use severian_fusion::{
    Dimension, ElementKind, FusionGraph, FusionNode, FusionRegion, KernelSpecialization, NodeId,
    NodeKind, Rank, StorageLayout, Stride,
};
use std::collections::BTreeMap;
use std::fmt::Write;

const BLOCK: u64 = 256;
const DOT_TILE: u64 = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtirModule {
    pub entry_point: String,
    pub text: String,
}

pub fn lower(
    graph: &FusionGraph,
    region: &FusionRegion,
    specialization: &KernelSpecialization,
) -> Result<TtirModule, String> {
    let shapes = ConcreteShapes::new(graph, specialization)?;
    let hero = region
        .nodes
        .iter()
        .map(|id| graph.node(*id).kind)
        .find(|kind| {
            matches!(
                kind,
                NodeKind::Contraction
                    | NodeKind::Reduction
                    | NodeKind::Gather
                    | NodeKind::Concatenate
                    | NodeKind::Scatter
            )
        });
    match hero {
        Some(NodeKind::Contraction) => emit_matmul(graph, region, &shapes),
        Some(NodeKind::Reduction) => emit_reduction(graph, region, &shapes),
        Some(NodeKind::Gather) => emit_gather(graph, region, &shapes),
        Some(NodeKind::Concatenate) => emit_concatenate(graph, region, &shapes),
        Some(NodeKind::Scatter) => emit_scatter(graph, region, &shapes),
        _ => emit_linear(graph, region, &shapes),
    }
}

struct ConcreteShapes {
    dimensions: BTreeMap<NodeId, Vec<u64>>,
    strides: BTreeMap<NodeId, (Vec<i64>, i64)>,
}

impl ConcreteShapes {
    fn new(graph: &FusionGraph, specialization: &KernelSpecialization) -> Result<Self, String> {
        let runtime = specialization
            .shapes
            .iter()
            .map(|shape| (shape.node, shape.dimensions.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut shapes = BTreeMap::new();
        let runtime_strides = specialization
            .strides
            .iter()
            .map(|layout| (layout.node, (layout.strides.clone(), layout.offset)))
            .collect::<BTreeMap<_, _>>();
        let mut strides = BTreeMap::new();
        for node in graph.nodes() {
            let dimensions = match &node.shape.rank {
                Rank::Unranked => runtime
                    .get(&node.id)
                    .cloned()
                    .ok_or_else(|| format!("node {} has no specialized rank", node.id.0))?,
                Rank::Ranked(dimensions) => dimensions
                    .iter()
                    .enumerate()
                    .map(|(axis, dimension)| match dimension {
                        Dimension::Known(value) => Ok(*value),
                        Dimension::Dynamic => runtime
                            .get(&node.id)
                            .and_then(|shape| shape.get(axis))
                            .copied()
                            .ok_or_else(|| {
                                format!("node {} axis {axis} has no specialized extent", node.id.0)
                            }),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            };
            shapes.insert(node.id, dimensions);
            if let StorageLayout::Strided {
                strides: layout_strides,
                offset,
            } = &node.layout
            {
                let concrete = runtime_strides.get(&node.id).cloned().or_else(|| {
                    let strides = layout_strides
                        .iter()
                        .map(|stride| match stride {
                            Stride::Known(value) => Some(*value),
                            Stride::Dynamic => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    let offset = match offset {
                        Stride::Known(value) => *value,
                        Stride::Dynamic => return None,
                    };
                    Some((strides, offset))
                });
                if let Some(concrete) = concrete {
                    strides.insert(node.id, concrete);
                }
            }
        }
        Ok(Self {
            dimensions: shapes,
            strides,
        })
    }

    fn get(&self, id: NodeId) -> &[u64] {
        &self.dimensions[&id]
    }

    fn strides(&self, id: NodeId) -> Option<(&[i64], i64)> {
        self.strides
            .get(&id)
            .map(|(strides, offset)| (strides.as_slice(), *offset))
    }

    fn elements(&self, id: NodeId) -> Result<u64, String> {
        self.get(id).iter().try_fold(1u64, |count, extent| {
            count
                .checked_mul(*extent)
                .ok_or_else(|| format!("node {} element count overflows u64", id.0))
        })
    }
}

struct Emitter {
    text: String,
    next: u32,
}

impl Emitter {
    fn new() -> Self {
        Self {
            text: String::new(),
            next: 0,
        }
    }

    fn value(&mut self) -> String {
        let value = format!("%v{}", self.next);
        self.next += 1;
        value
    }

    fn line(&mut self, line: impl AsRef<str>) {
        let _ = writeln!(self.text, "    {}", line.as_ref());
    }

    fn comment(&mut self, text: impl AsRef<str>) {
        self.line(format!("// severian.{}", text.as_ref()));
    }
}

#[derive(Clone)]
struct Value {
    name: String,
    scalar: String,
}

fn emit_linear(
    graph: &FusionGraph,
    region: &FusionRegion,
    shapes: &ConcreteShapes,
) -> Result<TtirModule, String> {
    let output = *region
        .outputs
        .first()
        .ok_or_else(|| "fusion region has no output".to_owned())?;
    let mut emitter = Emitter::new();
    let (arguments, input_arguments, output_arguments) = arguments(graph, region)?;
    begin_module(&mut emitter, region, &arguments);
    let output_elements = shapes.elements(output)?;
    let (offsets, mask) = linear_offsets(&mut emitter, output_elements, BLOCK);
    let root = region.nodes.contains(&output).then(|| graph.node(output));
    let mut values = BTreeMap::new();
    for (id, argument) in input_arguments {
        let node = graph.node(id);
        if node_operand_is_metadata(graph, region, id) {
            continue;
        }
        let input_elements = shapes.elements(id)?;
        let input_offsets = if root
            .is_some_and(|root| root.kind == NodeKind::Slice && root.inputs.first() == Some(&id))
            && shapes.strides(output).is_some()
        {
            let (strides, offset) = shapes.strides(output).expect("checked above");
            emitter.comment(format!("Slice.input_{}_strided_index", id.0));
            strided_offsets(
                &mut emitter,
                &offsets,
                shapes.get(output),
                strides,
                offset,
                BLOCK,
            )?
        } else if root.is_some_and(|root| {
            root.kind == NodeKind::Permute
                && root.operation == "reverse"
                && root.inputs.first() == Some(&id)
        }) {
            emitter.comment(format!("Permute.input_{}_row_major_index", id.0));
            let permutation = (0..shapes.get(output).len()).rev().collect::<Vec<_>>();
            permuted_offsets(
                &mut emitter,
                &offsets,
                shapes.get(id),
                shapes.get(output),
                &permutation,
                BLOCK,
            )?
        } else if input_elements != output_elements {
            emitter.comment(format!("Broadcast.input_{}_row_major_index", id.0));
            broadcast_offsets(
                &mut emitter,
                &offsets,
                shapes.get(id),
                shapes.get(output),
                BLOCK,
            )?
        } else {
            offsets.clone()
        };
        values.insert(
            id,
            load(&mut emitter, argument, node, &input_offsets, &mask, BLOCK),
        );
    }
    for id in &region.nodes {
        let node = graph.node(*id);
        let value = emit_linear_node(&mut emitter, graph, node, &values, BLOCK)?;
        values.insert(*id, value);
    }
    for (id, argument) in output_arguments {
        let value = values
            .get(&id)
            .ok_or_else(|| format!("region output {} has no lowered value", id.0))?;
        store(&mut emitter, argument, value, &offsets, &mask, BLOCK);
    }
    end_module(&mut emitter);
    finish(region, emitter)
}

fn emit_linear_node(
    emitter: &mut Emitter,
    graph: &FusionGraph,
    node: &FusionNode,
    values: &BTreeMap<NodeId, Value>,
    width: u64,
) -> Result<Value, String> {
    let operands = node
        .inputs
        .iter()
        .filter_map(|id| values.get(id))
        .collect::<Vec<_>>();
    let ty = scalar_type(node)?;
    let tensor = tensor_type(width, &ty);
    emitter.comment(format!("{:?}.{}", node.kind, node.operation));
    match node.kind {
        NodeKind::Elementwise => emit_elementwise(emitter, node, &operands, &tensor, width),
        NodeKind::Convert => emit_convert(emitter, graph, node, &operands, width),
        NodeKind::Reshape
        | NodeKind::StorageView
        | NodeKind::Broadcast
        | NodeKind::Permute
        | NodeKind::Slice
        | NodeKind::Gather => passthrough(node, &operands),
        NodeKind::Concatenate => {
            let [left, right, ..] = operands.as_slice() else {
                return passthrough(node, &operands);
            };
            let condition = emitter.value();
            let zero = emitter.value();
            emitter.line(format!(
                "{zero} = arith.constant dense<false> : tensor<{width}xi1>"
            ));
            emitter.line(format!(
                "{condition} = arith.select {zero}, {}, {} : tensor<{width}xi1>, {tensor}",
                right.name, left.name
            ));
            Ok(Value {
                name: condition,
                scalar: ty,
            })
        }
        other => Err(format!("{other:?} requires its structural TTIR emitter")),
    }
}

fn emit_elementwise(
    emitter: &mut Emitter,
    node: &FusionNode,
    operands: &[&Value],
    tensor: &str,
    width: u64,
) -> Result<Value, String> {
    let first = operands
        .first()
        .ok_or_else(|| format!("{} has no data operand", node.operation))?;
    let result = emitter.value();
    let float = matches!(
        node.shape.element_kind,
        ElementKind::IeeeFloat
            | ElementKind::BrainFloat
            | ElementKind::Float8E4M3Fn
            | ElementKind::Float8E5M2
    );
    let binary = match node.operation.as_str() {
        "add" | "add_scalar" => Some(if float { "arith.addf" } else { "arith.addi" }),
        "subtract" => Some(if float { "arith.subf" } else { "arith.subi" }),
        "multiply" | "scale" => Some(if float { "arith.mulf" } else { "arith.muli" }),
        "divide" => Some(if float { "arith.divf" } else { "arith.divsi" }),
        _ => None,
    };
    if let Some(operation) = binary {
        let rhs = operands.get(1).copied().unwrap_or(first);
        let rhs = coerce_value(emitter, rhs, &first.scalar, width)?;
        emitter.line(format!(
            "{result} = {operation} {}, {} : {tensor}",
            first.name, rhs.name
        ));
    } else if matches!(
        node.operation.as_str(),
        "exp" | "log" | "sin" | "tanh" | "rsqrt"
    ) {
        emitter.line(format!(
            "{result} = math.{} {} : {tensor}",
            node.operation, first.name
        ));
    } else if node.operation == "relu" {
        let zero = zero_tensor(emitter, &first.scalar, tensor);
        let compare = emitter.value();
        let compare_operation = if float {
            "arith.cmpf ogt"
        } else {
            "arith.cmpi sgt"
        };
        emitter.line(format!(
            "{compare} = {compare_operation}, {}, {zero} : {tensor}",
            first.name
        ));
        emitter.line(format!(
            "{result} = arith.select {compare}, {}, {zero} : tensor<{width}xi1>, {tensor}",
            first.name
        ));
    } else {
        return Err(format!(
            "unsupported elementwise operation `{}`",
            node.operation
        ));
    }
    Ok(Value {
        name: result,
        scalar: first.scalar.clone(),
    })
}

fn coerce_value(
    emitter: &mut Emitter,
    value: &Value,
    target_scalar: &str,
    width: u64,
) -> Result<Value, String> {
    if value.scalar == target_scalar {
        return Ok(value.clone());
    }
    let source_float = value.scalar.strip_prefix('f').and_then(|bits| bits.parse::<u16>().ok());
    let target_float = target_scalar
        .strip_prefix('f')
        .and_then(|bits| bits.parse::<u16>().ok());
    let operation = match (source_float, target_float) {
        (Some(source), Some(target)) if source > target => "arith.truncf",
        (Some(source), Some(target)) if source < target => "arith.extf",
        _ => {
            return Err(format!(
                "cannot coerce elementwise operand from {} to {target_scalar}",
                value.scalar
            ))
        }
    };
    let result = emitter.value();
    emitter.line(format!(
        "{result} = {operation} {} : tensor<{width}x{}> to tensor<{width}x{target_scalar}>",
        value.name, value.scalar
    ));
    Ok(Value {
        name: result,
        scalar: target_scalar.into(),
    })
}

fn emit_convert(
    emitter: &mut Emitter,
    graph: &FusionGraph,
    node: &FusionNode,
    operands: &[&Value],
    width: u64,
) -> Result<Value, String> {
    let input = operands
        .first()
        .ok_or_else(|| "convert has no operand".to_owned())?;
    let source = graph.node(node.inputs[0]);
    let target_scalar = scalar_type(node)?;
    let target = tensor_type(width, &target_scalar);
    if source.shape.element_kind == node.shape.element_kind
        && source.shape.element_bits == node.shape.element_bits
    {
        return Ok((*input).clone());
    }
    if matches!(
        source.shape.element_kind,
        ElementKind::SignedInteger | ElementKind::UnsignedInteger
    ) && matches!(
        node.shape.element_kind,
        ElementKind::SignedInteger | ElementKind::UnsignedInteger
    ) && source.shape.element_bits == node.shape.element_bits
    {
        return Ok(Value {
            name: input.name.clone(),
            scalar: target_scalar,
        });
    }
    let operation = match (source.shape.element_kind, node.shape.element_kind) {
        (
            ElementKind::IeeeFloat | ElementKind::BrainFloat,
            ElementKind::IeeeFloat | ElementKind::BrainFloat,
        ) => {
            if source.shape.element_bits < node.shape.element_bits {
                "arith.extf"
            } else {
                "arith.truncf"
            }
        }
        (ElementKind::SignedInteger, ElementKind::SignedInteger)
            if source.shape.element_bits < node.shape.element_bits =>
        {
            "arith.extsi"
        }
        (ElementKind::UnsignedInteger, ElementKind::UnsignedInteger)
            if source.shape.element_bits < node.shape.element_bits =>
        {
            "arith.extui"
        }
        (
            ElementKind::SignedInteger | ElementKind::UnsignedInteger,
            ElementKind::SignedInteger | ElementKind::UnsignedInteger,
        ) if source.shape.element_bits > node.shape.element_bits => "arith.trunci",
        (ElementKind::IeeeFloat | ElementKind::BrainFloat, ElementKind::SignedInteger) => {
            "arith.fptosi"
        }
        (ElementKind::IeeeFloat | ElementKind::BrainFloat, ElementKind::UnsignedInteger) => {
            "arith.fptoui"
        }
        (ElementKind::SignedInteger, ElementKind::IeeeFloat | ElementKind::BrainFloat) => {
            "arith.sitofp"
        }
        (ElementKind::UnsignedInteger, ElementKind::IeeeFloat | ElementKind::BrainFloat) => {
            "arith.uitofp"
        }
        _ => "arith.bitcast",
    };
    let result = emitter.value();
    emitter.line(format!(
        "{result} = {operation} {} : {} to {target}",
        input.name,
        tensor_type(width, &input.scalar)
    ));
    Ok(Value {
        name: result,
        scalar: target_scalar,
    })
}

fn emit_reduction(
    graph: &FusionGraph,
    region: &FusionRegion,
    shapes: &ConcreteShapes,
) -> Result<TtirModule, String> {
    let reduction_id = *region
        .nodes
        .iter()
        .find(|id| graph.node(**id).kind == NodeKind::Reduction)
        .ok_or_else(|| "reduction region has no reduction".to_owned())?;
    let reduction = graph.node(reduction_id);
    let source_id = reduction.inputs[0];
    let mut emitter = Emitter::new();
    let (arguments, input_arguments, output_arguments) = arguments(graph, region)?;
    begin_module(&mut emitter, region, &arguments);
    let extent = shapes.elements(source_id)?;
    let logical_width = if matches!(reduction.operation.as_str(), "mean_last" | "max_last") {
        shapes.get(source_id).last().copied().unwrap_or(1)
    } else {
        extent
    };
    let width = logical_width
        .checked_next_power_of_two()
        .ok_or_else(|| "reduction tile width overflows u64".to_owned())?;
    if width > i32::MAX as u64 {
        return Err(format!(
            "reduction tile width {width} exceeds TTIR i32 indexing"
        ));
    }
    let (offsets, mask) = linear_offsets(&mut emitter, extent, width);
    let mut values = BTreeMap::new();
    for (id, argument) in &input_arguments {
        if node_operand_is_metadata(graph, region, *id) {
            continue;
        }
        let input_offsets = if shapes.elements(*id)? != extent {
            broadcast_offsets(
                &mut emitter,
                &offsets,
                shapes.get(*id),
                shapes.get(source_id),
                width,
            )?
        } else {
            offsets.clone()
        };
        values.insert(
            *id,
            load(
                &mut emitter,
                *argument,
                graph.node(*id),
                &input_offsets,
                &mask,
                width,
            ),
        );
    }
    let mut reduced_scalar = None;
    for id in &region.nodes {
        let node = graph.node(*id);
        if *id == reduction_id {
            let source = values
                .get(&source_id)
                .cloned()
                .ok_or_else(|| format!("reduction source {} was not lowered", source_id.0))?;
            emitter.comment(format!("Reduction.{}", reduction.operation));
            let reduced_value = emitter.value();
            let combine = if reduction.operation == "max_last" {
                if matches!(
                    reduction.shape.element_kind,
                    ElementKind::IeeeFloat | ElementKind::BrainFloat
                ) {
                    "arith.maxnumf"
                } else {
                    "arith.maxsi"
                }
            } else if matches!(
                reduction.shape.element_kind,
                ElementKind::IeeeFloat | ElementKind::BrainFloat
            ) {
                "arith.addf"
            } else {
                "arith.addi"
            };
            emitter.line(format!(
                "{reduced_value} = \"tt.reduce\" ({}) ({{",
                source.name
            ));
            emitter.line(format!(
                "^bb0(%left: {}, %right: {}):",
                source.scalar, source.scalar
            ));
            emitter.line(format!(
                "  %combined = {combine} %left, %right : {}",
                source.scalar
            ));
            emitter.line(format!("  tt.reduce.return %combined : {}", source.scalar));
            emitter.line(format!(
                "}}) {{axis = 0 : i32}} : ({}) -> {}",
                tensor_type(width, &source.scalar),
                source.scalar
            ));
            let reduced = if reduction.operation == "mean_last" {
                let divisor = emitter.value();
                let literal = if source.scalar.starts_with('f') || source.scalar == "bf16" {
                    format!("{}.0", logical_width)
                } else {
                    logical_width.to_string()
                };
                emitter.line(format!(
                    "{divisor} = arith.constant {literal} : {}",
                    source.scalar
                ));
                let mean = emitter.value();
                let divide = if source.scalar.starts_with('f') || source.scalar == "bf16" {
                    "arith.divf"
                } else {
                    "arith.divsi"
                };
                emitter.line(format!(
                    "{mean} = {divide} {reduced_value}, {divisor} : {}",
                    source.scalar
                ));
                mean
            } else {
                reduced_value
            };
            let broadcast = emitter.value();
            emitter.line(format!(
                "{broadcast} = tt.splat {reduced} : {} -> {}",
                source.scalar,
                tensor_type(width, &source.scalar)
            ));
            values.insert(
                *id,
                Value {
                    name: broadcast,
                    scalar: source.scalar.clone(),
                },
            );
            reduced_scalar = Some((reduced, source.scalar.clone()));
        } else {
            let value = emit_linear_node(&mut emitter, graph, node, &values, width)?;
            values.insert(*id, value);
        }
    }
    for (id, argument) in output_arguments {
        if id == reduction_id {
            let (reduced, scalar) = reduced_scalar
                .as_ref()
                .ok_or_else(|| "reduction produced no scalar".to_owned())?;
            let program = emitter.value();
            emitter.line(format!("{program} = tt.get_program_id x : i32"));
            let address = emitter.value();
            emitter.line(format!(
                "{address} = tt.addptr %arg{argument}, {program} : !tt.ptr<{scalar}>, i32"
            ));
            emitter.line(format!("tt.store {address}, {reduced} : !tt.ptr<{scalar}>"));
        } else {
            let value = values
                .get(&id)
                .ok_or_else(|| format!("region output {} has no lowered value", id.0))?;
            store(&mut emitter, argument, value, &offsets, &mask, width);
        }
    }
    end_module(&mut emitter);
    finish(region, emitter)
}

fn emit_matmul(
    graph: &FusionGraph,
    region: &FusionRegion,
    shapes: &ConcreteShapes,
) -> Result<TtirModule, String> {
    let id = *region
        .nodes
        .iter()
        .find(|id| graph.node(**id).kind == NodeKind::Contraction)
        .ok_or_else(|| "contraction region has no matmul".to_owned())?;
    let node = graph.node(id);
    let contract = node
        .matmul
        .as_ref()
        .ok_or_else(|| "matmul has no contraction metadata".to_owned())?;
    let [lhs_id, rhs_id, ..] = node.inputs.as_slice() else {
        return Err("matmul requires lhs and rhs".into());
    };
    let lhs = graph.node(*lhs_id);
    let rhs = graph.node(*rhs_id);
    let lhs_shape = shapes.get(*lhs_id);
    let rhs_shape = shapes.get(*rhs_id);
    if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
        return Err("matmul specialization requires rank at least two".into());
    }
    let mut emitter = Emitter::new();
    let (arguments, input_arguments, output_arguments) = arguments(graph, region)?;
    begin_module(&mut emitter, region, &arguments);
    emitter.comment(format!(
        "Matmul batch={:?} contraction={:?}",
        contract.batch_dimensions, contract.contraction_dimensions
    ));
    let lhs_arg = input_arguments
        .iter()
        .find(|(input, _)| input == lhs_id)
        .map(|(_, arg)| *arg)
        .ok_or_else(|| "matmul lhs must be a region input".to_owned())?;
    let rhs_arg = input_arguments
        .iter()
        .find(|(input, _)| input == rhs_id)
        .map(|(_, arg)| *arg)
        .ok_or_else(|| "matmul rhs must be a region input".to_owned())?;
    let output_arg = output_arguments
        .first()
        .map(|(_, arg)| *arg)
        .ok_or_else(|| "matmul has no output".to_owned())?;
    let lhs_ty = scalar_type(lhs)?;
    let rhs_ty = scalar_type(rhs)?;
    let result_ty = scalar_type(node)?;
    let (offsets, mask) = linear_offsets(&mut emitter, DOT_TILE * DOT_TILE, DOT_TILE * DOT_TILE);
    let lhs_value = load(
        &mut emitter,
        lhs_arg,
        lhs,
        &offsets,
        &mask,
        DOT_TILE * DOT_TILE,
    );
    let rhs_value = load(
        &mut emitter,
        rhs_arg,
        rhs,
        &offsets,
        &mask,
        DOT_TILE * DOT_TILE,
    );
    let lhs_matrix = emitter.value();
    let rhs_matrix = emitter.value();
    emitter.line(format!(
        "{lhs_matrix} = tt.reshape {} : {} -> tensor<{DOT_TILE}x{DOT_TILE}x{lhs_ty}>",
        lhs_value.name,
        tensor_type(DOT_TILE * DOT_TILE, &lhs_ty)
    ));
    emitter.line(format!(
        "{rhs_matrix} = tt.reshape {} : {} -> tensor<{DOT_TILE}x{DOT_TILE}x{rhs_ty}>",
        rhs_value.name,
        tensor_type(DOT_TILE * DOT_TILE, &rhs_ty)
    ));
    let zero = emitter.value();
    emitter.line(format!(
        "{zero} = arith.constant dense<0.0> : tensor<{DOT_TILE}x{DOT_TILE}x{result_ty}>"
    ));
    let dot = emitter.value();
    emitter.line(format!("{dot} = tt.dot {lhs_matrix}, {rhs_matrix}, {zero} : tensor<{DOT_TILE}x{DOT_TILE}x{lhs_ty}> * tensor<{DOT_TILE}x{DOT_TILE}x{rhs_ty}> -> tensor<{DOT_TILE}x{DOT_TILE}x{result_ty}>"));
    let flat = emitter.value();
    emitter.line(format!(
        "{flat} = tt.reshape {dot} : tensor<{DOT_TILE}x{DOT_TILE}x{result_ty}> -> {}",
        tensor_type(DOT_TILE * DOT_TILE, &result_ty)
    ));
    store(
        &mut emitter,
        output_arg,
        &Value {
            name: flat,
            scalar: result_ty,
        },
        &offsets,
        &mask,
        DOT_TILE * DOT_TILE,
    );
    end_module(&mut emitter);
    finish(region, emitter)
}

fn emit_scatter(
    graph: &FusionGraph,
    region: &FusionRegion,
    shapes: &ConcreteShapes,
) -> Result<TtirModule, String> {
    let scatter = *region
        .nodes
        .iter()
        .find(|id| graph.node(**id).kind == NodeKind::Scatter)
        .ok_or_else(|| "scatter region has no scatter".to_owned())?;
    let node = graph.node(scatter);
    let mut emitter = Emitter::new();
    let (arguments, input_arguments, _) = arguments(graph, region)?;
    begin_module(&mut emitter, region, &arguments);
    let source = node
        .inputs
        .last()
        .copied()
        .ok_or_else(|| "scatter has no update operand".to_owned())?;
    let (offsets, mask) = linear_offsets(&mut emitter, shapes.elements(source)?, BLOCK);
    let (source_arg, source_node) = input_arguments
        .iter()
        .find(|(id, _)| *id == source)
        .map(|(_, arg)| (*arg, graph.node(source)))
        .ok_or_else(|| "scatter updates must be a region input".to_owned())?;
    let value = load(
        &mut emitter,
        source_arg,
        source_node,
        &offsets,
        &mask,
        BLOCK,
    );
    let destination = node.inputs[0];
    let destination_arg = input_arguments
        .iter()
        .find(|(id, _)| *id == destination)
        .map(|(_, arg)| *arg)
        .ok_or_else(|| "scatter destination must be a region input".to_owned())?;
    let (store_offsets, store_mask) = if let Some(indices) = node.inputs.get(1).copied() {
        if indices == source {
            (offsets.clone(), mask.clone())
        } else {
            let (indices_arg, indices_node) = input_arguments
                .iter()
                .find(|(id, _)| *id == indices)
                .map(|(_, arg)| (*arg, graph.node(indices)))
                .ok_or_else(|| "scatter indices must be a region input".to_owned())?;
            let indices = load(
                &mut emitter,
                indices_arg,
                indices_node,
                &offsets,
                &mask,
                BLOCK,
            );
            checked_indices(
                &mut emitter,
                &indices,
                shapes.elements(destination)?,
                &mask,
                BLOCK,
            )?
        }
    } else {
        (offsets.clone(), mask.clone())
    };
    emitter.comment("Scatter.indexed_masked_store");
    store(
        &mut emitter,
        destination_arg,
        &value,
        &store_offsets,
        &store_mask,
        BLOCK,
    );
    end_module(&mut emitter);
    finish(region, emitter)
}

fn emit_gather(
    graph: &FusionGraph,
    region: &FusionRegion,
    shapes: &ConcreteShapes,
) -> Result<TtirModule, String> {
    let gather = *region
        .nodes
        .iter()
        .find(|id| graph.node(**id).kind == NodeKind::Gather)
        .ok_or_else(|| "gather region has no gather".to_owned())?;
    let node = graph.node(gather);
    let [source, indices, ..] = node.inputs.as_slice() else {
        return Err("gather requires source and indices".to_owned());
    };
    let mut emitter = Emitter::new();
    let (arguments, input_arguments, output_arguments) = arguments(graph, region)?;
    begin_module(&mut emitter, region, &arguments);
    let (offsets, mask) = linear_offsets(&mut emitter, shapes.elements(gather)?, BLOCK);
    let source_arg = input_arguments
        .iter()
        .find(|(id, _)| id == source)
        .map(|(_, argument)| *argument)
        .ok_or_else(|| "gather source must be a region input".to_owned())?;
    let indices_arg = input_arguments
        .iter()
        .find(|(id, _)| id == indices)
        .map(|(_, argument)| *argument)
        .ok_or_else(|| "gather indices must be a region input".to_owned())?;
    let indices_value = load(
        &mut emitter,
        indices_arg,
        graph.node(*indices),
        &offsets,
        &mask,
        BLOCK,
    );
    let (source_offsets, source_mask) = checked_indices(
        &mut emitter,
        &indices_value,
        shapes.elements(*source)?,
        &mask,
        BLOCK,
    )?;
    emitter.comment("Gather.indexed_masked_load");
    let gathered = load(
        &mut emitter,
        source_arg,
        graph.node(*source),
        &source_offsets,
        &source_mask,
        BLOCK,
    );
    let output_arg = output_arguments
        .iter()
        .find(|(id, _)| *id == gather)
        .or_else(|| output_arguments.first())
        .map(|(_, argument)| *argument)
        .ok_or_else(|| "gather has no output".to_owned())?;
    store(&mut emitter, output_arg, &gathered, &offsets, &mask, BLOCK);
    end_module(&mut emitter);
    finish(region, emitter)
}

fn emit_concatenate(
    graph: &FusionGraph,
    region: &FusionRegion,
    shapes: &ConcreteShapes,
) -> Result<TtirModule, String> {
    let concatenate = *region
        .nodes
        .iter()
        .find(|id| graph.node(**id).kind == NodeKind::Concatenate)
        .ok_or_else(|| "concatenate region has no concatenate".to_owned())?;
    let node = graph.node(concatenate);
    let [left, right, ..] = node.inputs.as_slice() else {
        return Err("concatenate requires left and right operands".to_owned());
    };
    let (left_shape, right_shape, output_shape) = (
        shapes.get(*left),
        shapes.get(*right),
        shapes.get(concatenate),
    );
    if left_shape.len() != output_shape.len() || right_shape.len() != output_shape.len() {
        return Err("concatenate operands must have equal rank".to_owned());
    }
    let candidates = (0..output_shape.len())
        .filter(|&axis| {
            left_shape[axis]
                .checked_add(right_shape[axis])
                .is_some_and(|extent| extent == output_shape[axis])
                && (0..output_shape.len()).all(|other| {
                    other == axis
                        || (left_shape[other] == output_shape[other]
                            && right_shape[other] == output_shape[other])
                })
        })
        .collect::<Vec<_>>();
    let axis = node
        .attributes
        .first()
        .and_then(|axis| usize::try_from(*axis).ok())
        .filter(|axis| candidates.contains(axis))
        .or_else(|| (candidates.len() == 1).then_some(candidates[0]))
        .ok_or_else(|| "concatenate axis cannot be inferred from specialized shapes".to_owned())?;
    let mut emitter = Emitter::new();
    let (arguments, input_arguments, output_arguments) = arguments(graph, region)?;
    begin_module(&mut emitter, region, &arguments);
    let (offsets, mask) = linear_offsets(&mut emitter, shapes.elements(concatenate)?, BLOCK);
    let mut left_offsets = vector_constant(&mut emitter, 0, BLOCK)?;
    let mut right_offsets = vector_constant(&mut emitter, 0, BLOCK)?;
    let mut axis_coordinate = None;
    for output_axis in 0..output_shape.len() {
        let output_stride = product(&output_shape[output_axis + 1..])?;
        let divided =
            vector_binary_constant(&mut emitter, "arith.divui", &offsets, output_stride, BLOCK)?;
        let coordinate = vector_binary_constant(
            &mut emitter,
            "arith.remui",
            &divided,
            output_shape[output_axis],
            BLOCK,
        )?;
        let right_coordinate = if output_axis == axis {
            axis_coordinate = Some(coordinate.clone());
            vector_binary_constant(
                &mut emitter,
                "arith.subi",
                &coordinate,
                left_shape[axis],
                BLOCK,
            )?
        } else {
            coordinate.clone()
        };
        let left_term = vector_binary_constant(
            &mut emitter,
            "arith.muli",
            &coordinate,
            product(&left_shape[output_axis + 1..])?,
            BLOCK,
        )?;
        let right_term = vector_binary_constant(
            &mut emitter,
            "arith.muli",
            &right_coordinate,
            product(&right_shape[output_axis + 1..])?,
            BLOCK,
        )?;
        let next_left = emitter.value();
        emitter.line(format!(
            "{next_left} = arith.addi {left_offsets}, {left_term} : tensor<{BLOCK}xi32>"
        ));
        left_offsets = next_left;
        let next_right = emitter.value();
        emitter.line(format!(
            "{next_right} = arith.addi {right_offsets}, {right_term} : tensor<{BLOCK}xi32>"
        ));
        right_offsets = next_right;
    }
    let axis_coordinate = axis_coordinate.expect("concatenate has a selected axis");
    let left_limit = vector_constant(&mut emitter, left_shape[axis], BLOCK)?;
    let take_left = emitter.value();
    emitter.line(format!(
        "{take_left} = arith.cmpi slt, {axis_coordinate}, {left_limit} : tensor<{BLOCK}xi32>"
    ));
    let left_mask = emitter.value();
    emitter.line(format!(
        "{left_mask} = arith.andi {mask}, {take_left} : tensor<{BLOCK}xi1>"
    ));
    let take_right = emitter.value();
    emitter.line(format!(
        "{take_right} = arith.cmpi sge, {axis_coordinate}, {left_limit} : tensor<{BLOCK}xi32>"
    ));
    let right_mask = emitter.value();
    emitter.line(format!(
        "{right_mask} = arith.andi {mask}, {take_right} : tensor<{BLOCK}xi1>"
    ));
    let argument = |id: NodeId| {
        input_arguments
            .iter()
            .find(|(input, _)| *input == id)
            .map(|(_, argument)| *argument)
            .ok_or_else(|| format!("concatenate operand {} must be a region input", id.0))
    };
    let left_value = load(
        &mut emitter,
        argument(*left)?,
        graph.node(*left),
        &left_offsets,
        &left_mask,
        BLOCK,
    );
    let right_value = load(
        &mut emitter,
        argument(*right)?,
        graph.node(*right),
        &right_offsets,
        &right_mask,
        BLOCK,
    );
    emitter.comment("Concatenate.indexed_masked_select");
    let value = emitter.value();
    let tensor = tensor_type(BLOCK, &left_value.scalar);
    emitter.line(format!(
        "{value} = arith.select {take_left}, {}, {} : tensor<{BLOCK}xi1>, {tensor}",
        left_value.name, right_value.name
    ));
    let output = output_arguments
        .first()
        .map(|(_, argument)| *argument)
        .ok_or_else(|| "concatenate has no output".to_owned())?;
    store(
        &mut emitter,
        output,
        &Value {
            name: value,
            scalar: left_value.scalar,
        },
        &offsets,
        &mask,
        BLOCK,
    );
    end_module(&mut emitter);
    finish(region, emitter)
}

type KernelArguments = (String, Vec<(NodeId, usize)>, Vec<(NodeId, usize)>);

fn arguments(graph: &FusionGraph, region: &FusionRegion) -> Result<KernelArguments, String> {
    let mut declarations = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for id in &region.inputs {
        let index = declarations.len();
        let node = graph.node(*id);
        let scalar = scalar_type(node)?;
        declarations.push(if matches!(&node.shape.rank, Rank::Ranked(axes) if axes.is_empty()) {
            format!("%arg{index}: {scalar}")
        } else {
            format!("%arg{index}: !tt.ptr<{scalar}>")
        });
        inputs.push((*id, index));
    }
    for id in &region.outputs {
        let index = declarations.len();
        declarations.push(format!(
            "%arg{index}: !tt.ptr<{}>",
            scalar_type(graph.node(*id))?
        ));
        outputs.push((*id, index));
    }
    Ok((declarations.join(", "), inputs, outputs))
}

fn begin_module(emitter: &mut Emitter, region: &FusionRegion, arguments: &str) {
    let _ = writeln!(emitter.text, "module {{");
    let _ = writeln!(
        emitter.text,
        "  tt.func public @severian_region_{}({arguments}) {{",
        region.id.0
    );
}

fn end_module(emitter: &mut Emitter) {
    emitter.line("tt.return");
    let _ = writeln!(emitter.text, "  }}\n}}");
}

fn finish(region: &FusionRegion, emitter: Emitter) -> Result<TtirModule, String> {
    Ok(TtirModule {
        entry_point: format!("severian_region_{}", region.id.0),
        text: emitter.text,
    })
}

fn linear_offsets(emitter: &mut Emitter, elements: u64, width: u64) -> (String, String) {
    let pid = emitter.value();
    emitter.line(format!("{pid} = tt.get_program_id x : i32"));
    let block = emitter.value();
    emitter.line(format!("{block} = arith.constant {width} : i32"));
    let base = emitter.value();
    emitter.line(format!("{base} = arith.muli {pid}, {block} : i32"));
    let range = emitter.value();
    emitter.line(format!(
        "{range} = tt.make_range {{end = {width} : i32, start = 0 : i32}} : tensor<{width}xi32>"
    ));
    let base_vector = emitter.value();
    emitter.line(format!(
        "{base_vector} = tt.splat {base} : i32 -> tensor<{width}xi32>"
    ));
    let offsets = emitter.value();
    emitter.line(format!(
        "{offsets} = arith.addi {base_vector}, {range} : tensor<{width}xi32>"
    ));
    let count = emitter.value();
    emitter.line(format!(
        "{count} = arith.constant {} : i32",
        elements.min(i32::MAX as u64)
    ));
    let counts = emitter.value();
    emitter.line(format!(
        "{counts} = tt.splat {count} : i32 -> tensor<{width}xi32>"
    ));
    let mask = emitter.value();
    emitter.line(format!(
        "{mask} = arith.cmpi slt, {offsets}, {counts} : tensor<{width}xi32>"
    ));
    (offsets, mask)
}

fn broadcast_offsets(
    emitter: &mut Emitter,
    offsets: &str,
    input_shape: &[u64],
    output_shape: &[u64],
    width: u64,
) -> Result<String, String> {
    if input_shape.len() > output_shape.len() {
        return Err("broadcast input rank exceeds its result rank".to_owned());
    }
    let mut mapped = vector_constant(emitter, 0, width)?;
    let leading = output_shape.len() - input_shape.len();
    for (input_axis, &input_extent) in input_shape.iter().enumerate() {
        let output_axis = leading + input_axis;
        let output_extent = output_shape[output_axis];
        if input_extent != 1 && input_extent != output_extent {
            return Err(format!(
                "broadcast axis {input_axis} has incompatible extents {input_extent} and {output_extent}"
            ));
        }
        if input_extent == 1 {
            continue;
        }
        let output_stride = product(&output_shape[output_axis + 1..])?;
        let input_stride = product(&input_shape[input_axis + 1..])?;
        let divided =
            vector_binary_constant(emitter, "arith.divui", offsets, output_stride, width)?;
        let coordinate =
            vector_binary_constant(emitter, "arith.remui", &divided, output_extent, width)?;
        let term = vector_binary_constant(emitter, "arith.muli", &coordinate, input_stride, width)?;
        let next = emitter.value();
        emitter.line(format!(
            "{next} = arith.addi {mapped}, {term} : tensor<{width}xi32>"
        ));
        mapped = next;
    }
    Ok(mapped)
}

fn permuted_offsets(
    emitter: &mut Emitter,
    offsets: &str,
    input_shape: &[u64],
    output_shape: &[u64],
    output_to_input: &[usize],
    width: u64,
) -> Result<String, String> {
    if input_shape.len() != output_shape.len() || output_to_input.len() != output_shape.len() {
        return Err("permutation rank does not match its operands".to_owned());
    }
    let mut mapped = vector_constant(emitter, 0, width)?;
    for (output_axis, &input_axis) in output_to_input.iter().enumerate() {
        if input_axis >= input_shape.len() || input_shape[input_axis] != output_shape[output_axis] {
            return Err(format!(
                "permutation axis {output_axis} does not map compatible extents"
            ));
        }
        let output_stride = product(&output_shape[output_axis + 1..])?;
        let input_stride = product(&input_shape[input_axis + 1..])?;
        let divided =
            vector_binary_constant(emitter, "arith.divui", offsets, output_stride, width)?;
        let coordinate = vector_binary_constant(
            emitter,
            "arith.remui",
            &divided,
            output_shape[output_axis],
            width,
        )?;
        let term = vector_binary_constant(emitter, "arith.muli", &coordinate, input_stride, width)?;
        let next = emitter.value();
        emitter.line(format!(
            "{next} = arith.addi {mapped}, {term} : tensor<{width}xi32>"
        ));
        mapped = next;
    }
    Ok(mapped)
}

fn strided_offsets(
    emitter: &mut Emitter,
    offsets: &str,
    output_shape: &[u64],
    physical_strides: &[i64],
    physical_offset: i64,
    width: u64,
) -> Result<String, String> {
    if output_shape.len() != physical_strides.len() {
        return Err("slice rank does not match its physical strides".to_owned());
    }
    let mut mapped = signed_vector_constant(emitter, physical_offset, width)?;
    for (axis, &stride) in physical_strides.iter().enumerate() {
        let output_stride = product(&output_shape[axis + 1..])?;
        let divided =
            vector_binary_constant(emitter, "arith.divui", offsets, output_stride, width)?;
        let coordinate =
            vector_binary_constant(emitter, "arith.remui", &divided, output_shape[axis], width)?;
        let stride = signed_vector_constant(emitter, stride, width)?;
        let term = emitter.value();
        emitter.line(format!(
            "{term} = arith.muli {coordinate}, {stride} : tensor<{width}xi32>"
        ));
        let next = emitter.value();
        emitter.line(format!(
            "{next} = arith.addi {mapped}, {term} : tensor<{width}xi32>"
        ));
        mapped = next;
    }
    Ok(mapped)
}

fn product(dimensions: &[u64]) -> Result<u64, String> {
    dimensions.iter().try_fold(1u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| "tensor indexing product exceeds u64".to_owned())
    })
}

fn vector_constant(emitter: &mut Emitter, value: u64, width: u64) -> Result<String, String> {
    let value =
        i32::try_from(value).map_err(|_| format!("TTIR index constant {value} exceeds i32"))?;
    let scalar = emitter.value();
    emitter.line(format!("{scalar} = arith.constant {value} : i32"));
    let vector = emitter.value();
    emitter.line(format!(
        "{vector} = tt.splat {scalar} : i32 -> tensor<{width}xi32>"
    ));
    Ok(vector)
}

fn signed_vector_constant(emitter: &mut Emitter, value: i64, width: u64) -> Result<String, String> {
    let value =
        i32::try_from(value).map_err(|_| format!("TTIR index constant {value} exceeds i32"))?;
    let scalar = emitter.value();
    emitter.line(format!("{scalar} = arith.constant {value} : i32"));
    let vector = emitter.value();
    emitter.line(format!(
        "{vector} = tt.splat {scalar} : i32 -> tensor<{width}xi32>"
    ));
    Ok(vector)
}

fn vector_binary_constant(
    emitter: &mut Emitter,
    operation: &str,
    value: &str,
    constant: u64,
    width: u64,
) -> Result<String, String> {
    let constant = vector_constant(emitter, constant, width)?;
    let result = emitter.value();
    emitter.line(format!(
        "{result} = {operation} {value}, {constant} : tensor<{width}xi32>"
    ));
    Ok(result)
}

fn checked_indices(
    emitter: &mut Emitter,
    indices: &Value,
    extent: u64,
    outer_mask: &str,
    width: u64,
) -> Result<(String, String), String> {
    let index_type = tensor_type(width, &indices.scalar);
    let i32_type = tensor_type(width, "i32");
    let offsets = match indices.scalar.as_str() {
        "i32" => indices.name.clone(),
        "i64" => {
            let value = emitter.value();
            emitter.line(format!(
                "{value} = arith.trunci {} : {index_type} to {i32_type}",
                indices.name
            ));
            value
        }
        other => {
            return Err(format!(
                "indexed operation requires i32/i64 indices, found {other}"
            ))
        }
    };
    let zero = vector_constant(emitter, 0, width)?;
    let nonnegative = emitter.value();
    emitter.line(format!(
        "{nonnegative} = arith.cmpi sge, {offsets}, {zero} : {i32_type}"
    ));
    let upper = vector_constant(emitter, extent, width)?;
    let in_bounds = emitter.value();
    emitter.line(format!(
        "{in_bounds} = arith.cmpi slt, {offsets}, {upper} : {i32_type}"
    ));
    let bounded = emitter.value();
    emitter.line(format!(
        "{bounded} = arith.andi {nonnegative}, {in_bounds} : tensor<{width}xi1>"
    ));
    let mask = emitter.value();
    emitter.line(format!(
        "{mask} = arith.andi {outer_mask}, {bounded} : tensor<{width}xi1>"
    ));
    Ok((offsets, mask))
}

fn load(
    emitter: &mut Emitter,
    argument: usize,
    node: &FusionNode,
    offsets: &str,
    mask: &str,
    width: u64,
) -> Value {
    let scalar = scalar_type(node).expect("validated scalar type");
    if matches!(&node.shape.rank, Rank::Ranked(axes) if axes.is_empty()) {
        let value = emitter.value();
        emitter.line(format!(
            "{value} = tt.splat %arg{argument} : {scalar} -> tensor<{width}x{scalar}>"
        ));
        return Value {
            name: value,
            scalar,
        };
    }
    let pointers = emitter.value();
    emitter.line(format!("{pointers} = tt.splat %arg{argument} : !tt.ptr<{scalar}> -> tensor<{width}x!tt.ptr<{scalar}>>"));
    let addresses = emitter.value();
    emitter.line(format!("{addresses} = tt.addptr {pointers}, {offsets} : tensor<{width}x!tt.ptr<{scalar}>>, tensor<{width}xi32>"));
    let tensor = tensor_type(width, &scalar);
    let zero = zero_tensor(emitter, &scalar, &tensor);
    let value = emitter.value();
    emitter.line(format!(
        "{value} = tt.load {addresses}, {mask}, {zero} : tensor<{width}x!tt.ptr<{scalar}>>"
    ));
    Value {
        name: value,
        scalar,
    }
}

fn store(
    emitter: &mut Emitter,
    argument: usize,
    value: &Value,
    offsets: &str,
    mask: &str,
    width: u64,
) {
    let pointers = emitter.value();
    emitter.line(format!(
        "{pointers} = tt.splat %arg{argument} : !tt.ptr<{}> -> tensor<{width}x!tt.ptr<{}>>",
        value.scalar, value.scalar
    ));
    let addresses = emitter.value();
    emitter.line(format!("{addresses} = tt.addptr {pointers}, {offsets} : tensor<{width}x!tt.ptr<{}>>, tensor<{width}xi32>", value.scalar));
    emitter.line(format!(
        "tt.store {addresses}, {}, {mask} : tensor<{width}x!tt.ptr<{}>>",
        value.name, value.scalar
    ));
}

fn zero_tensor(emitter: &mut Emitter, scalar: &str, tensor: &str) -> String {
    let constant = emitter.value();
    let literal = if scalar.starts_with('f') || scalar == "bf16" {
        "0.0"
    } else {
        "0"
    };
    emitter.line(format!("{constant} = arith.constant {literal} : {scalar}"));
    let value = emitter.value();
    emitter.line(format!(
        "{value} = tt.splat {constant} : {scalar} -> {tensor}"
    ));
    value
}

fn passthrough(node: &FusionNode, operands: &[&Value]) -> Result<Value, String> {
    operands
        .first()
        .map(|value| (*value).clone())
        .ok_or_else(|| format!("{} has no data operand", node.operation))
}

fn tensor_type(width: u64, scalar: &str) -> String {
    format!("tensor<{width}x{scalar}>")
}

fn scalar_type(node: &FusionNode) -> Result<String, String> {
    Ok(match node.shape.element_kind {
        ElementKind::SignedInteger | ElementKind::UnsignedInteger => {
            format!("i{}", node.shape.element_bits)
        }
        ElementKind::IeeeFloat => format!("f{}", node.shape.element_bits),
        ElementKind::BrainFloat if node.shape.element_bits == 16 => "bf16".into(),
        ElementKind::Float8E4M3Fn => "f8E4M3FN".into(),
        ElementKind::Float8E5M2 => "f8E5M2".into(),
        ElementKind::Boolean => "i1".into(),
        ElementKind::Opaque => "i64".into(),
        kind => {
            return Err(format!(
                "unsupported {kind:?} element width {}",
                node.shape.element_bits
            ))
        }
    })
}

fn node_operand_is_metadata(graph: &FusionGraph, region: &FusionRegion, id: NodeId) -> bool {
    region.nodes.iter().any(|node| {
        let node = graph.node(*node);
        node.inputs
            .iter()
            .zip(&node.operand_roles)
            .any(|(input, role)| {
                *input == id && !matches!(role, severian_fusion::OperandRole::Data)
            })
    })
}
