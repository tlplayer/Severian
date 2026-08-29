//! Direct Severian fusion-to-GPU-MLIR lowering.
//!
//! Layout selection is Severian-owned and donor-derived. This emitter only
//! consumes structural fusion data and produces standard MLIR dialects; it
//! does not construct TTIR or call a Triton compiler.

use severian_compile::{CompileError, GpuKernelBundle};
use severian_fusion::{blocked_elementwise_schedule, ElementKind, NodeId, NodeKind, Rank};
use severian_mlir::{
    type_spelling, LoweredTensorDimension, LoweredTensorElement, LoweredTensorShape, LoweredType,
    MlirArtifact,
};
use std::collections::{BTreeMap, BTreeSet};

const WARPS_PER_BLOCK: u32 = 4;

pub fn lower_gpu_bundle_to_mlir(bundle: &GpuKernelBundle) -> Result<MlirArtifact, CompileError> {
    let region = match bundle.plan.regions.as_slice() {
        [region] => region,
        regions => {
            return Err(invalid(format!(
                "direct GPU MLIR currently requires one fusion region, found {}",
                regions.len()
            )))
        }
    };
    if region.nodes.is_empty() {
        return Err(invalid("direct GPU MLIR received an empty fusion region"));
    }
    if region.outputs.len() != bundle.outputs.len() {
        return Err(invalid(
            "fusion outputs do not match the compiled-region result signature",
        ));
    }

    let tensor_inputs = bundle
        .inputs
        .iter()
        .enumerate()
        .map(|(index, ty)| ranked_tensor(ty).map(|tensor| (index, tensor)))
        .collect::<Result<Vec<_>, _>>()?;
    let tensor_outputs = bundle
        .outputs
        .iter()
        .enumerate()
        .map(|(index, ty)| ranked_tensor(ty).map(|tensor| (index, tensor)))
        .collect::<Result<Vec<_>, _>>()?;
    let (_, reference) = tensor_inputs
        .first()
        .ok_or_else(|| invalid("direct GPU elementwise lowering requires a tensor input"))?;
    if reference.dimensions.is_empty() {
        return Err(invalid(
            "rank-zero values are scalar kernel arguments, not elementwise launch domains",
        ));
    }
    for (_, tensor) in tensor_inputs.iter().chain(&tensor_outputs) {
        if tensor.dimensions != reference.dimensions || tensor.element != reference.element {
            return Err(invalid(
                "the first direct GPU slice requires equal ranked elementwise tensor contracts",
            ));
        }
    }

    let warp_size = warp_size(&bundle.architecture);
    let static_elements = reference
        .dimensions
        .iter()
        .try_fold(1u64, |count, dimension| match dimension {
            LoweredTensorDimension::Known(value) => count.checked_mul(*value),
            LoweredTensorDimension::Dynamic => None,
        })
        .unwrap_or(1);
    let schedule = blocked_elementwise_schedule(static_elements, warp_size, WARPS_PER_BLOCK)
        .map_err(|error| invalid(format!("GPU layout legalization failed: {error}")))?;

    let parameter_nodes = (0..bundle.inputs.len())
        .map(|index| {
            bundle
                .value_nodes
                .get(&(index as u32))
                .copied()
                .ok_or_else(|| invalid(format!("GPU graph has no input slot {index}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parameter_indices = parameter_nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, node)| (node, index))
        .collect::<BTreeMap<_, _>>();
    let region_nodes = region.nodes.iter().copied().collect::<BTreeSet<_>>();
    for node in &region.nodes {
        let node = bundle.graph.node(*node);
        if node.kind != NodeKind::Elementwise {
            return Err(invalid(format!(
                "direct GPU MLIR has no {:?} lowering yet; it will not fall back to Triton",
                node.kind
            )));
        }
        if !matches!(
            node.operation.as_str(),
            "add" | "subtract" | "multiply" | "divide" | "relu"
        ) {
            return Err(invalid(format!(
                "direct GPU MLIR has no elementwise `{}` lowering yet",
                node.operation
            )));
        }
        if node.shape.rank != fusion_rank(&reference.dimensions)
            || node.shape.element_kind != fusion_element_kind(reference.element)
            || node.shape.element_bits != element_bits(reference.element)
        {
            return Err(invalid(format!(
                "elementwise node {} does not preserve the launch tensor contract",
                node.id.0
            )));
        }
    }

    let tensor_type = type_spelling(&bundle.inputs[tensor_inputs[0].0])
        .map_err(|error| invalid(error.to_string()))?;
    let memref_type = tensor_type
        .strip_prefix("tensor<")
        .and_then(|inner| inner.strip_suffix('>'))
        .map(|inner| format!("memref<{inner}>"))
        .ok_or_else(|| invalid("GPU tensor type has no builtin tensor spelling"))?;
    let scalar_type = scalar_type(reference.element)?;

    let parameters = bundle
        .inputs
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            type_spelling(ty)
                .map(|ty| format!("%arg{index}: {ty} {{bufferization.access = \"read\"}}"))
                .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let result_types = bundle
        .outputs
        .iter()
        .map(type_spelling)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid(error.to_string()))?;
    let result_signature = match result_types.as_slice() {
        [result] => format!(" -> {result}"),
        results => format!(" -> ({})", results.join(", ")),
    };

    let mut body = String::new();
    body.push_str("    %c0 = arith.constant 0 : index\n");
    body.push_str("    %c1 = arith.constant 1 : index\n");
    body.push_str(&format!(
        "    %threads = arith.constant {} : index\n",
        schedule.threads_per_block
    ));
    for axis in 2..reference.dimensions.len() {
        body.push_str(&format!("    %c{axis} = arith.constant {axis} : index\n"));
    }
    for axis in 0..reference.dimensions.len() {
        body.push_str(&format!(
            "    %d{axis} = tensor.dim %arg{}, %c{} : {tensor_type}\n",
            tensor_inputs[0].0, axis
        ));
    }
    body.push_str("    %elements_0 = arith.constant 1 : index\n");
    for axis in 0..reference.dimensions.len() {
        body.push_str(&format!(
            "    %elements_{} = arith.muli %elements_{}, %d{axis} : index\n",
            axis + 1,
            axis
        ));
    }
    let total = format!("%elements_{}", reference.dimensions.len());
    body.push_str(&format!(
        "    %blocks = arith.ceildivui {total}, %threads : index\n"
    ));

    let dynamic_dimensions = reference
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(axis, dimension)| {
            matches!(dimension, LoweredTensorDimension::Dynamic).then(|| format!("%d{axis}"))
        })
        .collect::<Vec<_>>()
        .join(", ");
    for (index, _) in &tensor_outputs {
        body.push_str(&format!(
            "    %out_tensor_{index} = tensor.empty({dynamic_dimensions}) : {}\n",
            result_types[*index]
        ));
    }
    for (index, _) in &tensor_inputs {
        body.push_str(&format!(
            "    %arg_buffer_{index} = bufferization.to_buffer %arg{index} read_only : {tensor_type} to {memref_type}\n"
        ));
    }
    for (index, _) in &tensor_outputs {
        body.push_str(&format!(
            "    %out_buffer_{index} = bufferization.to_buffer %out_tensor_{index} : {} to {memref_type}\n",
            result_types[*index]
        ));
    }
    body.push_str(
        "    gpu.launch blocks(%block_x, %block_y, %block_z) in (%grid_x = %blocks, %grid_y = %c1, %grid_z = %c1) threads(%thread_x, %thread_y, %thread_z) in (%block_size_x = %threads, %block_size_y = %c1, %block_size_z = %c1) {\n",
    );
    body.push_str("      %block_offset = arith.muli %block_x, %block_size_x : index\n");
    body.push_str("      %linear = arith.addi %block_offset, %thread_x : index\n");
    body.push_str(&format!(
        "      %active = arith.cmpi ult, %linear, {total} : index\n"
    ));
    body.push_str("      scf.if %active {\n");

    let indices = delinearize(&mut body, reference.dimensions.len());
    let index_list = indices.join(", ");
    let mut values = BTreeMap::<NodeId, String>::new();
    for node in &region.inputs {
        let input = parameter_indices.get(node).copied().ok_or_else(|| {
            invalid(format!(
                "fusion input node {} is not a compiled-region parameter",
                node.0
            ))
        })?;
        let value = format!("%node_{}", node.0);
        body.push_str(&format!(
            "        {value} = memref.load %arg_buffer_{input}[{index_list}] : {memref_type}\n"
        ));
        values.insert(*node, value);
    }
    for node_id in &region.nodes {
        let node = bundle.graph.node(*node_id);
        let operands = node
            .inputs
            .iter()
            .filter(|input| region_nodes.contains(input) || region.inputs.contains(input))
            .map(|input| {
                values.get(input).cloned().ok_or_else(|| {
                    invalid(format!(
                        "elementwise node {} uses unavailable node {}",
                        node.id.0, input.0
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = format!("%node_{}", node.id.0);
        emit_scalar_operation(&mut body, &result, &node.operation, &operands, &scalar_type)?;
        values.insert(node.id, result);
    }
    for (output_index, node) in region.outputs.iter().enumerate() {
        let value = values
            .get(node)
            .ok_or_else(|| invalid(format!("fusion output node {} has no scalar value", node.0)))?;
        body.push_str(&format!(
            "        memref.store {value}, %out_buffer_{output_index}[{index_list}] : {memref_type}\n"
        ));
    }
    body.push_str("      }\n");
    body.push_str("      gpu.terminator\n");
    body.push_str("    }\n");
    let returned = (0..tensor_outputs.len())
        .map(|index| format!("%out_tensor_{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str(&format!(
        "    return {returned} : {}\n",
        result_types.join(", ")
    ));

    Ok(MlirArtifact {
        module: format!(
            "module attributes {{severian.gpu.architecture = \"{}\"}} {{\n  func.func @entry({parameters}){result_signature} {{\n{body}  }}\n}}",
            bundle.architecture
        ),
        inputs: bundle.inputs.clone(),
        outputs: bundle.outputs.clone(),
    })
}

fn delinearize(body: &mut String, rank: usize) -> Vec<String> {
    let mut indices = vec![String::new(); rank];
    let mut remaining = "%linear".to_owned();
    for axis in (0..rank).rev() {
        let index = format!("%index_{axis}");
        body.push_str(&format!(
            "        {index} = arith.remui {remaining}, %d{axis} : index\n"
        ));
        indices[axis] = index;
        if axis != 0 {
            let next = format!("%remaining_{axis}");
            body.push_str(&format!(
                "        {next} = arith.divui {remaining}, %d{axis} : index\n"
            ));
            remaining = next;
        }
    }
    indices
}

fn emit_scalar_operation(
    body: &mut String,
    result: &str,
    operation: &str,
    operands: &[String],
    ty: &str,
) -> Result<(), CompileError> {
    let floating = ty.starts_with('f') || ty == "bf16";
    match (operation, operands) {
        ("add", [left, right]) => body.push_str(&format!(
            "        {result} = arith.{} {left}, {right} : {ty}\n",
            if floating { "addf" } else { "addi" }
        )),
        ("subtract", [left, right]) => body.push_str(&format!(
            "        {result} = arith.{} {left}, {right} : {ty}\n",
            if floating { "subf" } else { "subi" }
        )),
        ("multiply", [left, right]) => body.push_str(&format!(
            "        {result} = arith.{} {left}, {right} : {ty}\n",
            if floating { "mulf" } else { "muli" }
        )),
        ("divide", [left, right]) => body.push_str(&format!(
            "        {result} = arith.{} {left}, {right} : {ty}\n",
            if floating { "divf" } else { "divsi" }
        )),
        ("relu", [value]) => {
            let zero = format!("{result}_zero");
            let positive = format!("{result}_positive");
            body.push_str(&format!(
                "        {zero} = arith.constant {} : {ty}\n",
                if floating { "0.0" } else { "0" }
            ));
            body.push_str(&format!(
                "        {positive} = arith.cmp{} {}, {value}, {zero} : {ty}\n",
                if floating { "f" } else { "i" },
                if floating { "ogt" } else { "sgt" }
            ));
            body.push_str(&format!(
                "        {result} = arith.select {positive}, {value}, {zero} : {ty}\n"
            ));
        }
        _ => {
            return Err(invalid(format!(
                "elementwise `{operation}` received {} scalar operands",
                operands.len()
            )))
        }
    }
    Ok(())
}

struct RankedTensor<'a> {
    dimensions: &'a [LoweredTensorDimension],
    element: LoweredTensorElement,
}

fn ranked_tensor(ty: &LoweredType) -> Result<RankedTensor<'_>, CompileError> {
    let LoweredType::Tensor {
        shape: LoweredTensorShape::Ranked(dimensions),
        element,
    } = ty
    else {
        return Err(invalid(
            "direct GPU MLIR requires ranked tensors after specialization",
        ));
    };
    Ok(RankedTensor {
        dimensions,
        element: *element,
    })
}

fn fusion_rank(dimensions: &[LoweredTensorDimension]) -> Rank {
    Rank::Ranked(
        dimensions
            .iter()
            .map(|dimension| match dimension {
                LoweredTensorDimension::Known(value) => severian_fusion::Dimension::Known(*value),
                LoweredTensorDimension::Dynamic => severian_fusion::Dimension::Dynamic,
            })
            .collect(),
    )
}

const fn fusion_element_kind(element: LoweredTensorElement) -> ElementKind {
    match element {
        LoweredTensorElement::Integer { signed: true, .. } => ElementKind::SignedInteger,
        LoweredTensorElement::Integer { signed: false, .. } => ElementKind::UnsignedInteger,
        LoweredTensorElement::Float {
            format: severian_mlir::LoweredFloatFormat::Float8E4M3Fn,
        } => ElementKind::Float8E4M3Fn,
        LoweredTensorElement::Float {
            format: severian_mlir::LoweredFloatFormat::Float8E5M2,
        } => ElementKind::Float8E5M2,
        LoweredTensorElement::Float {
            format: severian_mlir::LoweredFloatFormat::BrainFloat16,
        } => ElementKind::BrainFloat,
        LoweredTensorElement::Float { .. } => ElementKind::IeeeFloat,
        LoweredTensorElement::Boolean => ElementKind::Boolean,
    }
}

const fn element_bits(element: LoweredTensorElement) -> u16 {
    match element {
        LoweredTensorElement::Integer { bits, .. } => bits,
        LoweredTensorElement::Float {
            format:
                severian_mlir::LoweredFloatFormat::Float8E4M3Fn
                | severian_mlir::LoweredFloatFormat::Float8E5M2,
        } => 8,
        LoweredTensorElement::Float {
            format: severian_mlir::LoweredFloatFormat::BrainFloat16,
        } => 16,
        LoweredTensorElement::Float {
            format: severian_mlir::LoweredFloatFormat::Ieee(bits),
        } => bits,
        LoweredTensorElement::Boolean => 1,
    }
}

fn scalar_type(element: LoweredTensorElement) -> Result<String, CompileError> {
    if matches!(
        element,
        LoweredTensorElement::Float {
            format: severian_mlir::LoweredFloatFormat::Float8E4M3Fn
                | severian_mlir::LoweredFloatFormat::Float8E5M2
        }
    ) {
        return Err(invalid(
            "direct GPU float8 arithmetic requires an explicit convert/accumulation plan",
        ));
    }
    type_spelling(&match element {
        LoweredTensorElement::Integer { bits, signed } => LoweredType::Integer { bits, signed },
        LoweredTensorElement::Float { format } => LoweredType::Float { format },
        LoweredTensorElement::Boolean => LoweredType::Boolean,
    })
    .map_err(|error| invalid(error.to_string()))
}

fn warp_size(architecture: &str) -> u32 {
    if architecture.starts_with("gfx9") {
        64
    } else {
        32
    }
}

fn invalid(message: impl Into<String>) -> CompileError {
    CompileError::InvalidArtifact(message.into())
}
