use severian_compile::CompileRegion;
use severian_fusion::{
    Dimension, ElementKind, FusionGraph, FusionNode, GraphError, NodeId, NodeKind, Shape,
};
use severian_universal::{
    tensor, FloatFormat, IntegerWidth, PrimitiveRepresentation, TensorDimension, TensorShape,
    TypeContext, TypeId,
};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionGraphError {
    UnknownOperation { index: usize },
    MissingValueSlot(u32),
    DuplicateValueSlot(u32),
    UnsupportedResultCount { index: usize, count: usize },
    UnsupportedType(TypeId),
    InvalidGraph(GraphError),
}

impl fmt::Display for FusionGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOperation { index } => {
                write!(formatter, "tensor operation {index} has no structural kind")
            }
            Self::MissingValueSlot(slot) => write!(formatter, "tensor region has no slot {slot}"),
            Self::DuplicateValueSlot(slot) => {
                write!(
                    formatter,
                    "tensor region defines slot {slot} more than once"
                )
            }
            Self::UnsupportedResultCount { index, count } => write!(
                formatter,
                "tensor operation {index} has {count} results; fusion currently requires one"
            ),
            Self::UnsupportedType(ty) => {
                write!(
                    formatter,
                    "type {ty:?} cannot be represented in a fusion graph"
                )
            }
            Self::InvalidGraph(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FusionGraphError {}

/// Converts a complete straight-line TensorCompiler region into the graph
/// consumed by the XLA-derived planner. Region slots, rather than operation
/// order guesses, preserve producer/consumer identity.
pub fn fusion_graph(
    region: &CompileRegion,
    types: &TypeContext,
) -> Result<FusionGraph, FusionGraphError> {
    let mut nodes = Vec::new();
    let mut slots = BTreeMap::new();
    for (slot, value) in region.inputs.iter().enumerate() {
        let id = NodeId(nodes.len() as u32);
        let mut node = FusionNode::structural(
            id.0,
            NodeKind::Parameter,
            [],
            fusion_shape(value.type_id, types)?,
        );
        node.operation = "parameter".into();
        nodes.push(node);
        slots.insert(slot as u32, id);
    }

    let mut next_implicit_slot = region.inputs.len() as u32;
    for (index, operation) in region.compile_operations.iter().enumerate() {
        let structural = tensor::TensorOp::decode(operation.id, &operation.attributes)
            .ok_or(FusionGraphError::UnknownOperation { index })?;
        if operation.results.len() != 1 {
            return Err(FusionGraphError::UnsupportedResultCount {
                index,
                count: operation.results.len(),
            });
        }
        let operand_slots =
            if operation.operand_slots.is_empty() && region.compile_operations.len() == 1 {
                (0..operation.operands.len() as u32).collect::<Vec<_>>()
            } else {
                operation.operand_slots.clone()
            };
        let result_slot =
            if operation.result_slots.is_empty() && region.compile_operations.len() == 1 {
                let slot = next_implicit_slot;
                next_implicit_slot += 1;
                slot
            } else {
                *operation
                    .result_slots
                    .first()
                    .ok_or(FusionGraphError::UnsupportedResultCount { index, count: 0 })?
            };
        let inputs = operand_slots
            .iter()
            .map(|slot| {
                slots
                    .get(slot)
                    .copied()
                    .ok_or(FusionGraphError::MissingValueSlot(*slot))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shape = fusion_shape(operation.results[0], types)?;
        let kind = node_kind(structural);
        let id = NodeId(nodes.len() as u32);
        let mut node = FusionNode::structural(id.0, kind, inputs, shape);
        node.operation = structural
            .kind()
            .unwrap_or_else(|| operation_name(structural))
            .into();
        node.bytes_read = operation
            .operands
            .iter()
            .filter_map(|ty| fusion_shape(*ty, types).ok()?.byte_size())
            .sum();
        node.bytes_written = node.shape.byte_size().unwrap_or(0);
        node.flops = estimate_flops(
            structural,
            operation.operands.as_slice(),
            &node.shape,
            types,
        );
        if matches!(structural, tensor::TensorOp::Reduce(_)) {
            // Mirrors XLA's conservative column-reduction cache estimate:
            // 32 * (maximum vector width * 32 + 1) elements.
            node.shared_memory_bytes = 32 * (4 * 32 + 1) * u64::from(node.shape.element_bytes);
            node.unnested_reductions = 1;
        }
        if structural == tensor::TensorOp::Scatter {
            node.has_side_effects = true;
            node.writes_input = Some(0);
        }
        if slots.insert(result_slot, id).is_some() {
            return Err(FusionGraphError::DuplicateValueSlot(result_slot));
        }
        nodes.push(node);
    }
    FusionGraph::new(nodes).map_err(FusionGraphError::InvalidGraph)
}

fn node_kind(operation: tensor::TensorOp) -> NodeKind {
    match operation {
        tensor::TensorOp::Elementwise(_) => NodeKind::Elementwise,
        tensor::TensorOp::Reduce(_) => NodeKind::Reduction,
        tensor::TensorOp::Matmul => NodeKind::Contraction,
        tensor::TensorOp::ReshapeView(_) => NodeKind::Reshape,
        tensor::TensorOp::Permute(_) => NodeKind::Permute,
        tensor::TensorOp::Slice => NodeKind::Slice,
        tensor::TensorOp::Broadcast(_) => NodeKind::Broadcast,
        tensor::TensorOp::Gather => NodeKind::Gather,
        tensor::TensorOp::Scatter => NodeKind::Scatter,
        tensor::TensorOp::Concatenate => NodeKind::Concatenate,
        tensor::TensorOp::Convert => NodeKind::Convert,
        tensor::TensorOp::StorageView(_) => NodeKind::StorageView,
    }
}

const fn operation_name(operation: tensor::TensorOp) -> &'static str {
    match operation {
        tensor::TensorOp::Matmul => "matmul",
        tensor::TensorOp::Slice => "slice",
        tensor::TensorOp::Gather => "gather",
        tensor::TensorOp::Scatter => "scatter",
        tensor::TensorOp::Concatenate => "concatenate",
        tensor::TensorOp::Convert => "convert",
        _ => "structural",
    }
}

fn fusion_shape(type_id: TypeId, types: &TypeContext) -> Result<Shape, FusionGraphError> {
    if let Some(tensor) = types.tensor(type_id) {
        let element_bytes = tensor::TensorElementKind::from_type(types, tensor.element)
            .map(tensor::TensorElementKind::byte_width)
            .ok_or(FusionGraphError::UnsupportedType(tensor.element))?;
        let dimensions = match tensor.shape {
            TensorShape::Unranked => None,
            TensorShape::Ranked(dimensions) => Some(dimensions
                .into_iter()
                .map(|dimension| match dimension {
                    TensorDimension::Dynamic => Dimension::Dynamic,
                    TensorDimension::Known(value) => Dimension::Known(value),
                })
                .collect::<Vec<_>>()),
        };
        let element_kind = match tensor::TensorElementKind::from_type(types, tensor.element)
            .ok_or(FusionGraphError::UnsupportedType(tensor.element))?
        {
            tensor::TensorElementKind::SignedInteger(_) => ElementKind::SignedInteger,
            tensor::TensorElementKind::UnsignedInteger(_) => ElementKind::UnsignedInteger,
            tensor::TensorElementKind::Float8E4M3Fn => ElementKind::Float8E4M3Fn,
            tensor::TensorElementKind::Float8E5M2 => ElementKind::Float8E5M2,
            tensor::TensorElementKind::IeeeFloat(_) => ElementKind::IeeeFloat,
            tensor::TensorElementKind::BrainFloat16 => ElementKind::BrainFloat,
        };
        return Ok(match dimensions {
            Some(dimensions) => {
                Shape::typed(dimensions, element_kind, u16::from(element_bytes))
            }
            None => Shape::unranked(element_kind, u16::from(element_bytes)),
        });
    }
    let primitive = types
        .primitive(type_id)
        .ok_or(FusionGraphError::UnsupportedType(type_id))?;
    let bits = match primitive.representation {
        PrimitiveRepresentation::Integer {
            bits: IntegerWidth::Fixed(bits),
            ..
        } => bits,
        PrimitiveRepresentation::Integer {
            bits: IntegerWidth::Machine,
            ..
        }
        | PrimitiveRepresentation::PointerInteger { .. } => 64,
        PrimitiveRepresentation::Float {
            format: FloatFormat::Float8E4M3Fn | FloatFormat::Float8E5M2,
        } => 8,
        PrimitiveRepresentation::Float {
            format: FloatFormat::BrainFloat16,
        } => 16,
        PrimitiveRepresentation::Float {
            format: FloatFormat::Ieee(bits),
        } => bits,
        PrimitiveRepresentation::Float {
            format: FloatFormat::Machine,
        } => 64,
        PrimitiveRepresentation::Boolean => 8,
        _ => return Err(FusionGraphError::UnsupportedType(type_id)),
    };
    let element_kind = match primitive.representation {
        PrimitiveRepresentation::Integer { signed: true, .. } => ElementKind::SignedInteger,
        PrimitiveRepresentation::Integer { signed: false, .. }
        | PrimitiveRepresentation::PointerInteger { .. } => ElementKind::UnsignedInteger,
        PrimitiveRepresentation::Float {
            format: FloatFormat::Float8E4M3Fn,
        } => ElementKind::Float8E4M3Fn,
        PrimitiveRepresentation::Float {
            format: FloatFormat::Float8E5M2,
        } => ElementKind::Float8E5M2,
        PrimitiveRepresentation::Float {
            format: FloatFormat::BrainFloat16,
        } => ElementKind::BrainFloat,
        PrimitiveRepresentation::Float { .. } => ElementKind::IeeeFloat,
        PrimitiveRepresentation::Boolean => ElementKind::Boolean,
        _ => return Err(FusionGraphError::UnsupportedType(type_id)),
    };
    Ok(Shape::typed([], element_kind, bits.div_ceil(8)))
}

fn estimate_flops(
    operation: tensor::TensorOp,
    operands: &[TypeId],
    result: &Shape,
    types: &TypeContext,
) -> u64 {
    if operation == tensor::TensorOp::Matmul {
        let Some(left) = operands.first().and_then(|ty| types.tensor(*ty)) else {
            return 0;
        };
        let Some(dimensions) = left.shape.dimensions() else {
            return 0;
        };
        let Some(TensorDimension::Known(k)) = dimensions.last() else {
            return 0;
        };
        return result
            .byte_size()
            .map(|bytes| bytes / u64::from(result.element_bytes) * k * 2)
            .unwrap_or(0);
    }
    let elements = result
        .byte_size()
        .map(|bytes| bytes / u64::from(result.element_bytes))
        .unwrap_or(0);
    match operation {
        tensor::TensorOp::Elementwise(
            tensor::ElementwiseOp::Exp
            | tensor::ElementwiseOp::Log
            | tensor::ElementwiseOp::Tanh
            | tensor::ElementwiseOp::Rsqrt,
        ) => elements.saturating_mul(4),
        tensor::TensorOp::Reduce(_) => operands
            .first()
            .and_then(|ty| fusion_shape(*ty, types).ok())
            .and_then(|shape| {
                shape
                    .byte_size()
                    .map(|bytes| bytes / u64::from(shape.element_bytes))
            })
            .unwrap_or(0),
        _ => elements,
    }
}
