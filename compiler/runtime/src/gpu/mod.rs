//! Severian-owned GPU execution services.
//!
//! Drivers implement one target-neutral contract. Triton is a compiler
//! provider, not the owner of devices, memory, launch ordering, or caching.

mod cache;

pub use cache::{CacheKey, KernelCache};

use severian_fusion::{
    Dimension, DimensionExpression, ElementKind, FusionGraph, FusionPlan, FusionRegion, GpuTarget,
    KernelSpecialization, NodeId, NodeKind, Rank, RegionId, RuntimeShape, RuntimeStrides,
    StorageLayout,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::Duration;

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
pub struct StorageSpecializationBinding {
    pub node: NodeId,
    pub view: crate::StorageView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedStorageInputs {
    pub specialization: KernelSpecialization,
    /// Pointer arguments in the same order as the supplied region bindings.
    /// Shape, stride, and offset fields live in `specialization` and therefore
    /// participate in TTIR construction and cache identity before launch.
    pub arguments: Vec<KernelArgument>,
}

/// One host-resident tensor descriptor and its safely borrowed payload. The
/// descriptor supplies type/rank/layout metadata; the payload supplies bytes
/// for the selected device. Native pointers are never reinterpreted as MLIR
/// tensors by this API.
pub struct HostStorageInput<'a> {
    pub node: NodeId,
    pub view: crate::StorageView,
    pub bytes: &'a [u8],
}

/// One rank-zero value passed directly through the kernel ABI. Scalars are
/// ordinary graph data, but unlike tensors they do not own or address a GPU
/// buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostScalarInput {
    pub node: NodeId,
    pub bytes: Vec<u8>,
    pub alignment: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphExecution {
    pub specialization: KernelSpecialization,
    pub buffers: BTreeMap<NodeId, BufferId>,
    pub execution: ExecutionResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageSpecializationError {
    UnknownNode(NodeId),
    ElementMismatch {
        node: NodeId,
        expected_kind: ElementKind,
        expected_bits: u16,
        found_kind: ElementKind,
        found_bits: u16,
    },
    InvalidElementWidth(u32),
    InvalidElementRepresentation(crate::StorageElementRepresentationAbi),
    ShapeInference {
        node: NodeId,
        message: String,
    },
    StrideInference {
        node: NodeId,
        message: String,
    },
    InvalidSpecialization(severian_fusion::SpecializationError),
}

impl fmt::Display for StorageSpecializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StorageSpecializationError {}

/// Copies runtime tensor metadata into the compiler-owned specialization.
/// This function never dereferences `StorageView::data`; native storage is not
/// an MLIR tensor and cannot be used to infer rank inside an emitter.
pub fn specialize_storage_views(
    graph: &FusionGraph,
    target: GpuTarget,
    bindings: &[StorageSpecializationBinding],
) -> Result<KernelSpecialization, StorageSpecializationError> {
    let mut shapes = Vec::with_capacity(bindings.len());
    let mut strides = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let Some(node) = graph.nodes().get(binding.node.0 as usize) else {
            return Err(StorageSpecializationError::UnknownNode(binding.node));
        };
        let (found_kind, found_bits) = storage_element(&binding.view.element)?;
        if found_kind != node.shape.element_kind || found_bits != node.shape.element_bits {
            return Err(StorageSpecializationError::ElementMismatch {
                node: binding.node,
                expected_kind: node.shape.element_kind,
                expected_bits: node.shape.element_bits,
                found_kind,
                found_bits,
            });
        }
        shapes.push(RuntimeShape {
            node: binding.node,
            dimensions: binding.view.dimensions.clone(),
        });
        strides.push(RuntimeStrides {
            node: binding.node,
            strides: binding.view.strides.clone(),
            offset: binding.view.offset,
        });
    }
    complete_kernel_specialization(
        graph,
        KernelSpecialization {
            shapes,
            strides,
            target,
        },
    )
}

/// Completes a partial runtime specialization in graph order. Input
/// descriptors remain the source of truth; result shapes are derived from the
/// structural operation and never from dtype- or rank-specific symbols.
pub fn complete_kernel_specialization(
    graph: &FusionGraph,
    partial: KernelSpecialization,
) -> Result<KernelSpecialization, StorageSpecializationError> {
    let target = partial.target;
    let mut shapes = partial
        .shapes
        .into_iter()
        .map(|shape| (shape.node, shape.dimensions))
        .collect::<BTreeMap<_, _>>();
    let mut strides = partial
        .strides
        .into_iter()
        .map(|strides| (strides.node, (strides.strides, strides.offset)))
        .collect::<BTreeMap<_, _>>();

    for node in graph.nodes() {
        let inferred = match shapes.get(&node.id) {
            Some(shape) => Some(shape.clone()),
            None => infer_node_shape(graph, node.id, &shapes)?,
        };
        let dimensions = concrete_node_shape(node.id, &node.shape.rank, inferred)?;
        shapes.insert(node.id, dimensions.clone());

        if !strides.contains_key(&node.id) {
            let inferred = infer_node_strides(graph, node.id, &dimensions, &strides)?;
            strides.insert(node.id, inferred);
        }
    }

    validate_symbolic_dimensions(graph, &shapes)?;

    let specialization = KernelSpecialization {
        shapes: shapes
            .into_iter()
            .map(|(node, dimensions)| RuntimeShape { node, dimensions })
            .collect(),
        strides: strides
            .into_iter()
            .map(|(node, (strides, offset))| RuntimeStrides {
                node,
                strides,
                offset,
            })
            .collect(),
        target,
    };
    specialization
        .validate(graph, target)
        .map_err(StorageSpecializationError::InvalidSpecialization)?;
    Ok(specialization)
}

fn validate_symbolic_dimensions(
    graph: &FusionGraph,
    shapes: &BTreeMap<NodeId, Vec<u64>>,
) -> Result<(), StorageSpecializationError> {
    let mut symbols = BTreeMap::new();
    for node in graph.nodes() {
        let Some(dimensions) = shapes.get(&node.id) else {
            continue;
        };
        if !node.shape.dimension_expressions.is_empty()
            && node.shape.dimension_expressions.len() != dimensions.len()
        {
            return Err(StorageSpecializationError::ShapeInference {
                node: node.id,
                message: format!(
                    "shape contract has {} expressions for runtime rank {}",
                    node.shape.dimension_expressions.len(),
                    dimensions.len()
                ),
            });
        }
        for (expression, extent) in node.shape.dimension_expressions.iter().zip(dimensions) {
            if let DimensionExpression::Symbol(symbol) = expression {
                if symbols
                    .insert(*symbol, *extent)
                    .is_some_and(|known| known != *extent)
                {
                    return Err(StorageSpecializationError::ShapeInference {
                        node: node.id,
                        message: format!(
                            "symbolic dimension {symbol} has conflicting runtime extents"
                        ),
                    });
                }
            }
        }
    }
    for node in graph.nodes() {
        let Some(dimensions) = shapes.get(&node.id) else {
            continue;
        };
        for (axis, (expression, extent)) in node
            .shape
            .dimension_expressions
            .iter()
            .zip(dimensions)
            .enumerate()
        {
            let expected =
                evaluate_dimension_expression(expression, &symbols).map_err(|message| {
                    StorageSpecializationError::ShapeInference {
                        node: node.id,
                        message,
                    }
                })?;
            if expected.is_some_and(|expected| expected != *extent) {
                return Err(StorageSpecializationError::ShapeInference {
                    node: node.id,
                    message: format!(
                        "axis {axis} violates its symbolic shape expression: expected {expected:?}, found {extent}"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn evaluate_dimension_expression(
    expression: &DimensionExpression,
    symbols: &BTreeMap<u64, u64>,
) -> Result<Option<u64>, String> {
    let binary = |left: &DimensionExpression, right: &DimensionExpression| {
        Ok::<_, String>(
            evaluate_dimension_expression(left, symbols)?
                .zip(evaluate_dimension_expression(right, symbols)?),
        )
    };
    match expression {
        DimensionExpression::Constant(value) => Ok(Some(*value)),
        DimensionExpression::Symbol(symbol) => Ok(symbols.get(symbol).copied()),
        DimensionExpression::Dynamic => Ok(None),
        DimensionExpression::Add(left, right) => binary(left, right)?
            .map(|(left, right)| {
                left.checked_add(right)
                    .ok_or_else(|| "symbolic dimension addition overflowed".to_owned())
            })
            .transpose(),
        DimensionExpression::Multiply(left, right) => binary(left, right)?
            .map(|(left, right)| {
                left.checked_mul(right)
                    .ok_or_else(|| "symbolic dimension multiplication overflowed".to_owned())
            })
            .transpose(),
        DimensionExpression::DivideExact(left, right) => binary(left, right)?
            .map(|(left, right)| {
                if right == 0 {
                    Err("symbolic dimension division by zero".to_owned())
                } else if left % right != 0 {
                    Err("symbolic dimension division was not exact".to_owned())
                } else {
                    Ok(left / right)
                }
            })
            .transpose(),
    }
}

fn concrete_node_shape(
    node: NodeId,
    contract: &Rank,
    inferred: Option<Vec<u64>>,
) -> Result<Vec<u64>, StorageSpecializationError> {
    match contract {
        Rank::Unranked => inferred.ok_or_else(|| StorageSpecializationError::ShapeInference {
            node,
            message: "unranked value requires runtime shape data or structural inference".into(),
        }),
        Rank::Ranked(dimensions) => {
            let provided = inferred.unwrap_or_else(|| {
                dimensions
                    .iter()
                    .filter_map(|dimension| match dimension {
                        Dimension::Known(value) => Some(*value),
                        Dimension::Dynamic => None,
                    })
                    .collect()
            });
            if provided.len() != dimensions.len() {
                return Err(StorageSpecializationError::ShapeInference {
                    node,
                    message: format!(
                        "ranked value requires {} dimensions but {} were inferred",
                        dimensions.len(),
                        provided.len()
                    ),
                });
            }
            for (axis, (expected, found)) in dimensions.iter().zip(&provided).enumerate() {
                if let Dimension::Known(expected) = expected {
                    if expected != found {
                        return Err(StorageSpecializationError::ShapeInference {
                            node,
                            message: format!(
                                "dimension {axis} requires {expected} but runtime inference produced {found}"
                            ),
                        });
                    }
                }
            }
            Ok(provided)
        }
    }
}

fn infer_node_shape(
    graph: &FusionGraph,
    node: NodeId,
    shapes: &BTreeMap<NodeId, Vec<u64>>,
) -> Result<Option<Vec<u64>>, StorageSpecializationError> {
    let node_data = graph.node(node);
    let input = |index: usize| {
        node_data
            .inputs
            .get(index)
            .and_then(|input| shapes.get(input))
            .cloned()
    };
    let data_shapes = || {
        node_data
            .inputs
            .iter()
            .zip(&node_data.operand_roles)
            .filter(|(_, role)| matches!(role, severian_fusion::OperandRole::Data))
            .filter_map(|(input, _)| shapes.get(input).cloned())
            .collect::<Vec<_>>()
    };
    let runtime_operand = |index: usize| {
        node_data
            .runtime_operands
            .iter()
            .find(|operand| usize::from(operand.input_index) == index)
            .map(|operand| operand.values.as_slice())
    };
    let inferred = match node_data.kind {
        NodeKind::Parameter | NodeKind::Constant => None,
        NodeKind::Elementwise => {
            let mut inputs = data_shapes().into_iter();
            let Some(mut result) = inputs.next() else {
                return Ok(None);
            };
            for shape in inputs {
                result = broadcast_runtime_shape(&result, &shape).ok_or_else(|| {
                    StorageSpecializationError::ShapeInference {
                        node,
                        message: format!(
                            "elementwise operands {:?} and {:?} do not broadcast",
                            result, shape
                        ),
                    }
                })?;
            }
            Some(result)
        }
        NodeKind::Reduction => {
            let Some(mut source) = input(0) else {
                return Ok(None);
            };
            match node_data.operation.as_str() {
                "sum" => Some(vec![1]),
                "sum_axis" => {
                    let axis = node_data.attributes.first().copied().ok_or_else(|| {
                        StorageSpecializationError::ShapeInference {
                            node,
                            message: "sum_axis has no structural axis identity".into(),
                        }
                    })?;
                    let axis = usize::try_from(axis).map_err(|_| {
                        StorageSpecializationError::ShapeInference {
                            node,
                            message: format!("sum_axis has invalid axis {axis}"),
                        }
                    })?;
                    if axis >= source.len() {
                        return Err(StorageSpecializationError::ShapeInference {
                            node,
                            message: format!(
                                "sum_axis axis {axis} is outside runtime rank {}",
                                source.len()
                            ),
                        });
                    }
                    source.remove(axis);
                    if source.is_empty() {
                        source.push(1);
                    }
                    Some(source)
                }
                "mean_last" | "max_last" => {
                    if source.is_empty() {
                        return Err(StorageSpecializationError::ShapeInference {
                            node,
                            message: "last-axis reduction requires rank at least one".into(),
                        });
                    }
                    *source.last_mut().expect("rank checked above") = 1;
                    Some(source)
                }
                operation => {
                    return Err(StorageSpecializationError::ShapeInference {
                        node,
                        message: format!("unknown reduction operation `{operation}`"),
                    })
                }
            }
        }
        NodeKind::Contraction => match (input(0), input(1)) {
            (Some(left), Some(right)) => Some(matmul_runtime_shape(node, &left, &right)?),
            _ => None,
        },
        NodeKind::Reshape => match node_data.operation.as_str() {
            "materialize" => input(0),
            "reshape" => match (input(0), runtime_operand(1)) {
                (Some(source), Some(specification)) => {
                    Some(reshape_runtime_shape(node, &source, specification)?)
                }
                _ => None,
            },
            _ => None,
        },
        NodeKind::Permute => match node_data.operation.as_str() {
            "reverse" => input(0).map(|mut shape| {
                shape.reverse();
                shape
            }),
            "axes" => match (input(0), runtime_operand(1)) {
                (Some(source), Some(axes)) => Some(permute_runtime_shape(node, &source, axes)?),
                _ => None,
            },
            _ => None,
        },
        NodeKind::Broadcast => match node_data.operation.as_str() {
            "like" => input(1),
            "repeat" => match (input(0), runtime_operand(1)) {
                (Some(source), Some(specification)) => {
                    Some(repeat_runtime_shape(node, &source, specification)?)
                }
                _ => None,
            },
            _ => None,
        },
        NodeKind::Slice => match (
            input(0),
            runtime_operand(1),
            runtime_operand(2),
            runtime_operand(3),
        ) {
            (Some(source), Some(starts), Some(ends), Some(steps)) => {
                Some(slice_runtime_shape(node, &source, starts, ends, steps)?)
            }
            _ => None,
        },
        NodeKind::Gather => match (input(0), input(1)) {
            (Some(source), Some(indices)) => {
                if source.is_empty() {
                    return Err(StorageSpecializationError::ShapeInference {
                        node,
                        message: "gather source requires rank at least one".into(),
                    });
                }
                Some(
                    indices
                        .into_iter()
                        .chain(source.into_iter().skip(1))
                        .collect(),
                )
            }
            _ => None,
        },
        NodeKind::Concatenate => match (input(0), input(1), runtime_operand(2)) {
            (Some(left), Some(right), Some([axis])) => {
                Some(concatenate_runtime_shape(node, &left, &right, *axis)?)
            }
            (Some(_), Some(_), Some(_)) => {
                return Err(StorageSpecializationError::ShapeInference {
                    node,
                    message: "concatenate requires exactly one axis".into(),
                })
            }
            _ => None,
        },
        NodeKind::Scatter | NodeKind::Convert => input(0),
        NodeKind::StorageView => match node_data.operation.as_str() {
            "from_elements" => runtime_operand(1)
                .map(|shape| nonnegative_dimensions(node, shape))
                .transpose()?,
            _ => None,
        },
    };
    Ok(inferred)
}

fn nonnegative_dimensions(
    node: NodeId,
    values: &[i64],
) -> Result<Vec<u64>, StorageSpecializationError> {
    values
        .iter()
        .map(|value| {
            u64::try_from(*value).map_err(|_| StorageSpecializationError::ShapeInference {
                node,
                message: format!("negative tensor dimension {value}"),
            })
        })
        .collect()
}

fn reshape_runtime_shape(
    node: NodeId,
    source: &[u64],
    specification: &[i64],
) -> Result<Vec<u64>, StorageSpecializationError> {
    let source_elements = source
        .iter()
        .try_fold(1u64, |size, dimension| size.checked_mul(*dimension));
    let source_elements =
        source_elements.ok_or_else(|| StorageSpecializationError::ShapeInference {
            node,
            message: "reshape source element count overflowed".into(),
        })?;
    let mut result = Vec::with_capacity(specification.len());
    let mut inferred_axis = None;
    let mut known_elements = 1u64;
    for (axis, dimension) in specification.iter().copied().enumerate() {
        if dimension == -1 {
            if inferred_axis.replace(axis).is_some() {
                return Err(StorageSpecializationError::ShapeInference {
                    node,
                    message: "reshape contains more than one inferred dimension".into(),
                });
            }
            result.push(1);
            continue;
        }
        let dimension =
            u64::try_from(dimension).map_err(|_| StorageSpecializationError::ShapeInference {
                node,
                message: format!("reshape contains negative dimension {dimension}"),
            })?;
        known_elements = known_elements.checked_mul(dimension).ok_or_else(|| {
            StorageSpecializationError::ShapeInference {
                node,
                message: "reshape result element count overflowed".into(),
            }
        })?;
        result.push(dimension);
    }
    if let Some(axis) = inferred_axis {
        if known_elements == 0 || source_elements % known_elements != 0 {
            return Err(StorageSpecializationError::ShapeInference {
                node,
                message: "reshape inferred dimension is not integral".into(),
            });
        }
        result[axis] = source_elements / known_elements;
    } else if known_elements != source_elements {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "reshape changes the element count".into(),
        });
    }
    Ok(result)
}

fn permute_runtime_shape(
    node: NodeId,
    source: &[u64],
    axes: &[i64],
) -> Result<Vec<u64>, StorageSpecializationError> {
    if axes.len() != source.len() {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "permutation rank does not match its source".into(),
        });
    }
    let mut seen = BTreeSet::new();
    axes.iter()
        .map(|axis| {
            let axis =
                usize::try_from(*axis).map_err(|_| StorageSpecializationError::ShapeInference {
                    node,
                    message: format!("invalid permutation axis {axis}"),
                })?;
            if axis >= source.len() || !seen.insert(axis) {
                return Err(StorageSpecializationError::ShapeInference {
                    node,
                    message: format!("invalid permutation axis {axis}"),
                });
            }
            Ok(source[axis])
        })
        .collect()
}

fn slice_runtime_shape(
    node: NodeId,
    source: &[u64],
    starts: &[i64],
    ends: &[i64],
    steps: &[i64],
) -> Result<Vec<u64>, StorageSpecializationError> {
    if starts.len() != source.len() || ends.len() != source.len() || steps.len() != source.len() {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "slice specification rank does not match its source".into(),
        });
    }
    (0..source.len())
        .map(|axis| {
            let (start, end, step) = (starts[axis], ends[axis], steps[axis]);
            if start < 0 || end < start || step <= 0 || end as u64 > source[axis] {
                return Err(StorageSpecializationError::ShapeInference {
                    node,
                    message: format!("invalid slice bounds on axis {axis}"),
                });
            }
            Ok(((end - start + step - 1) / step) as u64)
        })
        .collect()
}

fn repeat_runtime_shape(
    node: NodeId,
    source: &[u64],
    specification: &[i64],
) -> Result<Vec<u64>, StorageSpecializationError> {
    let [axis, repeats] = specification else {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "repeat requires [axis, copies]".into(),
        });
    };
    let axis = usize::try_from(*axis).map_err(|_| StorageSpecializationError::ShapeInference {
        node,
        message: format!("invalid repeat axis {axis}"),
    })?;
    let repeats = u64::try_from(*repeats)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| StorageSpecializationError::ShapeInference {
            node,
            message: format!("invalid repeat count {repeats}"),
        })?;
    if axis >= source.len() {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: format!(
                "repeat axis {axis} is outside runtime rank {}",
                source.len()
            ),
        });
    }
    let mut result = source.to_vec();
    result[axis] = result[axis].checked_mul(repeats).ok_or_else(|| {
        StorageSpecializationError::ShapeInference {
            node,
            message: "repeat dimension overflowed".into(),
        }
    })?;
    Ok(result)
}

fn concatenate_runtime_shape(
    node: NodeId,
    left: &[u64],
    right: &[u64],
    axis: i64,
) -> Result<Vec<u64>, StorageSpecializationError> {
    if left.len() != right.len() {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "concatenate ranks are incompatible".into(),
        });
    }
    let rank =
        i64::try_from(left.len()).map_err(|_| StorageSpecializationError::ShapeInference {
            node,
            message: "concatenate rank exceeds the runtime ABI".into(),
        })?;
    let normalized = if axis < 0 {
        rank.checked_add(axis)
    } else {
        Some(axis)
    };
    let axis = normalized
        .and_then(|axis| usize::try_from(axis).ok())
        .filter(|axis| *axis < left.len())
        .ok_or_else(|| StorageSpecializationError::ShapeInference {
            node,
            message: format!("invalid concatenate axis {axis} for rank {rank}"),
        })?;
    if left
        .iter()
        .zip(right)
        .enumerate()
        .any(|(candidate, (left, right))| candidate != axis && left != right)
    {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "concatenate non-axis dimensions differ".into(),
        });
    }
    let mut result = left.to_vec();
    result[axis] = result[axis].checked_add(right[axis]).ok_or_else(|| {
        StorageSpecializationError::ShapeInference {
            node,
            message: "concatenate dimension overflowed".into(),
        }
    })?;
    Ok(result)
}

fn broadcast_runtime_shape(left: &[u64], right: &[u64]) -> Option<Vec<u64>> {
    let rank = left.len().max(right.len());
    let mut result = vec![1; rank];
    for offset in 0..rank {
        let left = left
            .len()
            .checked_sub(offset + 1)
            .map(|axis| left[axis])
            .unwrap_or(1);
        let right = right
            .len()
            .checked_sub(offset + 1)
            .map(|axis| right[axis])
            .unwrap_or(1);
        result[rank - offset - 1] = if left == right {
            left
        } else if left == 1 {
            right
        } else if right == 1 {
            left
        } else {
            return None;
        };
    }
    Some(result)
}

fn matmul_runtime_shape(
    node: NodeId,
    left: &[u64],
    right: &[u64],
) -> Result<Vec<u64>, StorageSpecializationError> {
    if left.len() < 2 || right.len() < 2 {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "matmul requires runtime rank at least two".into(),
        });
    }
    if left[left.len() - 1] != right[right.len() - 2] {
        return Err(StorageSpecializationError::ShapeInference {
            node,
            message: "matmul contraction dimensions do not match".into(),
        });
    }
    let batch = broadcast_runtime_shape(&left[..left.len() - 2], &right[..right.len() - 2])
        .ok_or_else(|| StorageSpecializationError::ShapeInference {
            node,
            message: "matmul batch dimensions do not broadcast".into(),
        })?;
    let mut result = batch;
    result.push(left[left.len() - 2]);
    result.push(right[right.len() - 1]);
    Ok(result)
}

fn infer_node_strides(
    graph: &FusionGraph,
    node: NodeId,
    dimensions: &[u64],
    known: &BTreeMap<NodeId, (Vec<i64>, i64)>,
) -> Result<(Vec<i64>, i64), StorageSpecializationError> {
    let node_data = graph.node(node);
    if node_data.kind == NodeKind::Parameter && matches!(node_data.layout, StorageLayout::Runtime) {
        return Err(StorageSpecializationError::StrideInference {
            node,
            message: "runtime-layout parameter requires descriptor strides".into(),
        });
    }
    if node_data.kind == NodeKind::Scatter {
        if let Some(input) = node_data.inputs.first().and_then(|input| known.get(input)) {
            return Ok(input.clone());
        }
    }
    let source_layout = || {
        node_data
            .inputs
            .first()
            .and_then(|input| known.get(input))
            .cloned()
    };
    if node_data.kind == NodeKind::Permute {
        if let Some((mut strides, offset)) = source_layout() {
            if node_data.operation == "axes" {
                let axes = node_data
                    .runtime_operands
                    .iter()
                    .find(|operand| operand.input_index == 1)
                    .map(|operand| operand.values.as_slice())
                    .ok_or_else(|| StorageSpecializationError::StrideInference {
                        node,
                        message: "permutation has no runtime axes".into(),
                    })?;
                let source = strides.clone();
                strides = axes
                    .iter()
                    .map(|axis| {
                        usize::try_from(*axis)
                            .ok()
                            .and_then(|axis| source.get(axis).copied())
                            .ok_or_else(|| StorageSpecializationError::StrideInference {
                                node,
                                message: format!("invalid permutation axis {axis}"),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok((strides, offset));
            }
            strides.reverse();
            return Ok((strides, offset));
        }
    }
    if node_data.kind == NodeKind::Slice {
        if let Some((source_strides, mut offset)) = source_layout() {
            let operand = |index| {
                node_data
                    .runtime_operands
                    .iter()
                    .find(|operand| usize::from(operand.input_index) == index)
                    .map(|operand| operand.values.as_slice())
            };
            let (Some(starts), Some(steps)) = (operand(1), operand(3)) else {
                return Err(StorageSpecializationError::StrideInference {
                    node,
                    message: "slice has no runtime starts/steps".into(),
                });
            };
            if starts.len() != source_strides.len() || steps.len() != source_strides.len() {
                return Err(StorageSpecializationError::StrideInference {
                    node,
                    message: "slice layout rank does not match its source".into(),
                });
            }
            let mut strides = Vec::with_capacity(source_strides.len());
            for axis in 0..source_strides.len() {
                offset = offset
                    .checked_add(starts[axis].checked_mul(source_strides[axis]).ok_or_else(
                        || StorageSpecializationError::StrideInference {
                            node,
                            message: "slice offset overflowed".into(),
                        },
                    )?)
                    .ok_or_else(|| StorageSpecializationError::StrideInference {
                        node,
                        message: "slice offset overflowed".into(),
                    })?;
                strides.push(
                    steps[axis]
                        .checked_mul(source_strides[axis])
                        .ok_or_else(|| StorageSpecializationError::StrideInference {
                            node,
                            message: "slice stride overflowed".into(),
                        })?,
                );
            }
            return Ok((strides, offset));
        }
    }
    contiguous_runtime_strides(node, dimensions)
}

fn contiguous_runtime_strides(
    node: NodeId,
    dimensions: &[u64],
) -> Result<(Vec<i64>, i64), StorageSpecializationError> {
    let mut stride = 1i64;
    let mut strides = vec![0; dimensions.len()];
    for axis in (0..dimensions.len()).rev() {
        strides[axis] = stride;
        let dimension = i64::try_from(dimensions[axis]).map_err(|_| {
            StorageSpecializationError::StrideInference {
                node,
                message: format!("dimension {} exceeds the stride ABI", dimensions[axis]),
            }
        })?;
        stride = stride.checked_mul(dimension).ok_or_else(|| {
            StorageSpecializationError::StrideInference {
                node,
                message: "contiguous stride calculation overflowed".into(),
            }
        })?;
    }
    Ok((strides, 0))
}

pub fn prepare_storage_inputs(
    graph: &FusionGraph,
    target: GpuTarget,
    bindings: &[StorageSpecializationBinding],
) -> Result<PreparedStorageInputs, StorageSpecializationError> {
    let specialization = specialize_storage_views(graph, target, bindings)?;
    let arguments = bindings
        .iter()
        .map(|binding| KernelArgument::Scalar {
            bytes: binding.view.data.to_ne_bytes().to_vec(),
            alignment: core::mem::align_of::<u64>() as u8,
        })
        .collect();
    Ok(PreparedStorageInputs {
        specialization,
        arguments,
    })
}

fn storage_element(
    element: &crate::StorageElementRepresentationAbi,
) -> Result<(ElementKind, u16), StorageSpecializationError> {
    let bits = u16::try_from(element.bits)
        .map_err(|_| StorageSpecializationError::InvalidElementWidth(element.bits))?;
    let kind = match (element.kind, element.float_format) {
        (crate::StorageElementKind::SignedInteger, crate::StorageFloatFormat::None) => {
            ElementKind::SignedInteger
        }
        (crate::StorageElementKind::UnsignedInteger, crate::StorageFloatFormat::None) => {
            ElementKind::UnsignedInteger
        }
        (crate::StorageElementKind::Float, crate::StorageFloatFormat::BrainFloat) => {
            ElementKind::BrainFloat
        }
        (crate::StorageElementKind::Float, crate::StorageFloatFormat::Float8E4M3Fn) => {
            ElementKind::Float8E4M3Fn
        }
        (crate::StorageElementKind::Float, crate::StorageFloatFormat::Float8E5M2) => {
            ElementKind::Float8E5M2
        }
        (crate::StorageElementKind::Float, crate::StorageFloatFormat::Ieee) => {
            ElementKind::IeeeFloat
        }
        _ => {
            return Err(StorageSpecializationError::InvalidElementRepresentation(
                *element,
            ))
        }
    };
    Ok((kind, bits))
}

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
    /// Triton appends these two launcher-owned pointer arguments after the
    /// structural region inputs and outputs. Sizes are per grid program.
    pub global_scratch_bytes_per_program: u64,
    pub global_scratch_alignment: u64,
    pub profile_scratch_bytes_per_program: u64,
    pub profile_scratch_alignment: u64,
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

/// The complete runtime compiler kept in a Severian executable. It owns no
/// graph capture, Python state, device state, or shape inference: callers hand
/// it one already-typed fusion region plus the minimal concrete specialization
/// that the selected backend still requires.
pub struct TensorJit<C> {
    compiler: C,
    cache: KernelCache,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedKernelArtifact {
    pub key: CacheKey,
    pub artifact: KernelArtifact,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorJitError {
    Compiler(String),
    Cache(String),
}

impl fmt::Display for TensorJitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiler(error) => write!(formatter, "tensor JIT compiler failed: {error}"),
            Self::Cache(error) => write!(formatter, "tensor JIT cache failed: {error}"),
        }
    }
}

impl std::error::Error for TensorJitError {}

impl<C: GpuCompiler> TensorJit<C> {
    pub fn new(compiler: C, cache: KernelCache) -> Self {
        Self { compiler, cache }
    }

    pub fn resolve(
        &mut self,
        request: &KernelCompileRequest<'_>,
    ) -> Result<ResolvedKernelArtifact, TensorJitError> {
        request
            .specialization
            .validate_region(request.graph, request.region, request.options.target)
            .map_err(|error| TensorJitError::Compiler(error.to_string()))?;
        let key = CacheKey::for_kernel(
            request.graph,
            request.region,
            request.specialization,
            request.options,
            self.compiler.donor_revision(),
        );
        if let Some(artifact) = self
            .cache
            .get(&key)
            .map_err(|error| TensorJitError::Cache(error.to_string()))?
        {
            return Ok(ResolvedKernelArtifact {
                key,
                artifact,
                cache_hit: true,
            });
        }
        let artifact = self
            .compiler
            .compile(request)
            .map_err(TensorJitError::Compiler)?;
        self.cache
            .insert(key, artifact.clone())
            .map_err(|error| TensorJitError::Cache(error.to_string()))?;
        Ok(ResolvedKernelArtifact {
            key,
            artifact,
            cache_hit: false,
        })
    }

    pub fn cache(&self) -> &KernelCache {
        &self.cache
    }
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
    GpuSingleAllocationLimit {
        requested: u64,
        limit: u64,
    },
    GpuMemoryLimit {
        requested: u64,
        live: u64,
        limit: u64,
    },
    AddressOverflow,
    GridOverflow,
    MissingOutputShape(NodeId),
    Specialization(StorageSpecializationError),
    DuplicateStorageInput(NodeId),
    MissingNodeBuffer(NodeId),
    HostStorageLength {
        node: NodeId,
        descriptor_bytes: u64,
        supplied_bytes: usize,
    },
    TensorSizeOverflow(NodeId),
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
            Self::GpuSingleAllocationLimit { requested, limit } => write!(
                formatter,
                "GPU allocation of {requested} bytes exceeds the per-allocation limit of {limit} bytes"
            ),
            Self::GpuMemoryLimit {
                requested,
                live,
                limit,
            } => write!(
                formatter,
                "GPU allocation of {requested} bytes with {live} live bytes would exceed the {limit}-byte runtime budget"
            ),
            Self::AddressOverflow => formatter.write_str("GPU buffer address overflow"),
            Self::GridOverflow => formatter.write_str("GPU launch grid overflow"),
            Self::MissingOutputShape(node) => {
                write!(formatter, "node {} has no concrete runtime shape", node.0)
            }
            Self::Specialization(error) => {
                write!(formatter, "GPU specialization failed: {error}")
            }
            Self::DuplicateStorageInput(node) => {
                write!(formatter, "node {} has more than one storage input", node.0)
            }
            Self::MissingNodeBuffer(node) => {
                write!(formatter, "node {} has no allocated GPU buffer", node.0)
            }
            Self::HostStorageLength {
                node,
                descriptor_bytes,
                supplied_bytes,
            } => write!(
                formatter,
                "node {} describes {} storage bytes but only {} were supplied",
                node.0, descriptor_bytes, supplied_bytes
            ),
            Self::TensorSizeOverflow(node) => {
                write!(formatter, "node {} tensor byte size overflowed", node.0)
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

fn env_u64(name: &str, fallback: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn allocation_pressure(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("memory") || error.contains("alloc") || error.contains("resource")
}

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
    temporary_buffers: Vec<BufferId>,
}

pub struct GpuRuntime<D, C> {
    driver: D,
    jit: TensorJit<C>,
    devices: Vec<DeviceInfo>,
    loaded: BTreeMap<(DeviceId, CacheKey), KernelId>,
    allocations: BTreeMap<BufferId, u64>,
    live_allocation_bytes: u64,
    max_live_allocation_bytes: u64,
    max_single_allocation_bytes: u64,
}

impl<D: GpuDriver, C: GpuCompiler> GpuRuntime<D, C> {
    pub fn new(driver: D, compiler: C, cache: KernelCache) -> Result<Self, RuntimeError> {
        let devices = driver.discover_devices().map_err(RuntimeError::Driver)?;
        let available = devices
            .iter()
            .map(|device| device.total_memory_bytes)
            .min()
            .unwrap_or(0);
        let reserved = 1u64 << 30;
        let default_live = available
            .saturating_sub(reserved)
            .min(available.saturating_mul(3) / 4);
        let max_live_allocation_bytes =
            env_u64("SEVERIAN_GPU_MAX_LIVE_BYTES", default_live.max(256 << 20));
        let max_single_allocation_bytes = env_u64(
            "SEVERIAN_GPU_MAX_SINGLE_ALLOCATION_BYTES",
            (max_live_allocation_bytes / 2).max(64 << 20),
        );
        Ok(Self {
            driver,
            jit: TensorJit::new(compiler, cache),
            devices,
            loaded: BTreeMap::new(),
            allocations: BTreeMap::new(),
            live_allocation_bytes: 0,
            max_live_allocation_bytes,
            max_single_allocation_bytes,
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
        if bytes > self.max_single_allocation_bytes {
            return Err(RuntimeError::GpuSingleAllocationLimit {
                requested: bytes,
                limit: self.max_single_allocation_bytes,
            });
        }
        let projected =
            self.live_allocation_bytes
                .checked_add(bytes)
                .ok_or(RuntimeError::GpuMemoryLimit {
                    requested: bytes,
                    live: self.live_allocation_bytes,
                    limit: self.max_live_allocation_bytes,
                })?;
        if projected > self.max_live_allocation_bytes {
            return Err(RuntimeError::GpuMemoryLimit {
                requested: bytes,
                live: self.live_allocation_bytes,
                limit: self.max_live_allocation_bytes,
            });
        }
        let retries = env_u64("SEVERIAN_GPU_ALLOCATION_RETRIES", 3);
        let initial_backoff_ms = env_u64("SEVERIAN_GPU_ALLOCATION_BACKOFF_MS", 1000);
        let mut attempt = 0;
        let buffer = loop {
            match self.driver.allocate(device, bytes, alignment) {
                Ok(buffer) => break buffer,
                Err(error) if allocation_pressure(&error) && attempt < retries => {
                    let delay = initial_backoff_ms.saturating_mul(1u64 << attempt.min(20));
                    std::thread::sleep(Duration::from_millis(delay));
                    attempt += 1;
                }
                Err(error) => return Err(RuntimeError::Driver(error)),
            }
        };
        self.allocations.insert(buffer, bytes);
        self.live_allocation_bytes = projected;
        Ok(buffer)
    }

    pub fn deallocate(&mut self, buffer: BufferId) -> Result<(), RuntimeError> {
        self.driver
            .deallocate(buffer)
            .map_err(RuntimeError::Driver)?;
        if let Some(bytes) = self.allocations.remove(&buffer) {
            self.live_allocation_bytes = self.live_allocation_bytes.saturating_sub(bytes);
        }
        Ok(())
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

    /// Executes a complete fusion plan from versioned StorageView metadata.
    /// Runtime rank/shape/stride facts are completed before compilation; each
    /// fusion region then flows through the normal compiler cache and driver
    /// launcher. Dtype and rank never participate in symbol identity.
    pub fn execute_storage_graph(
        &mut self,
        device: DeviceId,
        graph: &FusionGraph,
        plan: &FusionPlan,
        inputs: &[HostStorageInput<'_>],
        scalars: &[HostScalarInput],
        options: &CompilerOptions,
    ) -> Result<GraphExecution, RuntimeError> {
        let target = self.device(device)?.target;
        let bindings = inputs
            .iter()
            .map(|input| StorageSpecializationBinding {
                node: input.node,
                view: input.view.clone(),
            })
            .collect::<Vec<_>>();
        let specialization = specialize_storage_views(graph, target, &bindings)
            .map_err(RuntimeError::Specialization)?;
        let mut buffers = BTreeMap::new();
        for input in inputs {
            if buffers.contains_key(&input.node) {
                return Err(RuntimeError::DuplicateStorageInput(input.node));
            }
            if input.bytes.len() < input.view.byte_length as usize {
                return Err(RuntimeError::HostStorageLength {
                    node: input.node,
                    descriptor_bytes: input.view.byte_length,
                    supplied_bytes: input.bytes.len(),
                });
            }
            let buffer = self.allocate(device, input.view.byte_length, 256)?;
            self.upload(buffer, 0, &input.bytes[..input.view.byte_length as usize])?;
            buffers.insert(input.node, buffer);
        }

        let schedule = schedule_fusion_regions(graph, plan)?;
        let shape_by_node = specialization
            .shapes
            .iter()
            .map(|shape| (shape.node, shape.dimensions.as_slice()))
            .collect::<BTreeMap<_, _>>();
        let scalar_by_node = scalars
            .iter()
            .map(|scalar| (scalar.node, scalar))
            .collect::<BTreeMap<_, _>>();
        for scheduled in &schedule {
            let region = &plan.regions[scheduled.region.0 as usize];
            for output in &region.outputs {
                if buffers.contains_key(output) {
                    continue;
                }
                let node = graph.node(*output);
                let aliased = node.aliases.first().and_then(|alias| {
                    node.inputs
                        .get(usize::from(alias.input_index))
                        .and_then(|input| buffers.get(input))
                        .copied()
                });
                let buffer = if let Some(buffer) = aliased {
                    buffer
                } else {
                    let dimensions = shape_by_node
                        .get(output)
                        .ok_or(RuntimeError::MissingOutputShape(*output))?;
                    let elements = dimensions
                        .iter()
                        .try_fold(1u64, |size, dimension| size.checked_mul(*dimension));
                    let bytes = elements
                        .and_then(|elements| {
                            elements.checked_mul(u64::from(node.shape.element_bits.div_ceil(8)))
                        })
                        .ok_or(RuntimeError::TensorSizeOverflow(*output))?;
                    self.allocate(device, bytes, 256)?
                };
                buffers.insert(*output, buffer);
            }
        }

        let execution_ids = schedule
            .iter()
            .enumerate()
            .map(|(index, scheduled)| (scheduled.region, RegionExecutionId(index as u32)))
            .collect::<BTreeMap<_, _>>();
        let invocations = schedule
            .iter()
            .map(|scheduled| {
                let region = &plan.regions[scheduled.region.0 as usize];
                let arguments = region
                    .inputs
                    .iter()
                    .chain(&region.outputs)
                    .map(|node| {
                        if let Some(buffer) = buffers.get(node).copied() {
                            Ok(KernelArgument::Buffer {
                                buffer,
                                byte_offset: 0,
                            })
                        } else if region.inputs.contains(node) {
                            scalar_by_node
                                .get(node)
                                .map(|scalar| KernelArgument::Scalar {
                                    bytes: scalar.bytes.clone(),
                                    alignment: scalar.alignment,
                                })
                                .ok_or(RuntimeError::MissingNodeBuffer(*node))
                        } else {
                            Err(RuntimeError::MissingNodeBuffer(*node))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let dependencies = scheduled
                    .dependencies
                    .iter()
                    .map(|dependency| {
                        execution_ids
                            .get(dependency)
                            .copied()
                            .ok_or(RuntimeError::CyclicRegionDependencies)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(RegionInvocation {
                    id: execution_ids[&scheduled.region],
                    graph,
                    region,
                    specialization: &specialization,
                    options,
                    arguments,
                    dependencies,
                })
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        let execution = self.execute(device, &invocations)?;
        Ok(GraphExecution {
            specialization,
            buffers,
            execution,
        })
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
        let mut temporary_buffers = Vec::new();
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
            let request = KernelCompileRequest {
                graph: invocation.graph,
                region: invocation.region,
                specialization: invocation.specialization,
                options: invocation.options,
            };
            let resolved = self.jit.resolve(&request).map_err(|error| match error {
                TensorJitError::Compiler(error) => RuntimeError::Compiler(error),
                TensorJitError::Cache(error) => RuntimeError::Cache(error),
            })?;
            let key = resolved.key;
            let artifact = resolved.artifact;
            if resolved.cache_hit {
                cache_hits += 1;
            } else {
                cache_misses += 1;
            }
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
            let grid = calculate_grid(
                invocation.graph,
                invocation.specialization,
                &artifact.launch.grid,
            )?;
            let programs = grid
                .into_iter()
                .try_fold(1u64, |count, dimension| count.checked_mul(dimension))
                .ok_or(RuntimeError::TensorSizeOverflow(
                    invocation.region.outputs[0],
                ))?;
            let mut invocation_arguments = invocation.arguments.clone();
            for (bytes_per_program, alignment) in [
                (
                    artifact.launch.global_scratch_bytes_per_program,
                    artifact.launch.global_scratch_alignment,
                ),
                (
                    artifact.launch.profile_scratch_bytes_per_program,
                    artifact.launch.profile_scratch_alignment,
                ),
            ] {
                let bytes = bytes_per_program.checked_mul(programs).ok_or(
                    RuntimeError::TensorSizeOverflow(invocation.region.outputs[0]),
                )?;
                if bytes == 0 {
                    invocation_arguments.push(KernelArgument::scalar(0u64));
                } else {
                    let buffer = self.allocate(device, bytes, alignment.max(1))?;
                    temporary_buffers.push(buffer);
                    invocation_arguments.push(KernelArgument::Buffer {
                        buffer,
                        byte_offset: 0,
                    });
                }
            }
            let arguments = pack_arguments(&self.driver, &invocation_arguments)?;
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
            temporary_buffers,
        })
    }

    pub fn synchronize(&mut self, result: &ExecutionResult) -> Result<(), RuntimeError> {
        self.driver
            .wait(&result.events.values().copied().collect::<Vec<_>>())
            .map_err(RuntimeError::Driver)?;
        for buffer in &result.temporary_buffers {
            self.deallocate(*buffer)?;
        }
        Ok(())
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
