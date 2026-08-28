#![forbid(unsafe_code)]

mod fusion;

pub use fusion::{fusion_graph, fusion_graph_with_slots, FusionGraphError};

use severian_compile::{
    CompileContext, CompileError, CompileHandler, CompileRegion, CompileRegionSpecialization,
    CompiledRegionArtifact, GpuKernelBundle, GpuTarget,
};
use severian_fusion::{plan as plan_fusion, DeviceModel};
use severian_mlir::{
    structured::{
        AffineExpression, AffineMap, FunctionBuilder as StructuredFunctionBuilder, GenericBody,
        IteratorKind, ModuleBuilder as StructuredModuleBuilder, ScalarBinaryOperation,
        ScalarOperation, ScalarUnaryOperation, SliceComponent, Value as StructuredValue,
    },
    type_spelling, LoweredFloatFormat, LoweredTensorDimension, LoweredTensorElement,
    LoweredTensorShape, LoweredType, MlirArtifact,
};
use severian_universal::{
    tensor, AttrValue, Attrs, ExecutionPlacement, FloatFormat, IntegerWidth,
    PrimitiveRepresentation, TensorDimension, TensorShape, TypeContext, TypeId,
};
use std::collections::{BTreeMap, BTreeSet};

/// Generic compiler for structural `Tensor[T]` operations. Dtypes are lowered
/// exclusively through `PrimitiveRepresentation`; operation names never
/// select an f32/i32-specific implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorCompiler;

impl CompileHandler for TensorCompiler {
    fn compile(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
    ) -> Result<CompiledRegionArtifact, CompileError> {
        if region.placement == Some(ExecutionPlacement::Gpu) {
            return self
                .compile_gpu(region, context)
                .map(CompiledRegionArtifact::GpuKernel);
        }
        self.compile_cpu(region, context)
            .map(CompiledRegionArtifact::CpuMlir)
    }

    fn compile_specialized(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
        specialization: &CompileRegionSpecialization,
    ) -> Result<CompiledRegionArtifact, CompileError> {
        if region.placement == Some(ExecutionPlacement::Gpu) {
            let (region, types) = region
                .specialize_for_emission(context.types, specialization)
                .map_err(|error| {
                    invalid(format!("GPU tensor runtime specialization failed: {error}"))
                })?;
            return self
                .compile_gpu(
                    &region,
                    &CompileContext {
                        types: &types,
                        target: context.target,
                    },
                )
                .map(CompiledRegionArtifact::GpuKernel);
        }
        self.compile_specialized_cpu(region, context, specialization)
            .map(CompiledRegionArtifact::CpuMlir)
    }
}

impl TensorCompiler {
    /// Runtime/JIT entry point for a CPU region whose source contract may be
    /// unranked. The region is refined from explicit shape/stride metadata
    /// before the ordinary MLIR builder is entered.
    pub fn compile_specialized_cpu(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
        specialization: &CompileRegionSpecialization,
    ) -> Result<MlirArtifact, CompileError> {
        let (region, types) = region
            .specialize_for_emission(context.types, specialization)
            .map_err(|error| {
                invalid(format!("CPU tensor runtime specialization failed: {error}"))
            })?;
        self.compile_cpu(
            &region,
            &CompileContext {
                types: &types,
                target: context.target,
            },
        )
    }

    fn compile_cpu(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
    ) -> Result<MlirArtifact, CompileError> {
        if region.compile_operations.is_empty() {
            return Err(invalid("the tensor compiler requires a non-empty region"));
        }
        legalize_cpu_region_before_emission(region, context)?;
        if region_uses_structured_mlir(region) {
            return compile_structured_cpu_region(region, context);
        }
        let inputs = region
            .inputs
            .iter()
            .map(|value| lower_type(value.type_id, context))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = region
            .outputs
            .iter()
            .map(|value| lower_type(value.type_id, context))
            .collect::<Result<Vec<_>, _>>()?;
        let input_spellings = inputs
            .iter()
            .map(type_spelling)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| invalid(error.to_string()))?;
        let output_spellings = outputs
            .iter()
            .map(type_spelling)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| invalid(error.to_string()))?;
        let parameters = input_spellings
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("%arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let result_signature = match output_spellings.as_slice() {
            [] => String::new(),
            [output] => format!(" -> {output}"),
            outputs => format!(" -> ({})", outputs.join(", ")),
        };
        let mut declarations = BTreeSet::new();
        let mut helpers = String::new();
        let mut entry_body = String::new();
        let mut slot_types = inputs
            .iter()
            .cloned()
            .enumerate()
            .map(|(slot, ty)| (slot as u32, ty))
            .collect::<BTreeMap<_, _>>();
        let mut inferred_next_slot = inputs.len() as u32;
        let mut inferred_result_slots = Vec::new();

        for (index, operation) in region.compile_operations.iter().enumerate() {
            let operation_kind = tensor::TensorOp::decode(operation.id, &operation.attributes)
                .ok_or_else(|| {
                    invalid(format!(
                        "tensor operation {:?} has no structural operation kind",
                        operation.id
                    ))
                })?;
            let operation_inputs = operation
                .operands
                .iter()
                .map(|ty| lower_type(*ty, context))
                .collect::<Result<Vec<_>, _>>()?;
            let operation_outputs = operation
                .results
                .iter()
                .map(|ty| lower_type(*ty, context))
                .collect::<Result<Vec<_>, _>>()?;
            let operand_slots =
                if operation.operand_slots.is_empty() && region.compile_operations.len() == 1 {
                    (0..operation_inputs.len() as u32).collect::<Vec<_>>()
                } else {
                    operation.operand_slots.clone()
                };
            let result_slots =
                if operation.result_slots.is_empty() && region.compile_operations.len() == 1 {
                    let slots = (inferred_next_slot
                        ..inferred_next_slot + operation_outputs.len() as u32)
                        .collect::<Vec<_>>();
                    inferred_next_slot += operation_outputs.len() as u32;
                    slots
                } else {
                    operation.result_slots.clone()
                };
            if operand_slots.len() != operation_inputs.len()
                || result_slots.len() != operation_outputs.len()
            {
                return Err(invalid(
                    "tensor region value slots do not match operation arity",
                ));
            }
            for (slot, expected) in operand_slots.iter().zip(&operation_inputs) {
                if slot_types.get(slot) != Some(expected) {
                    return Err(invalid(format!(
                        "tensor region operand slot {slot} has the wrong type"
                    )));
                }
            }
            for (slot, ty) in result_slots.iter().zip(&operation_outputs) {
                if slot_types.insert(*slot, ty.clone()).is_some() {
                    return Err(invalid(format!(
                        "tensor region slot {slot} is defined twice"
                    )));
                }
            }
            inferred_result_slots = result_slots.clone();
            legalize_cpu_operation(operation_kind, &operation_inputs, &operation_outputs)?;
            let operation_input_spellings = operation_inputs
                .iter()
                .map(type_spelling)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| invalid(error.to_string()))?;
            let operation_output_spellings = operation_outputs
                .iter()
                .map(type_spelling)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| invalid(error.to_string()))?;
            for declaration in
                operation_declarations(operation_kind, &operation_inputs, &operation_outputs)?
                    .lines()
                    .filter(|line| !line.is_empty())
            {
                declarations.insert(declaration.to_owned());
            }
            let helper_parameters = operation_input_spellings
                .iter()
                .enumerate()
                .map(|(operand, ty)| format!("%arg{operand}: {ty}"))
                .collect::<Vec<_>>()
                .join(", ");
            let helper_result = match operation_output_spellings.as_slice() {
                [output] => format!(" -> {output}"),
                outputs => format!(" -> ({})", outputs.join(", ")),
            };
            let helper_body = lower_operation(
                operation_kind,
                &operation_inputs,
                &operation_outputs,
                &operation_input_spellings,
                &operation_output_spellings,
                &operation.attributes,
            )?;
            let helper_symbol = format!(
                "__sev_tensor_region_{}_op_{index}",
                region.id.index()
            );
            helpers.push_str(&format!(
                "  func.func private @{helper_symbol}({helper_parameters}){helper_result} {{\n{helper_body}  }}\n"
            ));
            let arguments = operand_slots
                .iter()
                .map(|slot| {
                    if (*slot as usize) < inputs.len() {
                        format!("%arg{slot}")
                    } else {
                        format!("%v{slot}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let result_names = result_slots
                .iter()
                .map(|slot| format!("%v{slot}"))
                .collect::<Vec<_>>()
                .join(", ");
            entry_body.push_str(&format!(
                "    {result_names} = func.call @{helper_symbol}({arguments}) : ({input_types}){helper_result}\n",
                input_types = operation_input_spellings.join(", "),
            ));
        }
        let output_slots = if region.output_slots.is_empty() {
            inferred_result_slots
        } else {
            region.output_slots.clone()
        };
        if output_slots.len() != outputs.len() {
            return Err(invalid(
                "tensor region output slots do not match its result signature",
            ));
        }
        let return_values = output_slots
            .iter()
            .map(|slot| {
                if (*slot as usize) < inputs.len() {
                    format!("%arg{slot}")
                } else {
                    format!("%v{slot}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        if outputs.is_empty() {
            entry_body.push_str("    return\n");
        } else {
            entry_body.push_str(&format!(
                "    return {return_values} : {}\n",
                output_spellings.join(", ")
            ));
        }
        let declarations = declarations
            .into_iter()
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        Ok(MlirArtifact {
            module: format!(
                "module {{\n{declarations}{helpers}  func.func @entry({parameters}){result_signature} {{\n{entry_body}  }}\n}}"
            ),
            inputs,
            outputs,
        })
    }

    fn compile_gpu(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
    ) -> Result<GpuKernelBundle, CompileError> {
        if region.compile_operations.is_empty() {
            return Err(invalid("the tensor compiler requires a non-empty region"));
        }
        let device = context.target.triton_gpu().ok_or_else(|| {
            CompileError::Target(
                "GPU placement requires an AMD or NVIDIA device in the target".into(),
            )
        })?;
        let target =
            if device.features.contains("vendor.amd") || device.architecture.starts_with("gfx") {
                GpuTarget::Amd
            } else if device.features.contains("vendor.nvidia")
                || device.architecture.starts_with("sm_")
                || device.architecture.starts_with("compute_")
            {
                GpuTarget::Nvidia
            } else {
                return Err(CompileError::Target(format!(
                    "GPU `{}` with architecture `{}` is not a Triton AMD/NVIDIA target",
                    device.name, device.architecture
                )));
            };
        let inputs = region
            .inputs
            .iter()
            .map(|value| lower_type(value.type_id, context))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = region
            .outputs
            .iter()
            .map(|value| lower_type(value.type_id, context))
            .collect::<Result<Vec<_>, _>>()?;
        let (graph, value_nodes) = fusion_graph_with_slots(region, context.types)
            .map_err(|error| CompileError::InvalidArtifact(error.to_string()))?;
        let plan = plan_fusion(&graph, DeviceModel::conservative_gpu());
        Ok(GpuKernelBundle {
            target,
            architecture: device.architecture.clone(),
            graph,
            plan,
            value_nodes,
            inputs,
            outputs,
        })
    }
}

fn legalize_cpu_region_before_emission(
    region: &CompileRegion,
    context: &CompileContext<'_>,
) -> Result<(), CompileError> {
    for operation in &region.compile_operations {
        let kind =
            tensor::TensorOp::decode(operation.id, &operation.attributes).ok_or_else(|| {
                invalid(format!(
                    "tensor operation {:?} has no structural operation kind",
                    operation.id
                ))
            })?;
        let inputs = operation
            .operands
            .iter()
            .map(|type_id| lower_type(*type_id, context))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = operation
            .results
            .iter()
            .map(|type_id| lower_type(*type_id, context))
            .collect::<Result<Vec<_>, _>>()?;
        legalize_cpu_operation(kind, &inputs, &outputs)?;
    }
    Ok(())
}

fn region_uses_structured_mlir(region: &CompileRegion) -> bool {
    region.compile_operations.iter().any(|operation| {
        matches!(
            tensor::TensorOp::decode(operation.id, &operation.attributes),
            Some(tensor::TensorOp::Elementwise(_)) | Some(tensor::TensorOp::Reduce(
                tensor::ReductionOp::Sum
                    | tensor::ReductionOp::SumAxis
                    | tensor::ReductionOp::MeanLast
                    | tensor::ReductionOp::MaxLast
            )) | Some(tensor::TensorOp::Slice)
                | Some(tensor::TensorOp::Concatenate)
                | Some(tensor::TensorOp::ReshapeView(_))
                | Some(tensor::TensorOp::Permute(_))
                | Some(tensor::TensorOp::Broadcast(_))
                | Some(tensor::TensorOp::Gather)
                | Some(tensor::TensorOp::Scatter)
        )
    })
}

fn compile_structured_cpu_region(
    region: &CompileRegion,
    context: &CompileContext<'_>,
) -> Result<MlirArtifact, CompileError> {
    let inputs = region
        .inputs
        .iter()
        .map(|value| lower_type(value.type_id, context))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = region
        .outputs
        .iter()
        .map(|value| lower_type(value.type_id, context))
        .collect::<Result<Vec<_>, _>>()?;
    let parameters = inputs
        .iter()
        .enumerate()
        .map(|(index, ty)| (format!("arg{index}"), ty.clone()))
        .collect();
    let mut function = StructuredFunctionBuilder::new("entry", false, parameters, outputs.clone())
        .map_err(structured_error)?;
    let mut slots = (0..inputs.len())
        .map(|index| {
            function
                .parameter(index)
                .map(|value| (index as u32, value))
                .map_err(structured_error)
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut inferred_next_slot = inputs.len() as u32;
    let mut inferred_results = Vec::new();

    for operation in &region.compile_operations {
        let kind =
            tensor::TensorOp::decode(operation.id, &operation.attributes).ok_or_else(|| {
                invalid(format!(
                    "tensor operation {:?} has no structural operation kind",
                    operation.id
                ))
            })?;
        let operation_inputs = operation
            .operands
            .iter()
            .map(|ty| lower_type(*ty, context))
            .collect::<Result<Vec<_>, _>>()?;
        let operation_outputs = operation
            .results
            .iter()
            .map(|ty| lower_type(*ty, context))
            .collect::<Result<Vec<_>, _>>()?;
        legalize_cpu_operation(kind, &operation_inputs, &operation_outputs)?;
        let operand_slots =
            if operation.operand_slots.is_empty() && region.compile_operations.len() == 1 {
                (0..operation_inputs.len() as u32).collect::<Vec<_>>()
            } else {
                operation.operand_slots.clone()
            };
        let result_slots = if operation.result_slots.is_empty()
            && region.compile_operations.len() == 1
        {
            let result = (inferred_next_slot..inferred_next_slot + operation_outputs.len() as u32)
                .collect::<Vec<_>>();
            inferred_next_slot += operation_outputs.len() as u32;
            result
        } else {
            operation.result_slots.clone()
        };
        if operand_slots.len() != operation_inputs.len()
            || result_slots.len() != operation_outputs.len()
        {
            return Err(invalid(
                "structured tensor region value slots do not match operation arity",
            ));
        }
        let operands = operand_slots
            .iter()
            .map(|slot| {
                slots.get(slot).cloned().ok_or_else(|| {
                    invalid(format!(
                        "structured tensor operand slot {slot} is undefined"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [result_slot] = result_slots.as_slice() else {
            return Err(invalid(
                "structured tensor operations currently require one result",
            ));
        };
        let [result_type] = operation_outputs.as_slice() else {
            return Err(invalid(
                "structured tensor operations currently require one result type",
            ));
        };
        let result = match kind {
            tensor::TensorOp::Elementwise(operation) => lower_structured_elementwise(
                &mut function,
                operation,
                &operands,
                result_type,
                *result_slot,
            )?,
            tensor::TensorOp::Reduce(reduction) => lower_structured_reduction(
                &mut function,
                reduction,
                &operands,
                result_type,
                &operation_inputs,
                &operation.attributes,
                *result_slot,
            )?,
            tensor::TensorOp::Slice => lower_structured_slice(
                &mut function,
                &operands,
                result_type,
                &operation.attributes,
                *result_slot,
            )?,
            tensor::TensorOp::Concatenate => lower_structured_concatenate(
                &mut function,
                &operands,
                result_type,
                &operation.attributes,
                *result_slot,
            )?,
            tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Reshape) => {
                lower_structured_reshape(
                    &mut function,
                    &operands,
                    result_type,
                    &operation.attributes,
                    *result_slot,
                )?
            }
            tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Materialize) => operands
                .first()
                .cloned()
                .ok_or_else(|| invalid("Materialize requires one tensor operand"))?,
            tensor::TensorOp::Permute(permutation) => lower_structured_permute(
                &mut function,
                permutation,
                &operands,
                result_type,
                &operation.attributes,
                *result_slot,
            )?,
            tensor::TensorOp::Broadcast(broadcast) => lower_structured_broadcast(
                &mut function,
                broadcast,
                &operands,
                result_type,
                &operation.attributes,
                *result_slot,
            )?,
            tensor::TensorOp::Gather => lower_structured_gather(
                &mut function,
                &operands,
                result_type,
                *result_slot,
            )?,
            tensor::TensorOp::Scatter => lower_structured_scatter(
                &mut function,
                &operands,
                result_type,
                *result_slot,
            )?,
            _ => {
                return Err(invalid(format!(
                    "structured MLIR capability is missing for {kind:?}"
                )))
            }
        };
        if slots.insert(*result_slot, result).is_some() {
            return Err(invalid(format!(
                "structured tensor slot {result_slot} is defined twice"
            )));
        }
        inferred_results = result_slots;
    }

    let output_slots = if region.output_slots.is_empty() {
        inferred_results
    } else {
        region.output_slots.clone()
    };
    let returned = output_slots
        .iter()
        .map(|slot| {
            slots.get(slot).cloned().ok_or_else(|| {
                invalid(format!("structured tensor output slot {slot} is undefined"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    function.return_values(returned).map_err(structured_error)?;
    let mut module = StructuredModuleBuilder::default();
    module
        .declare_function(
            "__sev_list_get_i64",
            vec![
                LoweredType::Bytes,
                LoweredType::Integer {
                    bits: 64,
                    signed: true,
                },
            ],
            vec![LoweredType::Integer {
                bits: 64,
                signed: true,
            }],
        )
        .map_err(structured_error)?;
    module
        .add_function(function.finish().map_err(structured_error)?)
        .map_err(structured_error)?;
    Ok(MlirArtifact {
        module: module.print().map_err(structured_error)?,
        inputs,
        outputs,
    })
}

fn lower_structured_elementwise(
    function: &mut StructuredFunctionBuilder,
    operation: tensor::ElementwiseOp,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let LoweredType::Tensor {
        element,
        shape: LoweredTensorShape::Ranked(output_shape),
    } = result_type
    else {
        return Err(invalid(
            "structured elementwise lowering requires a ranked tensor result",
        ));
    };
    let rank = output_shape.len();
    let input = operands
        .first()
        .ok_or_else(|| invalid("structured elementwise operation has no tensor operand"))?;
    let input_shape = ranked_dimensions(
        input
            .lowered_type()
            .ok_or_else(|| invalid("elementwise tensor operand is not lowered"))?,
    )
    .ok_or_else(|| invalid("elementwise tensor operand has unknown rank"))?;
    let mut dynamic_sizes = Vec::new();
    for (axis, dimension) in output_shape.iter().enumerate() {
        if dimension != &LoweredTensorDimension::Dynamic {
            continue;
        }
        let (source, source_axis) = if matches!(
            operation,
            tensor::ElementwiseOp::Add
                | tensor::ElementwiseOp::Subtract
                | tensor::ElementwiseOp::Multiply
                | tensor::ElementwiseOp::Divide
        ) {
            let right = operands
                .get(1)
                .ok_or_else(|| invalid("binary elementwise operation has no right operand"))?;
            let right_shape = ranked_dimensions(
                right
                    .lowered_type()
                    .ok_or_else(|| invalid("elementwise right operand is not lowered"))?,
            )
            .ok_or_else(|| invalid("elementwise right operand has unknown rank"))?;
            structured_broadcast_dimension_source(
                axis,
                rank,
                input,
                input_shape,
                right,
                right_shape,
            )?
        } else {
            (input, axis)
        };
        let axis_value = function
            .index_constant(format!("v{result_slot}_c{axis}"), source_axis)
            .map_err(structured_error)?;
        dynamic_sizes.push(
            function
                .tensor_dim(format!("v{result_slot}_d{axis}"), source, &axis_value)
                .map_err(structured_error)?,
        );
    }
    let empty = function
        .tensor_empty(
            format!("v{result_slot}_empty"),
            result_type.clone(),
            dynamic_sizes,
        )
        .map_err(structured_error)?;
    let scalar = lowered_scalar(*element);
    let mut tensor_inputs = vec![input.clone()];
    let mut maps = vec![structured_broadcast_map(input_shape, rank)?];
    let (arguments, captures, operations) = match operation {
        tensor::ElementwiseOp::Add
        | tensor::ElementwiseOp::Subtract
        | tensor::ElementwiseOp::Multiply
        | tensor::ElementwiseOp::Divide => {
            let [_, right] = operands else {
                return Err(invalid(
                    "structured binary elementwise lowering requires two operands",
                ));
            };
            let right_shape = ranked_dimensions(
                right
                    .lowered_type()
                    .ok_or_else(|| invalid("elementwise right operand is not lowered"))?,
            )
            .ok_or_else(|| invalid("elementwise right operand has unknown rank"))?;
            tensor_inputs.push(right.clone());
            maps.push(structured_broadcast_map(right_shape, rank)?);
            (
                vec![
                    ("lhs".into(), scalar.clone()),
                    ("rhs".into(), scalar.clone()),
                    ("unused".into(), scalar.clone()),
                ],
                Vec::new(),
                vec![
                    ScalarOperation::Binary {
                        result: "computed".into(),
                        operation: structural_binary_operation(operation, *element)?,
                        left: "lhs".into(),
                        right: "rhs".into(),
                        ty: scalar.clone(),
                    },
                    ScalarOperation::Yield {
                        value: "computed".into(),
                        ty: scalar.clone(),
                    },
                ],
            )
        }
        tensor::ElementwiseOp::Exp
        | tensor::ElementwiseOp::Log
        | tensor::ElementwiseOp::Tanh
        | tensor::ElementwiseOp::Rsqrt => {
            if !is_float(*element) || operands.len() != 1 {
                return Err(invalid(
                    "floating unary elementwise lowering requires one floating tensor",
                ));
            }
            let unary = match operation {
                tensor::ElementwiseOp::Exp => ScalarUnaryOperation::Exp,
                tensor::ElementwiseOp::Log => ScalarUnaryOperation::Log,
                tensor::ElementwiseOp::Tanh => ScalarUnaryOperation::Tanh,
                tensor::ElementwiseOp::Rsqrt => ScalarUnaryOperation::Rsqrt,
                _ => unreachable!(),
            };
            let (computed_type, operand_name, mut operations) = if matches!(
                element,
                LoweredTensorElement::Float {
                    format: LoweredFloatFormat::Float8E4M3Fn
                        | LoweredFloatFormat::Float8E5M2
                }
            ) {
                let wide = LoweredType::Float {
                    format: LoweredFloatFormat::Ieee(16),
                };
                (
                    wide.clone(),
                    "wide".to_owned(),
                    vec![ScalarOperation::FloatConvert {
                        result: "wide".into(),
                        operand: "value".into(),
                        source: scalar.clone(),
                        target: wide,
                    }],
                )
            } else {
                (scalar.clone(), "value".to_owned(), Vec::new())
            };
            operations.push(ScalarOperation::Unary {
                result: "computed".into(),
                operation: unary,
                operand: operand_name,
                ty: computed_type.clone(),
            });
            let yielded = if computed_type == scalar {
                "computed"
            } else {
                operations.push(ScalarOperation::FloatConvert {
                    result: "narrow".into(),
                    operand: "computed".into(),
                    source: computed_type,
                    target: scalar.clone(),
                });
                "narrow"
            };
            operations.push(ScalarOperation::Yield {
                value: yielded.into(),
                ty: scalar.clone(),
            });
            (
                vec![
                    ("value".into(), scalar.clone()),
                    ("unused".into(), scalar.clone()),
                ],
                Vec::new(),
                operations,
            )
        }
        tensor::ElementwiseOp::Relu => {
            if operands.len() != 1 {
                return Err(invalid("Relu requires one tensor operand"));
            }
            let zero = function
                .scalar_constant(
                    format!("v{result_slot}_zero"),
                    if is_float(*element) { "0.0" } else { "0" },
                    scalar.clone(),
                )
                .map_err(structured_error)?;
            let maximum = match element {
                LoweredTensorElement::Float { .. } => ScalarBinaryOperation::MaximumFloat,
                LoweredTensorElement::Integer { signed: true, .. } => {
                    ScalarBinaryOperation::MaximumSigned
                }
                LoweredTensorElement::Integer { signed: false, .. }
                | LoweredTensorElement::Boolean => ScalarBinaryOperation::MaximumUnsigned,
            };
            (
                vec![
                    ("value".into(), scalar.clone()),
                    ("unused".into(), scalar.clone()),
                ],
                vec![zero],
                vec![
                    ScalarOperation::Binary {
                        result: "computed".into(),
                        operation: maximum,
                        left: "value".into(),
                        right: format!("v{result_slot}_zero"),
                        ty: scalar.clone(),
                    },
                    ScalarOperation::Yield {
                        value: "computed".into(),
                        ty: scalar.clone(),
                    },
                ],
            )
        }
        tensor::ElementwiseOp::Scale | tensor::ElementwiseOp::AddScalar => {
            let [_, argument] = operands else {
                return Err(invalid(
                    "tensor-scalar elementwise lowering requires tensor and scalar operands",
                ));
            };
            if !is_float(*element) {
                return Err(invalid("tensor-scalar operations require floating tensors"));
            }
            let captured = if argument.lowered_type() == Some(&scalar) {
                argument.clone()
            } else {
                function
                    .float_convert(format!("v{result_slot}_scalar"), argument, scalar.clone())
                    .map_err(structured_error)?
            };
            let captured_name = captured.name().to_owned();
            (
                vec![
                    ("value".into(), scalar.clone()),
                    ("unused".into(), scalar.clone()),
                ],
                vec![captured],
                vec![
                    ScalarOperation::Binary {
                        result: "computed".into(),
                        operation: if operation == tensor::ElementwiseOp::Scale {
                            ScalarBinaryOperation::MultiplyFloat
                        } else {
                            ScalarBinaryOperation::AddFloat
                        },
                        left: "value".into(),
                        right: captured_name,
                        ty: scalar.clone(),
                    },
                    ScalarOperation::Yield {
                        value: "computed".into(),
                        ty: scalar.clone(),
                    },
                ],
            )
        }
    };
    let body = GenericBody::with_captures(arguments, captures, operations)
        .map_err(structured_error)?;
    maps.push(AffineMap::identity(rank));
    function
        .linalg_generic(
            format!("v{result_slot}"),
            tensor_inputs,
            empty,
            maps,
            vec![IteratorKind::Parallel; rank],
            body,
        )
        .map_err(structured_error)
}

fn lower_structured_reduction(
    function: &mut StructuredFunctionBuilder,
    operation: tensor::ReductionOp,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    input_types: &[LoweredType],
    attributes: &Attrs,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let [input] = operands else {
        return Err(invalid(
            "structured reduction lowering requires one tensor operand",
        ));
    };
    let [LoweredType::Tensor {
        element,
        shape: LoweredTensorShape::Ranked(input_shape),
    }] = input_types
    else {
        return Err(invalid(
            "structured reduction lowering requires a ranked tensor operand",
        ));
    };
    let LoweredType::Tensor {
        shape: LoweredTensorShape::Ranked(output_shape),
        ..
    } = result_type
    else {
        return Err(invalid(
            "structured reduction lowering requires a ranked tensor result",
        ));
    };
    let axes = structural_reduction_axes(operation, input_shape.len(), attributes)?;
    let retained_axes = (0..input_shape.len())
        .filter(|axis| !axes.contains(axis))
        .collect::<Vec<_>>();
    let keep_dimensions = matches!(
        operation,
        tensor::ReductionOp::MeanLast | tensor::ReductionOp::MaxLast
    ) && output_shape.len() == input_shape.len();
    let mut dynamic_sizes = Vec::new();
    for (output_axis, dimension) in output_shape.iter().enumerate() {
        if dimension != &LoweredTensorDimension::Dynamic {
            continue;
        }
        let source_axis = if keep_dimensions {
            output_axis
        } else {
            *retained_axes.get(output_axis).ok_or_else(|| {
                invalid("dynamic reduction result dimension has no retained input axis")
            })?
        };
        let axis = function
            .index_constant(format!("v{result_slot}_c{output_axis}"), source_axis)
            .map_err(structured_error)?;
        dynamic_sizes.push(
            function
                .tensor_dim(format!("v{result_slot}_d{output_axis}"), input, &axis)
                .map_err(structured_error)?,
        );
    }
    let empty = function
        .tensor_empty(
            format!("v{result_slot}_empty"),
            result_type.clone(),
            dynamic_sizes,
        )
        .map_err(structured_error)?;
    let scalar = lowered_scalar(*element);
    let mean_width = if operation == tensor::ReductionOp::MeanLast {
        if !is_float(*element) {
            return Err(invalid(
                "MeanLast requires a floating element representation",
            ));
        }
        let last_axis = input_shape.len() - 1;
        Some(match input_shape[last_axis] {
            LoweredTensorDimension::Known(width) => function
                .scalar_constant(
                    format!("v{result_slot}_width"),
                    format!("{width}.0"),
                    scalar.clone(),
                )
                .map_err(structured_error)?,
            LoweredTensorDimension::Dynamic => {
                let axis = function
                    .index_constant(format!("v{result_slot}_width_axis"), last_axis)
                    .map_err(structured_error)?;
                let width_index = function
                    .tensor_dim(format!("v{result_slot}_width_index"), input, &axis)
                    .map_err(structured_error)?;
                let width_i64 = function
                    .index_cast(
                        format!("v{result_slot}_width_i64"),
                        &width_index,
                        LoweredType::Integer {
                            bits: 64,
                            signed: false,
                        },
                    )
                    .map_err(structured_error)?;
                function
                    .unsigned_to_float(format!("v{result_slot}_width"), &width_i64, scalar.clone())
                    .map_err(structured_error)?
            }
        })
    } else {
        None
    };
    let (identity, combine) = structural_reduction_combiner(operation, *element)?;
    let initial = function
        .scalar_constant(format!("v{result_slot}_identity"), identity, scalar.clone())
        .map_err(structured_error)?;
    let initialized = function
        .linalg_fill(format!("v{result_slot}_initialized"), &initial, &empty)
        .map_err(structured_error)?;
    let input_map = AffineMap::identity(input_shape.len());
    let output_map = if keep_dimensions {
        AffineMap::new(
            input_shape.len(),
            (0..input_shape.len())
                .map(|axis| {
                    if axes.contains(&axis) {
                        AffineExpression::Constant(0)
                    } else {
                        AffineExpression::Dimension(axis)
                    }
                })
                .collect(),
        )
        .map_err(structured_error)?
    } else if retained_axes.len() == output_shape.len() {
        AffineMap::new(
            input_shape.len(),
            retained_axes
                .iter()
                .map(|axis| AffineExpression::Dimension(*axis))
                .collect(),
        )
        .map_err(structured_error)?
    } else if retained_axes.is_empty()
        && output_shape
            .iter()
            .all(|dimension| dimension == &LoweredTensorDimension::Known(1))
    {
        AffineMap::new(
            input_shape.len(),
            vec![AffineExpression::Constant(0); output_shape.len()],
        )
        .map_err(structured_error)?
    } else {
        return Err(invalid(
            "structured reduction result rank does not match its retained axes",
        ));
    };
    let mut scalar_operations = Vec::new();
    let combined_right = if let Some(width) = &mean_width {
        scalar_operations.push(ScalarOperation::Binary {
            result: "finalized".into(),
            operation: ScalarBinaryOperation::DivideFloat,
            left: "value".into(),
            right: width.name().into(),
            ty: scalar.clone(),
        });
        "finalized"
    } else {
        "value"
    };
    scalar_operations.push(ScalarOperation::Binary {
        result: "combined".into(),
        operation: combine,
        left: "accumulator".into(),
        right: combined_right.into(),
        ty: scalar.clone(),
    });
    scalar_operations.push(ScalarOperation::Yield {
        value: "combined".into(),
        ty: scalar.clone(),
    });
    let body = GenericBody::with_captures(
        vec![
            ("value".into(), scalar.clone()),
            ("accumulator".into(), scalar.clone()),
        ],
        mean_width.into_iter().collect(),
        scalar_operations,
    )
    .map_err(structured_error)?;
    function
        .linalg_generic(
            format!("v{result_slot}"),
            vec![input.clone()],
            initialized,
            vec![input_map, output_map],
            (0..input_shape.len())
                .map(|axis| {
                    if axes.contains(&axis) {
                        IteratorKind::Reduction
                    } else {
                        IteratorKind::Parallel
                    }
                })
                .collect(),
            body,
        )
        .map_err(structured_error)
}

fn lower_structured_scatter(
    function: &mut StructuredFunctionBuilder,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let [source, indices, updates] = operands else {
        return Err(invalid("Scatter requires source, indices, and updates tensors"));
    };
    let source_shape = ranked_dimensions(
        source
            .lowered_type()
            .ok_or_else(|| invalid("Scatter source is not lowered"))?,
    )
    .ok_or_else(|| invalid("Scatter source rank must be known"))?;
    let result_shape = ranked_dimensions(result_type)
        .ok_or_else(|| invalid("Scatter result rank must be known"))?;
    let mut dynamic_sizes = Vec::new();
    for (axis, dimension) in result_shape.iter().enumerate() {
        if dimension == &LoweredTensorDimension::Dynamic {
            let axis_value = function
                .index_constant(format!("v{result_slot}_scatter_axis_{axis}"), axis)
                .map_err(structured_error)?;
            dynamic_sizes.push(
                function
                    .tensor_dim(
                        format!("v{result_slot}_scatter_dim_{axis}"),
                        source,
                        &axis_value,
                    )
                    .map_err(structured_error)?,
            );
        }
    }
    if source_shape.len() != result_shape.len() {
        return Err(invalid("Scatter must preserve source rank"));
    }
    function
        .tensor_scatter(
            format!("v{result_slot}_scatter"),
            source,
            indices,
            updates,
            dynamic_sizes,
            result_type.clone(),
        )
        .map_err(structured_error)
}

fn lower_structured_gather(
    function: &mut StructuredFunctionBuilder,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let [source, indices] = operands else {
        return Err(invalid("Gather requires source and indices tensors"));
    };
    let source_shape = ranked_dimensions(
        source
            .lowered_type()
            .ok_or_else(|| invalid("Gather source is not lowered"))?,
    )
    .ok_or_else(|| invalid("Gather source rank must be known"))?;
    let index_shape = ranked_dimensions(
        indices
            .lowered_type()
            .ok_or_else(|| invalid("Gather indices are not lowered"))?,
    )
    .ok_or_else(|| invalid("Gather index rank must be known"))?;
    let output_shape = ranked_dimensions(result_type)
        .ok_or_else(|| invalid("Gather result rank must be known"))?;
    if source_shape.is_empty() || output_shape.len() != index_shape.len() + source_shape.len() - 1 {
        return Err(invalid(
            "Gather result rank must be index rank plus source rank minus one",
        ));
    }
    let mut dynamic_sizes = Vec::new();
    for (output_axis, dimension) in output_shape.iter().enumerate() {
        if dimension != &LoweredTensorDimension::Dynamic {
            continue;
        }
        let (tensor, source_axis) = if output_axis < index_shape.len() {
            (indices, output_axis)
        } else {
            (source, output_axis - index_shape.len() + 1)
        };
        let axis = function
            .index_constant(format!("v{result_slot}_gather_axis_{output_axis}"), source_axis)
            .map_err(structured_error)?;
        dynamic_sizes.push(
            function
                .tensor_dim(
                    format!("v{result_slot}_gather_dim_{output_axis}"),
                    tensor,
                    &axis,
                )
                .map_err(structured_error)?,
        );
    }
    function
        .tensor_gather(
            format!("v{result_slot}_gather"),
            source,
            indices,
            dynamic_sizes,
            result_type.clone(),
        )
        .map_err(structured_error)
}

fn lower_structured_broadcast(
    function: &mut StructuredFunctionBuilder,
    operation: tensor::BroadcastOp,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    attributes: &Attrs,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let input = operands
        .first()
        .ok_or_else(|| invalid("Broadcast requires one tensor operand"))?;
    let input_shape = ranked_dimensions(
        input
            .lowered_type()
            .ok_or_else(|| invalid("Broadcast input is not lowered"))?,
    )
    .ok_or_else(|| invalid("Broadcast input rank must be known"))?;
    let output_shape = ranked_dimensions(result_type)
        .ok_or_else(|| invalid("Broadcast result rank must be known"))?;
    let output_rank = output_shape.len();
    let (input_map, dynamic_sizes) = match operation {
        tensor::BroadcastOp::Like => {
            let [_, shape_donor] = operands else {
                return Err(invalid("BroadcastLike requires value and shape-donor tensors"));
            };
            let donor_shape = ranked_dimensions(
                shape_donor
                    .lowered_type()
                    .ok_or_else(|| invalid("BroadcastLike donor is not lowered"))?,
            )
            .ok_or_else(|| invalid("BroadcastLike donor rank must be known"))?;
            if donor_shape.len() != output_rank {
                return Err(invalid("BroadcastLike result rank must match its shape donor"));
            }
            let mut dynamic_sizes = Vec::new();
            for (axis, dimension) in output_shape.iter().enumerate() {
                if dimension == &LoweredTensorDimension::Dynamic {
                    let axis_value = function
                        .index_constant(format!("v{result_slot}_broadcast_axis_{axis}"), axis)
                        .map_err(structured_error)?;
                    dynamic_sizes.push(
                        function
                            .tensor_dim(
                                format!("v{result_slot}_broadcast_dim_{axis}"),
                                shape_donor,
                                &axis_value,
                            )
                            .map_err(structured_error)?,
                    );
                }
            }
            (
                structured_broadcast_map(input_shape, output_rank)?,
                dynamic_sizes,
            )
        }
        tensor::BroadcastOp::Repeat => {
            let [_, _specification] = operands else {
                return Err(invalid("Repeat requires one tensor and one axis/count operand"));
            };
            if input_shape.len() != output_rank {
                return Err(invalid("Repeat must preserve tensor rank"));
            }
            let runtime = decode_runtime_operands(attributes)?;
            let [axis, count] = runtime.get(&1).map(Vec::as_slice).unwrap_or_default() else {
                return Err(invalid(
                    "Repeat axis and count must be known before CPU MLIR emission",
                ));
            };
            let rank_i128 = i128::try_from(output_rank)
                .map_err(|_| invalid("tensor rank is outside i128"))?;
            let axis = if *axis < 0 { rank_i128 + axis } else { *axis };
            let axis = usize::try_from(axis)
                .map_err(|_| invalid("Repeat axis is outside the tensor rank"))?;
            let count = u64::try_from(*count)
                .ok()
                .filter(|count| *count > 0)
                .ok_or_else(|| invalid("Repeat count must be positive"))?;
            if axis >= output_rank {
                return Err(invalid("Repeat axis is outside the tensor rank"));
            }
            let mut dynamic_sizes = Vec::new();
            for (output_axis, dimension) in output_shape.iter().enumerate() {
                if dimension != &LoweredTensorDimension::Dynamic {
                    continue;
                }
                let axis_value = function
                    .index_constant(
                        format!("v{result_slot}_repeat_axis_{output_axis}"),
                        output_axis,
                    )
                    .map_err(structured_error)?;
                let input_dimension = function
                    .tensor_dim(
                        format!("v{result_slot}_repeat_input_dim_{output_axis}"),
                        input,
                        &axis_value,
                    )
                    .map_err(structured_error)?;
                dynamic_sizes.push(if output_axis == axis {
                    let count_value = function
                        .index_constant(
                            format!("v{result_slot}_repeat_count_{output_axis}"),
                            usize::try_from(count)
                                .map_err(|_| invalid("Repeat count is outside usize"))?,
                        )
                        .map_err(structured_error)?;
                    function
                        .index_multiply(
                            format!("v{result_slot}_repeat_dim_{output_axis}"),
                            &input_dimension,
                            &count_value,
                        )
                        .map_err(structured_error)?
                } else {
                    input_dimension
                });
            }
            (
                AffineMap::new(
                    output_rank,
                    (0..output_rank)
                        .map(|dimension| {
                            if dimension == axis {
                                AffineExpression::FloorDiv {
                                    dimension,
                                    divisor: count,
                                }
                            } else {
                                AffineExpression::Dimension(dimension)
                            }
                        })
                        .collect(),
                )
                .map_err(structured_error)?,
                dynamic_sizes,
            )
        }
    };
    let empty = function
        .tensor_empty(
            format!("v{result_slot}_broadcast_empty"),
            result_type.clone(),
            dynamic_sizes,
        )
        .map_err(structured_error)?;
    let scalar = lowered_scalar(tensor_element(result_type)?);
    let body = GenericBody::new(
        vec![
            ("element".into(), scalar.clone()),
            ("unused".into(), scalar.clone()),
        ],
        vec![ScalarOperation::Yield {
            value: "element".into(),
            ty: scalar,
        }],
    )
    .map_err(structured_error)?;
    function
        .linalg_generic(
            format!("v{result_slot}_broadcast"),
            vec![input.clone()],
            empty,
            vec![input_map, AffineMap::identity(output_rank)],
            vec![IteratorKind::Parallel; output_rank],
            body,
        )
        .map_err(structured_error)
}

fn lower_structured_permute(
    function: &mut StructuredFunctionBuilder,
    operation: tensor::PermuteOp,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    attributes: &Attrs,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let source = operands
        .first()
        .ok_or_else(|| invalid("Permute requires one tensor operand"))?;
    let source_shape = ranked_dimensions(
        source
            .lowered_type()
            .ok_or_else(|| invalid("Permute source is not lowered"))?,
    )
    .ok_or_else(|| invalid("Permute source rank must be known"))?;
    let result_shape = ranked_dimensions(result_type)
        .ok_or_else(|| invalid("Permute result rank must be known"))?;
    let rank = source_shape.len();
    if result_shape.len() != rank {
        return Err(invalid("Permute must preserve tensor rank"));
    }
    let output_to_input = match operation {
        tensor::PermuteOp::Reverse => (0..rank).rev().collect::<Vec<_>>(),
        tensor::PermuteOp::Axes => {
            let runtime = decode_runtime_operands(attributes)?;
            let axes = runtime.get(&1).ok_or_else(|| {
                invalid("Permute axis identities must be known before CPU MLIR emission")
            })?;
            axes.iter()
                .map(|axis| {
                    usize::try_from(*axis)
                        .map_err(|_| invalid("Permute axis must be a non-negative usize"))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    if output_to_input.len() != rank
        || output_to_input.iter().any(|axis| *axis >= rank)
        || output_to_input.iter().copied().collect::<BTreeSet<_>>().len() != rank
    {
        return Err(invalid("Permute axes must be a permutation of the known rank"));
    }
    let mut dynamic_sizes = Vec::new();
    for (output_axis, dimension) in result_shape.iter().enumerate() {
        if dimension != &LoweredTensorDimension::Dynamic {
            continue;
        }
        let source_axis = output_to_input[output_axis];
        let axis = function
            .index_constant(format!("v{result_slot}_permute_axis_{output_axis}"), source_axis)
            .map_err(structured_error)?;
        dynamic_sizes.push(
            function
                .tensor_dim(
                    format!("v{result_slot}_permute_dim_{output_axis}"),
                    source,
                    &axis,
                )
                .map_err(structured_error)?,
        );
    }
    let empty = function
        .tensor_empty(
            format!("v{result_slot}_permute_empty"),
            result_type.clone(),
            dynamic_sizes,
        )
        .map_err(structured_error)?;
    let mut input_to_output = vec![0usize; rank];
    for (output_axis, input_axis) in output_to_input.iter().copied().enumerate() {
        input_to_output[input_axis] = output_axis;
    }
    let input_map = AffineMap::new(
        rank,
        input_to_output
            .into_iter()
            .map(AffineExpression::Dimension)
            .collect(),
    )
    .map_err(structured_error)?;
    let scalar = lowered_scalar(tensor_element(result_type)?);
    let body = GenericBody::new(
        vec![
            ("element".into(), scalar.clone()),
            ("unused".into(), scalar.clone()),
        ],
        vec![ScalarOperation::Yield {
            value: "element".into(),
            ty: scalar,
        }],
    )
    .map_err(structured_error)?;
    function
        .linalg_generic(
            format!("v{result_slot}_permute"),
            vec![source.clone()],
            empty,
            vec![input_map, AffineMap::identity(rank)],
            vec![IteratorKind::Parallel; rank],
            body,
        )
        .map_err(structured_error)
}

fn lower_structured_reshape(
    function: &mut StructuredFunctionBuilder,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    attributes: &Attrs,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let [source, shape_list] = operands else {
        return Err(invalid("Reshape requires one tensor and one shape operand"));
    };
    let result_rank = ranked_dimensions(result_type)
        .ok_or_else(|| invalid("Reshape result rank must be known before MLIR emission"))?
        .len();
    let runtime = decode_runtime_operands(attributes)?;
    let dimensions = if let Some(values) = runtime.get(&1) {
        if values.len() != result_rank {
            return Err(invalid("Reshape shape length must equal its result rank"));
        }
        values
            .iter()
            .enumerate()
            .map(|(axis, value)| {
                let value = usize::try_from(*value).map_err(|_| {
                    invalid("structured Reshape currently requires non-negative dimensions")
                })?;
                function
                    .index_constant(format!("v{result_slot}_reshape_dim_{axis}"), value)
                    .map_err(structured_error)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        if !matches!(
            shape_list.lowered_type(),
            Some(LoweredType::Bytes | LoweredType::String)
        ) {
            return Err(invalid(
                "dynamic Reshape shape must use the opaque runtime list ABI",
            ));
        }
        let i64_type = LoweredType::Integer {
            bits: 64,
            signed: true,
        };
        let mut dimensions = Vec::with_capacity(result_rank);
        for axis in 0..result_rank {
            let axis_value = function
                .scalar_constant(
                    format!("v{result_slot}_reshape_axis_{axis}"),
                    axis.to_string(),
                    i64_type.clone(),
                )
                .map_err(structured_error)?;
            let [dimension] = function
                .call(
                    vec![format!("v{result_slot}_reshape_i64_{axis}")],
                    "__sev_list_get_i64",
                    vec![shape_list.clone(), axis_value],
                    vec![i64_type.clone()],
                )
                .map_err(structured_error)?
                .try_into()
                .map_err(|_| invalid("shape list index call returned the wrong arity"))?;
            dimensions.push(
                function
                    .integer_to_index(
                        format!("v{result_slot}_reshape_dim_{axis}"),
                        &dimension,
                    )
                    .map_err(structured_error)?,
            );
        }
        dimensions
    };
    let shape = function
        .shape_tensor_from_elements(format!("v{result_slot}_shape"), dimensions)
        .map_err(structured_error)?;
    function
        .tensor_reshape(
            format!("v{result_slot}_reshape"),
            source,
            &shape,
            result_type.clone(),
        )
        .map_err(structured_error)
}

fn lower_structured_slice(
    function: &mut StructuredFunctionBuilder,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    attributes: &Attrs,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let source = operands
        .first()
        .ok_or_else(|| invalid("structured Slice requires a tensor operand"))?;
    let rank = ranked_dimensions(
        source
            .lowered_type()
            .ok_or_else(|| invalid("structured Slice operand is not lowered"))?,
    )
    .ok_or_else(|| invalid("structured Slice operand has unknown rank"))?
    .len();
    let runtime = decode_runtime_operands(attributes)?;
    let (starts, sizes, strides) = if let (Some(starts), Some(ends), Some(strides)) =
        (runtime.get(&1), runtime.get(&2), runtime.get(&3))
    {
        if starts.len() != rank || ends.len() != rank || strides.len() != rank {
            return Err(invalid("Slice starts, ends, and steps must match tensor rank"));
        }
        let starts = starts
            .iter()
            .copied()
            .map(|value| i64::try_from(value).map_err(|_| invalid("Slice start is outside i64")))
            .collect::<Result<Vec<_>, _>>()?;
        let ends = ends
            .iter()
            .copied()
            .map(|value| i64::try_from(value).map_err(|_| invalid("Slice end is outside i64")))
            .collect::<Result<Vec<_>, _>>()?;
        let strides = strides
            .iter()
            .copied()
            .map(|value| i64::try_from(value).map_err(|_| invalid("Slice step is outside i64")))
            .collect::<Result<Vec<_>, _>>()?;
        let sizes = starts
            .iter()
            .zip(&ends)
            .zip(&strides)
            .map(|((start, end), stride)| {
                if *start < 0 || *end < *start || *stride <= 0 {
                    return Err(invalid(
                        "CPU tensor.extract_slice currently requires non-negative ordered bounds and positive steps",
                    ));
                }
                let extent = end - start;
                Ok((extent + stride - 1) / stride)
            })
            .collect::<Result<Vec<_>, _>>()?;
        (
            starts.into_iter().map(SliceComponent::Static).collect(),
            sizes.into_iter().map(SliceComponent::Static).collect(),
            strides.into_iter().map(SliceComponent::Static).collect(),
        )
    } else {
        let [_, starts_list, ends_list, strides_list] = operands else {
            return Err(invalid(
                "dynamic Slice requires source, starts, ends, and steps operands",
            ));
        };
        for list in [starts_list, ends_list, strides_list] {
            if !matches!(
                list.lowered_type(),
                Some(LoweredType::Bytes | LoweredType::String)
            ) {
                return Err(invalid(
                    format!(
                        "dynamic Slice shape operands must use the opaque runtime list ABI, got {:?}",
                        list.lowered_type()
                    ),
                ));
            }
        }
        let i64_type = LoweredType::Integer {
            bits: 64,
            signed: true,
        };
        let mut starts = Vec::with_capacity(rank);
        let mut sizes = Vec::with_capacity(rank);
        let mut strides = Vec::with_capacity(rank);
        for axis in 0..rank {
            let axis_value = function
                .scalar_constant(format!("v{result_slot}_axis_{axis}"), axis.to_string(), i64_type.clone())
                .map_err(structured_error)?;
            let load = |function: &mut StructuredFunctionBuilder,
                        name: String,
                        list: &StructuredValue|
             -> Result<StructuredValue, CompileError> {
                let [value] = function
                    .call(
                        vec![name],
                        "__sev_list_get_i64",
                        vec![list.clone(), axis_value.clone()],
                        vec![i64_type.clone()],
                    )
                    .map_err(structured_error)?
                    .try_into()
                    .map_err(|_| invalid("list index call returned the wrong arity"))?;
                Ok(value)
            };
            let start_i64 = load(function, format!("v{result_slot}_start_i64_{axis}"), starts_list)?;
            let end_i64 = load(function, format!("v{result_slot}_end_i64_{axis}"), ends_list)?;
            let stride_i64 = load(
                function,
                format!("v{result_slot}_stride_i64_{axis}"),
                strides_list,
            )?;
            let start = function
                .integer_to_index(format!("v{result_slot}_start_{axis}"), &start_i64)
                .map_err(structured_error)?;
            let end = function
                .integer_to_index(format!("v{result_slot}_end_{axis}"), &end_i64)
                .map_err(structured_error)?;
            let stride = function
                .integer_to_index(format!("v{result_slot}_stride_{axis}"), &stride_i64)
                .map_err(structured_error)?;
            let extent = function
                .index_subtract(format!("v{result_slot}_extent_{axis}"), &end, &start)
                .map_err(structured_error)?;
            let size = function
                .index_ceil_divide_signed(
                    format!("v{result_slot}_size_{axis}"),
                    &extent,
                    &stride,
                )
                .map_err(structured_error)?;
            starts.push(SliceComponent::Dynamic(start));
            sizes.push(SliceComponent::Dynamic(size));
            strides.push(SliceComponent::Dynamic(stride));
        }
        (starts, sizes, strides)
    };
    let slice_type = match result_type {
        LoweredType::Tensor { element, .. }
            if sizes
                .iter()
                .all(|size| matches!(size, SliceComponent::Static(_))) =>
        {
            LoweredType::Tensor {
                element: *element,
                shape: LoweredTensorShape::Ranked(
                    sizes
                        .iter()
                        .map(|size| match size {
                            SliceComponent::Static(value) => u64::try_from(*value)
                                .map(LoweredTensorDimension::Known)
                                .map_err(|_| invalid("Slice size is outside u64")),
                            SliceComponent::Dynamic(_) => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            }
        }
        _ => result_type.clone(),
    };
    let sliced = function
        .tensor_extract_slice(
            format!("v{result_slot}_slice"),
            source,
            slice_type.clone(),
            starts,
            sizes,
            strides,
        )
        .map_err(structured_error)?;
    if &slice_type == result_type {
        Ok(sliced)
    } else {
        function
            .tensor_cast(
                format!("v{result_slot}_slice_contract"),
                &sliced,
                result_type.clone(),
            )
            .map_err(structured_error)
    }
}

fn lower_structured_concatenate(
    function: &mut StructuredFunctionBuilder,
    operands: &[StructuredValue],
    result_type: &LoweredType,
    attributes: &Attrs,
    result_slot: u32,
) -> Result<StructuredValue, CompileError> {
    let [left, right, _axis_operand] = operands else {
        return Err(invalid(
            "Concatenate requires two tensors and one axis-list operand",
        ));
    };
    let rank = ranked_dimensions(
        left.lowered_type()
            .ok_or_else(|| invalid("Concatenate left operand is not lowered"))?,
    )
    .ok_or_else(|| invalid("Concatenate left operand has unknown rank"))?
    .len();
    let runtime = decode_runtime_operands(attributes)?;
    let [axis] = runtime.get(&2).map(Vec::as_slice).unwrap_or_default() else {
        return Err(invalid(
            "Concatenate axis identity must be known before CPU MLIR emission",
        ));
    };
    let rank_i128 = i128::try_from(rank).map_err(|_| invalid("tensor rank is outside i128"))?;
    let axis = if *axis < 0 { rank_i128 + axis } else { *axis };
    let axis = usize::try_from(axis)
        .map_err(|_| invalid("Concatenate axis is outside the tensor rank"))?;
    if axis >= rank {
        return Err(invalid("Concatenate axis is outside the tensor rank"));
    }
    let left_shape = ranked_dimensions(
        left.lowered_type()
            .ok_or_else(|| invalid("Concatenate left operand is not lowered"))?,
    )
    .expect("Concatenate rank was established above");
    let right_shape = ranked_dimensions(
        right
            .lowered_type()
            .ok_or_else(|| invalid("Concatenate right operand is not lowered"))?,
    )
    .ok_or_else(|| invalid("Concatenate right operand has unknown rank"))?;
    let LoweredType::Tensor {
        element,
        shape: LoweredTensorShape::Ranked(result_shape),
    } = result_type
    else {
        return Err(invalid("Concatenate result must be a ranked tensor"));
    };
    let concat_type = LoweredType::Tensor {
        element: *element,
        shape: LoweredTensorShape::Ranked(
            result_shape
                .iter()
                .enumerate()
                .map(|(dimension, result)| {
                    if left_shape[dimension] == LoweredTensorDimension::Dynamic
                        || right_shape[dimension] == LoweredTensorDimension::Dynamic
                    {
                        LoweredTensorDimension::Dynamic
                    } else {
                        result.clone()
                    }
                })
                .collect(),
        ),
    };
    let concatenated = function
        .tensor_concat(
            format!("v{result_slot}_concat"),
            vec![left.clone(), right.clone()],
            axis,
            concat_type.clone(),
        )
        .map_err(structured_error)?;
    if &concat_type == result_type {
        Ok(concatenated)
    } else {
        function
            .tensor_cast(
                format!("v{result_slot}_concat_contract"),
                &concatenated,
                result_type.clone(),
            )
            .map_err(structured_error)
    }
}

fn decode_runtime_operands(attributes: &Attrs) -> Result<BTreeMap<usize, Vec<i128>>, CompileError> {
    let Some(AttrValue::Integers(encoded)) = attributes.get(&tensor::RUNTIME_OPERANDS) else {
        return Ok(BTreeMap::new());
    };
    let mut decoded = BTreeMap::new();
    let mut cursor = 0usize;
    while cursor < encoded.len() {
        let operand = usize::try_from(encoded[cursor])
            .map_err(|_| invalid("runtime operand index is outside usize"))?;
        let count = encoded
            .get(cursor + 1)
            .copied()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| invalid("runtime operand encoding has an invalid value count"))?;
        let start = cursor + 2;
        let end = start
            .checked_add(count)
            .filter(|end| *end <= encoded.len())
            .ok_or_else(|| invalid("runtime operand encoding is truncated"))?;
        if decoded.insert(operand, encoded[start..end].to_vec()).is_some() {
            return Err(invalid("runtime operand encoding defines one operand twice"));
        }
        cursor = end;
    }
    Ok(decoded)
}

fn structured_broadcast_dimension_source<'a>(
    output_axis: usize,
    output_rank: usize,
    left: &'a StructuredValue,
    left_shape: &[LoweredTensorDimension],
    right: &'a StructuredValue,
    right_shape: &[LoweredTensorDimension],
) -> Result<(&'a StructuredValue, usize), CompileError> {
    let left_axis = output_axis.checked_sub(output_rank - left_shape.len());
    let right_axis = output_axis.checked_sub(output_rank - right_shape.len());
    for (value, axis, shape) in [
        (left, left_axis, left_shape),
        (right, right_axis, right_shape),
    ] {
        if let Some(axis) = axis {
            if shape[axis] != LoweredTensorDimension::Known(1) {
                return Ok((value, axis));
            }
        }
    }
    left_axis
        .map(|axis| (left, axis))
        .or_else(|| right_axis.map(|axis| (right, axis)))
        .ok_or_else(|| invalid("dynamic broadcast dimension has no runtime source"))
}

fn structured_broadcast_map(
    source: &[LoweredTensorDimension],
    output_rank: usize,
) -> Result<AffineMap, CompileError> {
    let offset = output_rank - source.len();
    AffineMap::new(
        output_rank,
        source
            .iter()
            .enumerate()
            .map(|(axis, dimension)| {
                if dimension == &LoweredTensorDimension::Known(1) {
                    AffineExpression::Constant(0)
                } else {
                    AffineExpression::Dimension(offset + axis)
                }
            })
            .collect(),
    )
    .map_err(structured_error)
}

fn structural_binary_operation(
    operation: tensor::ElementwiseOp,
    element: LoweredTensorElement,
) -> Result<ScalarBinaryOperation, CompileError> {
    use ScalarBinaryOperation as Scalar;
    Ok(match (operation, element) {
        (tensor::ElementwiseOp::Add, LoweredTensorElement::Float { .. }) => Scalar::AddFloat,
        (tensor::ElementwiseOp::Add, _) => Scalar::AddInteger,
        (tensor::ElementwiseOp::Subtract, LoweredTensorElement::Float { .. }) => {
            Scalar::SubtractFloat
        }
        (tensor::ElementwiseOp::Subtract, _) => Scalar::SubtractInteger,
        (tensor::ElementwiseOp::Multiply, LoweredTensorElement::Float { .. }) => {
            Scalar::MultiplyFloat
        }
        (tensor::ElementwiseOp::Multiply, _) => Scalar::MultiplyInteger,
        (tensor::ElementwiseOp::Divide, LoweredTensorElement::Float { .. }) => Scalar::DivideFloat,
        (tensor::ElementwiseOp::Divide, LoweredTensorElement::Integer { signed: true, .. }) => {
            Scalar::DivideSigned
        }
        (tensor::ElementwiseOp::Divide, _) => Scalar::DivideUnsigned,
        _ => {
            return Err(invalid(format!(
                "structured MLIR capability is missing for elementwise {operation:?}"
            )))
        }
    })
}

fn structural_reduction_axes(
    operation: tensor::ReductionOp,
    rank: usize,
    attributes: &Attrs,
) -> Result<BTreeSet<usize>, CompileError> {
    let axes = match operation {
        tensor::ReductionOp::Sum => (0..rank).collect(),
        tensor::ReductionOp::MeanLast | tensor::ReductionOp::MaxLast => [rank
            .checked_sub(1)
            .ok_or_else(|| invalid("last-axis reduction requires a non-scalar tensor"))?]
        .into_iter()
        .collect(),
        tensor::ReductionOp::SumAxis => match attributes.get(&tensor::REDUCTION_AXES) {
            Some(AttrValue::Integers(axes)) => axes
                .iter()
                .map(|axis| {
                    usize::try_from(*axis)
                        .map_err(|_| invalid("reduction axis must be a non-negative usize"))
                })
                .collect::<Result<BTreeSet<_>, _>>()?,
            _ => {
                return Err(invalid(
                    "SumAxis requires structural reduction axis identities",
                ))
            }
        },
    };
    if axes.is_empty() || axes.iter().any(|axis| *axis >= rank) {
        return Err(invalid(
            "reduction axes are empty or outside the known rank",
        ));
    }
    Ok(axes)
}

fn structural_reduction_combiner(
    operation: tensor::ReductionOp,
    element: LoweredTensorElement,
) -> Result<(String, ScalarBinaryOperation), CompileError> {
    use ScalarBinaryOperation as Scalar;
    Ok(match operation {
        tensor::ReductionOp::Sum | tensor::ReductionOp::SumAxis | tensor::ReductionOp::MeanLast => {
            (
                if is_float(element) { "0.0" } else { "0" }.into(),
                if is_float(element) {
                    Scalar::AddFloat
                } else {
                    Scalar::AddInteger
                },
            )
        }
        tensor::ReductionOp::MaxLast => match element {
            LoweredTensorElement::Float { format } => {
                (negative_infinity(format)?.into(), Scalar::MaximumFloat)
            }
            LoweredTensorElement::Integer { bits, signed: true } => {
                (signed_minimum(bits), Scalar::MaximumSigned)
            }
            LoweredTensorElement::Integer { signed: false, .. } | LoweredTensorElement::Boolean => {
                ("0".into(), Scalar::MaximumUnsigned)
            }
        },
    })
}

fn negative_infinity(format: LoweredFloatFormat) -> Result<String, CompileError> {
    match format {
        // E4M3FN has no infinity; 0xFE is its lowest finite value (-448).
        LoweredFloatFormat::Float8E4M3Fn => Ok("0xFE".into()),
        LoweredFloatFormat::Float8E5M2 => Ok("0xFC".into()),
        LoweredFloatFormat::BrainFloat16 => Ok("0xFF80".into()),
        LoweredFloatFormat::Ieee(16) => Ok("0xFC00".into()),
        LoweredFloatFormat::Ieee(32) => Ok("0xFF800000".into()),
        LoweredFloatFormat::Ieee(64) => Ok("0xFFF0000000000000".into()),
        LoweredFloatFormat::Ieee(80) => Ok("0xFFFF8000000000000000".into()),
        LoweredFloatFormat::Ieee(128) => {
            Ok("0xFFFF0000000000000000000000000000".into())
        }
        _ => Err(invalid(format!(
            "structured Max has no negative-infinity literal for {format:?}"
        ))),
    }
}

fn signed_minimum(bits: u16) -> String {
    if bits == 0 {
        "0".into()
    } else {
        format!("-{}", 1u128 << (bits - 1))
    }
}

fn lowered_scalar(element: LoweredTensorElement) -> LoweredType {
    match element {
        LoweredTensorElement::Integer { bits, signed } => LoweredType::Integer { bits, signed },
        LoweredTensorElement::Float { format } => LoweredType::Float { format },
        LoweredTensorElement::Boolean => LoweredType::Boolean,
    }
}

fn structured_error(error: impl std::fmt::Display) -> CompileError {
    invalid(format!("structured MLIR construction failed: {error}"))
}

fn legalize_cpu_operation(
    operation: tensor::TensorOp,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
) -> Result<(), CompileError> {
    let fail = |message: &str| {
        Err(invalid(format!(
            "CPU tensor MLIR legalization failed for {operation:?}: {message}"
        )))
    };
    let tensor_types = inputs
        .iter()
        .chain(outputs)
        .filter(|ty| matches!(ty, LoweredType::Tensor { .. }))
        .collect::<Vec<_>>();
    let require_known_ranks = || {
        if tensor_types
            .iter()
            .all(|ty| ranked_dimensions(ty).is_some())
        {
            Ok(())
        } else {
            fail("rank-dependent emission requires known tensor rank; dynamic dimensions are legal, unranked tensors are not")
        }
    };

    match operation {
        tensor::TensorOp::Elementwise(elementwise) => {
            require_known_ranks()?;
            if tensor_types.is_empty() {
                return fail("elementwise operations require tensor values");
            }
            let [output] = outputs else {
                return fail("elementwise operations require one result");
            };
            let output_element = tensor_element(output)?;
            if matches!(
                elementwise,
                tensor::ElementwiseOp::Exp
                    | tensor::ElementwiseOp::Log
                    | tensor::ElementwiseOp::Tanh
                    | tensor::ElementwiseOp::Rsqrt
                    | tensor::ElementwiseOp::Scale
                    | tensor::ElementwiseOp::AddScalar
            ) && !is_float(output_element)
            {
                return fail(
                    "this elementwise operation requires a floating element representation",
                );
            }
            for input in inputs
                .iter()
                .filter(|input| matches!(input, LoweredType::Tensor { .. }))
            {
                if tensor_element(input)? != output_element {
                    return fail("elementwise tensor operands and result must share one element representation");
                }
            }
            match elementwise {
                tensor::ElementwiseOp::Add
                | tensor::ElementwiseOp::Subtract
                | tensor::ElementwiseOp::Multiply
                | tensor::ElementwiseOp::Divide => {
                    let [left, right] = inputs else {
                        return fail("binary elementwise operations require two tensor operands");
                    };
                    let (Some(left), Some(right), Some(output)) = (
                        ranked_dimensions(left),
                        ranked_dimensions(right),
                        ranked_dimensions(output),
                    ) else {
                        return fail("binary elementwise values must be ranked tensors");
                    };
                    if output.len() != left.len().max(right.len())
                        || !broadcast_shape_is_compatible(left, output)
                        || !broadcast_shape_is_compatible(right, output)
                    {
                        return fail(
                            "binary elementwise operand shapes do not broadcast to the result",
                        );
                    }
                }
                tensor::ElementwiseOp::Exp
                | tensor::ElementwiseOp::Log
                | tensor::ElementwiseOp::Tanh
                | tensor::ElementwiseOp::Rsqrt
                | tensor::ElementwiseOp::Relu => {
                    if inputs.len() != 1 || inputs.first() != Some(output) {
                        return fail(
                            "unary elementwise operations must preserve the exact tensor type",
                        );
                    }
                }
                tensor::ElementwiseOp::Scale | tensor::ElementwiseOp::AddScalar => {
                    if inputs.len() != 2
                        || inputs.first() != Some(output)
                        || !matches!(inputs.get(1), Some(LoweredType::Float { .. }))
                    {
                        return fail("tensor-scalar elementwise operations require an unchanged tensor and a floating scalar");
                    }
                }
            }
        }
        tensor::TensorOp::Convert => {
            require_known_ranks()?;
            let ([input], [output]) = (inputs, outputs) else {
                return fail("Convert requires one tensor operand and one tensor result");
            };
            let (Some(input_shape), Some(output_shape)) =
                (ranked_dimensions(input), ranked_dimensions(output))
            else {
                unreachable!("known ranks were checked above")
            };
            if input_shape.len() != output_shape.len()
                || !input_shape
                    .iter()
                    .zip(output_shape)
                    .all(|(input, output)| dimensions_compatible(input, output))
            {
                return fail("Convert must preserve tensor rank and dimensions");
            }
        }
        tensor::TensorOp::Reduce(tensor::ReductionOp::MeanLast) => {
            require_known_ranks()?;
            let ([input], [output]) = (inputs, outputs) else {
                return fail("MeanLast requires one tensor operand and one tensor result");
            };
            let (Some(input), Some(output)) = (ranked_dimensions(input), ranked_dimensions(output))
            else {
                unreachable!("known ranks were checked above")
            };
            if input.is_empty() {
                return fail("MeanLast requires a rank of at least one");
            }
            if output.len() != input.len()
                || !output[..output.len() - 1]
                    .iter()
                    .zip(input)
                    .all(|(result, source)| dimensions_compatible(result, source))
                || output.last() != Some(&LoweredTensorDimension::Known(1))
            {
                return fail("MeanLast result must retain the final axis with extent one");
            }
            if matches!(input.last(), Some(LoweredTensorDimension::Known(0))) {
                return fail("MeanLast cannot reduce a statically empty final axis");
            }
            if !matches!(
                tensor_element(inputs.first().unwrap())?,
                LoweredTensorElement::Float { .. }
            ) {
                return fail("MeanLast requires a floating element representation");
            }
            if tensor_element(&inputs[0])? != tensor_element(&outputs[0])? {
                return fail("MeanLast operand and result must share one element representation");
            }
        }
        tensor::TensorOp::Reduce(
            reduction @ (tensor::ReductionOp::Sum
            | tensor::ReductionOp::SumAxis
            | tensor::ReductionOp::MaxLast),
        ) => {
            require_known_ranks()?;
            let ([input], [output]) = (inputs, outputs) else {
                return fail(
                    "structural reductions require one tensor operand and one tensor result",
                );
            };
            let (Some(input_shape), Some(output_shape)) =
                (ranked_dimensions(input), ranked_dimensions(output))
            else {
                unreachable!("known ranks were checked above")
            };
            if input_shape.is_empty() {
                return fail("reduction requires a rank of at least one");
            }
            if tensor_element(input)? != tensor_element(output)? {
                return fail("reduction operand and result must share one element representation");
            }
            if reduction == tensor::ReductionOp::MaxLast
                && (output_shape.len() != input_shape.len()
                    || output_shape.last() != Some(&LoweredTensorDimension::Known(1)))
            {
                return fail("MaxLast result must retain the final axis with extent one");
            }
        }
        tensor::TensorOp::Matmul => {
            require_known_ranks()?;
            let ([left, right], [output]) = (inputs, outputs) else {
                return fail("Matmul requires two tensor operands and one tensor result");
            };
            let (Some(left), Some(right), Some(output)) = (
                ranked_dimensions(left),
                ranked_dimensions(right),
                ranked_dimensions(output),
            ) else {
                unreachable!("known ranks were checked above")
            };
            if left.len() < 2 || right.len() < 2 || output.len() < 2 {
                return fail("Matmul operands and result require rank of at least two");
            }
            if output.len() != left.len().max(right.len()) {
                return fail("Matmul result rank must equal the maximum operand rank");
            }
            let left_element = tensor_element(inputs.first().unwrap())?;
            if tensor_element(&inputs[1])? != left_element
                || tensor_element(outputs.first().unwrap())? != left_element
            {
                return fail("Matmul operands and result must share one element representation");
            }
            if !dimensions_compatible(&left[left.len() - 1], &right[right.len() - 2]) {
                return fail("Matmul contraction dimensions are incompatible");
            }
            if !dimensions_compatible(&output[output.len() - 2], &left[left.len() - 2])
                || !dimensions_compatible(&output[output.len() - 1], &right[right.len() - 1])
            {
                return fail("Matmul result matrix dimensions do not match M and N");
            }
            legalize_matmul_batches(left, right, output).map_err(|message| {
                invalid(format!(
                    "CPU tensor MLIR legalization failed for {operation:?}: {message}"
                ))
            })?;
        }
        tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Materialize) => {
            require_known_ranks()?;
            if inputs.len() != 1 || outputs.len() != 1 || inputs[0] != outputs[0] {
                return fail("Materialize must preserve the exact tensor type");
            }
        }
        tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Reshape) => {
            require_known_ranks()?;
            let ([input, shape], [output]) = (inputs, outputs) else {
                return fail("Reshape requires one tensor, one shape operand, and one result");
            };
            if !matches!(shape, LoweredType::Bytes | LoweredType::String) {
                return fail("Reshape shape must use the opaque runtime list ABI");
            }
            if tensor_element(input)? != tensor_element(output)? {
                return fail("Reshape must preserve its element representation");
            }
        }
        tensor::TensorOp::Broadcast(broadcast) => {
            require_known_ranks()?;
            let ([input, specification], [output]) = (inputs, outputs) else {
                return fail("Broadcast requires two operands and one tensor result");
            };
            if tensor_element(input)? != tensor_element(output)? {
                return fail("Broadcast must preserve its element representation");
            }
            match broadcast {
                tensor::BroadcastOp::Like if !matches!(specification, LoweredType::Tensor { .. }) => {
                    return fail("BroadcastLike requires a tensor shape donor");
                }
                tensor::BroadcastOp::Repeat
                    if !matches!(specification, LoweredType::Bytes | LoweredType::String) =>
                {
                    return fail("Repeat requires an opaque axis/count list operand");
                }
                _ => {}
            }
        }
        tensor::TensorOp::Gather => {
            require_known_ranks()?;
            let ([source, indices], [output]) = (inputs, outputs) else {
                return fail("Gather requires source and indices tensors and one result");
            };
            let (Some(source_shape), Some(index_shape), Some(output_shape)) = (
                ranked_dimensions(source),
                ranked_dimensions(indices),
                ranked_dimensions(output),
            ) else {
                unreachable!("known ranks were checked above")
            };
            if source_shape.is_empty()
                || output_shape.len() != index_shape.len() + source_shape.len() - 1
            {
                return fail(
                    "Gather result rank must be index rank plus source rank minus one",
                );
            }
            if !matches!(
                tensor_element(indices)?,
                LoweredTensorElement::Integer { .. }
            ) {
                return fail("Gather indices must have an integer element representation");
            }
            if tensor_element(source)? != tensor_element(output)? {
                return fail("Gather must preserve the source element representation");
            }
        }
        tensor::TensorOp::Permute(permutation) => {
            require_known_ranks()?;
            let Some(input) = inputs.first() else {
                return fail("Permute requires one tensor operand");
            };
            let [output] = outputs else {
                return fail("Permute requires one tensor result");
            };
            if permutation == tensor::PermuteOp::Reverse && inputs.len() != 1
                || permutation == tensor::PermuteOp::Axes
                    && (inputs.len() != 2
                        || !matches!(inputs[1], LoweredType::Bytes | LoweredType::String))
            {
                return fail("Permute operands do not match its structural variant");
            }
            if tensor_element(input)? != tensor_element(output)? {
                return fail("Permute must preserve its element representation");
            }
            let (Some(input), Some(output)) = (ranked_dimensions(input), ranked_dimensions(output))
            else {
                unreachable!("known ranks were checked above")
            };
            if input.len() != output.len() {
                return fail("Permute must preserve tensor rank");
            }
            if permutation == tensor::PermuteOp::Reverse
                && !input
                    .iter()
                    .rev()
                    .zip(output)
                    .all(|(source, result)| dimensions_compatible(source, result))
            {
                return fail("Reverse result dimensions must reverse the input dimensions");
            }
        }
        tensor::TensorOp::StorageView(tensor::StorageViewOp::Shape) => {
            // This is metadata inspection of a genuine MLIR tensor value. A
            // host StorageViewAbi pointer is deliberately not a LoweredType::Tensor.
            let ([input], [_output]) = (inputs, outputs) else {
                return fail("Shape requires one tensor operand and one metadata result");
            };
            if !matches!(input, LoweredType::Tensor { .. }) {
                return fail(
                    "a host StorageViewAbi pointer cannot enter compute MLIR as a builtin tensor",
                );
            }
        }
        tensor::TensorOp::StorageView(tensor::StorageViewOp::FromAbi) => {
            require_known_ranks()?;
            let ([input], [output]) = (inputs, outputs) else {
                return fail("FromAbi requires one descriptor and one tensor result");
            };
            if !matches!(input, LoweredType::Bytes) {
                return fail("FromAbi requires an opaque StorageViewAbi pointer");
            }
            if !matches!(output, LoweredType::Tensor { .. }) {
                return fail("FromAbi requires one ranked tensor result");
            }
        }
        tensor::TensorOp::StorageView(
            storage_operation @ (tensor::StorageViewOp::FromElements
            | tensor::StorageViewOp::Strides
            | tensor::StorageViewOp::Values),
        ) => {
            require_known_ranks()?;
            match storage_operation {
                tensor::StorageViewOp::FromElements
                    if outputs.len() != 1 || !matches!(outputs[0], LoweredType::Tensor { .. }) =>
                {
                    return fail("FromElements requires one ranked tensor result");
                }
                tensor::StorageViewOp::Strides | tensor::StorageViewOp::Values
                    if inputs.len() != 1 || !matches!(inputs[0], LoweredType::Tensor { .. }) =>
                {
                    return fail(
                        "storage metadata inspection requires one genuine MLIR tensor operand",
                    );
                }
                _ => {}
            }
            if tensor_types.iter().any(|ty| {
                ranked_dimensions(ty).is_some_and(|dimensions| {
                    dimensions
                        .iter()
                        .any(|dimension| *dimension == LoweredTensorDimension::Dynamic)
                })
            }) {
                return fail(
                    "this storage metadata operation currently requires known dimension sizes",
                );
            }
        }
        tensor::TensorOp::Concatenate => {
            require_known_ranks()?;
            let ([left, right, axis], [output]) = (inputs, outputs) else {
                return fail("Concatenate requires two tensors, one axis operand, and one result");
            };
            if !matches!(axis, LoweredType::Bytes | LoweredType::String) {
                return fail("Concatenate axis must use the opaque runtime list ABI");
            }
            let (Some(left_shape), Some(right_shape), Some(output_shape)) = (
                ranked_dimensions(left),
                ranked_dimensions(right),
                ranked_dimensions(output),
            ) else {
                unreachable!("known ranks were checked above")
            };
            if left_shape.len() != right_shape.len() || left_shape.len() != output_shape.len() {
                return fail("Concatenate operands and result must have the same rank");
            }
            if tensor_element(left)? != tensor_element(right)?
                || tensor_element(left)? != tensor_element(output)?
            {
                return fail("Concatenate operands and result must share one element type");
            }
        }
        tensor::TensorOp::Scatter => {
            require_known_ranks()?;
            let ([source, indices, updates], [output]) = (inputs, outputs) else {
                return fail("Scatter requires source, indices, updates, and one result");
            };
            let (Some(source_shape), Some(index_shape), Some(update_shape), Some(output_shape)) = (
                ranked_dimensions(source),
                ranked_dimensions(indices),
                ranked_dimensions(updates),
                ranked_dimensions(output),
            ) else {
                unreachable!("known ranks were checked above")
            };
            if source_shape.is_empty()
                || index_shape.len() != 1
                || update_shape.len() != source_shape.len()
                || output_shape.len() != source_shape.len()
            {
                return fail(
                    "Scatter currently requires rank-one indices and rank-preserving updates",
                );
            }
            if !matches!(
                tensor_element(indices)?,
                LoweredTensorElement::Integer { .. }
            ) || tensor_element(source)? != tensor_element(updates)?
                || tensor_element(source)? != tensor_element(output)?
            {
                return fail("Scatter has incompatible index or element representations");
            }
        }
        tensor::TensorOp::Slice => {
            require_known_ranks()?;
            let (Some(input), Some(output)) = (inputs.first(), outputs.first()) else {
                return fail("Slice requires a tensor operand and tensor result");
            };
            let (Some(input), Some(output)) =
                (ranked_dimensions(input), ranked_dimensions(output))
            else {
                unreachable!("known ranks were checked above")
            };
            if input.len() != output.len() {
                return fail("Slice must preserve tensor rank");
            }
        }
    }
    Ok(())
}

fn ranked_dimensions(ty: &LoweredType) -> Option<&[LoweredTensorDimension]> {
    match ty {
        LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(dimensions),
            ..
        } => Some(dimensions),
        _ => None,
    }
}

fn dimensions_compatible(left: &LoweredTensorDimension, right: &LoweredTensorDimension) -> bool {
    matches!(left, LoweredTensorDimension::Dynamic)
        || matches!(right, LoweredTensorDimension::Dynamic)
        || left == right
}

fn broadcast_shape_is_compatible(
    source: &[LoweredTensorDimension],
    output: &[LoweredTensorDimension],
) -> bool {
    if source.len() > output.len() {
        return false;
    }
    let offset = output.len() - source.len();
    source.iter().enumerate().all(|(axis, source)| {
        source == &LoweredTensorDimension::Known(1)
            || dimensions_compatible(source, &output[offset + axis])
    })
}

fn legalize_matmul_batches(
    left: &[LoweredTensorDimension],
    right: &[LoweredTensorDimension],
    output: &[LoweredTensorDimension],
) -> Result<(), &'static str> {
    let output_batch = &output[..output.len() - 2];
    for result_axis in 0..output_batch.len() {
        let left_axis = result_axis.checked_sub(output_batch.len() - (left.len() - 2));
        let right_axis = result_axis.checked_sub(output_batch.len() - (right.len() - 2));
        let left_dimension = left_axis.map(|axis| &left[axis]);
        let right_dimension = right_axis.map(|axis| &right[axis]);
        let result = &output_batch[result_axis];
        for dimension in [left_dimension, right_dimension].into_iter().flatten() {
            if dimension != &LoweredTensorDimension::Known(1)
                && !dimensions_compatible(dimension, result)
            {
                return Err("Matmul batch dimensions are not broadcast-compatible with the result");
            }
        }
        if let (Some(left), Some(right)) = (left_dimension, right_dimension) {
            if left != &LoweredTensorDimension::Known(1)
                && right != &LoweredTensorDimension::Known(1)
                && !dimensions_compatible(left, right)
            {
                return Err("Matmul operand batch dimensions are not broadcast-compatible");
            }
        }
    }
    Ok(())
}

fn operation_declarations(
    operation: tensor::TensorOp,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
) -> Result<String, CompileError> {
    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::FromAbi) {
        return Ok("  func.func private @__sev_storage_view_validate(!llvm.ptr, i32, i32, i32, i64) -> i32\n  func.func private @__sev_storage_view_data(!llvm.ptr) -> !llvm.ptr\n  func.func private @__sev_storage_view_dimension(!llvm.ptr, i64) -> i64\n  func.func private @__sev_storage_view_stride(!llvm.ptr, i64) -> i64\n  func.func private @__sev_storage_view_offset(!llvm.ptr) -> i64\n".into());
    }
    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::FromElements) {
        return Ok("  func.func private @__sev_list_get_f64(!llvm.ptr, i64) -> f64\n".into());
    }
    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::Values) {
        let [LoweredType::Tensor { .. }] = inputs else {
            return Err(invalid("values requires one structural tensor operand"));
        };
        return Ok("  func.func private @__sev_list_create() -> !llvm.ptr\n  func.func private @__sev_list_push_f64(!llvm.ptr, f64)\n".into());
    }
    if matches!(
        operation,
        tensor::TensorOp::StorageView(
            tensor::StorageViewOp::Shape | tensor::StorageViewOp::Strides
        )
    ) {
        return Ok(
            "  func.func private @__sev_list_create() -> !llvm.ptr\n  func.func private @__sev_list_push_i64(!llvm.ptr, i64)\n"
                .into(),
        );
    }
    let _ = outputs;
    Ok(String::new())
}

fn lower_storage_view_from_abi(
    inputs: &[LoweredType],
    outputs: &[LoweredType],
    output_spellings: &[String],
) -> Result<String, CompileError> {
    let ([LoweredType::Bytes], [LoweredType::Tensor { element, shape }], [output]) =
        (inputs, outputs, output_spellings)
    else {
        return Err(invalid(
            "from_abi requires one opaque descriptor and one ranked tensor result",
        ));
    };
    let LoweredTensorShape::Ranked(dimensions) = shape else {
        return Err(invalid(
            "from_abi requires runtime specialization to establish result rank",
        ));
    };
    let (kind, bits, float_format) = storage_abi_representation(*element)?;
    let scalar = tensor_element_spelling(*element)?;
    let byte_width = u64::from(bits.div_ceil(8));
    let mut body = String::new();
    body.push_str(&format!(
        "    %expected_kind = arith.constant {kind} : i32\n    %expected_bits = arith.constant {bits} : i32\n    %expected_float_format = arith.constant {float_format} : i32\n    %expected_rank = arith.constant {} : i64\n    %valid_i32 = func.call @__sev_storage_view_validate(%arg0, %expected_kind, %expected_bits, %expected_float_format, %expected_rank) : (!llvm.ptr, i32, i32, i32, i64) -> i32\n    %zero_i32 = arith.constant 0 : i32\n    %valid = arith.cmpi ne, %valid_i32, %zero_i32 : i32\n    cf.assert %valid, \"StorageView element representation or rank does not match Tensor[T]\"\n    %data = func.call @__sev_storage_view_data(%arg0) : (!llvm.ptr) -> !llvm.ptr\n    %storage_offset = func.call @__sev_storage_view_offset(%arg0) : (!llvm.ptr) -> i64\n",
        dimensions.len()
    ));
    let mut dynamic_dimensions = Vec::new();
    for (axis, dimension) in dimensions.iter().enumerate() {
        body.push_str(&format!(
            "    %axis{axis} = arith.constant {axis} : i64\n    %dim{axis}_i64 = func.call @__sev_storage_view_dimension(%arg0, %axis{axis}) : (!llvm.ptr, i64) -> i64\n    %dim{axis} = arith.index_cast %dim{axis}_i64 : i64 to index\n    %stride{axis} = func.call @__sev_storage_view_stride(%arg0, %axis{axis}) : (!llvm.ptr, i64) -> i64\n"
        ));
        match dimension {
            LoweredTensorDimension::Dynamic => dynamic_dimensions.push(format!("%dim{axis}")),
            LoweredTensorDimension::Known(expected) => body.push_str(&format!(
                "    %known_dim{axis} = arith.constant {expected} : i64\n    %dim{axis}_matches = arith.cmpi eq, %dim{axis}_i64, %known_dim{axis} : i64\n    cf.assert %dim{axis}_matches, \"StorageView dimension {axis} does not match the specialized tensor contract\"\n"
            )),
        }
    }
    body.push_str(&format!(
        "    %empty = tensor.empty({}) : {output}\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constant 1 : index\n    %element_count0 = arith.constant 1 : index\n",
        dynamic_dimensions.join(", ")
    ));
    let mut count = "%element_count0".to_owned();
    for axis in 0..dimensions.len() {
        let next = format!("%element_count{}", axis + 1);
        body.push_str(&format!(
            "    {next} = arith.muli {count}, %dim{axis} : index\n"
        ));
        count = next;
    }
    body.push_str("    %zero_i64 = arith.constant 0 : i64\n");
    body.push_str(&format!(
        "    %filled = scf.for %linear = %c0 to {count} step %c1 iter_args(%current = %empty) -> ({output}) {{\n"
    ));
    let mut remaining = "%linear".to_owned();
    let mut coordinates = vec![String::new(); dimensions.len()];
    for axis in (0..dimensions.len()).rev() {
        let coordinate = format!("%coordinate{axis}");
        let next_remaining = format!("%remaining{axis}");
        body.push_str(&format!(
            "      {coordinate} = arith.remui {remaining}, %dim{axis} : index\n      {next_remaining} = arith.divui {remaining}, %dim{axis} : index\n"
        ));
        coordinates[axis] = coordinate;
        remaining = next_remaining;
    }
    body.push_str("      %physical0 = arith.addi %storage_offset, %zero_i64 : i64\n");
    let mut physical = "%physical0".to_owned();
    for (axis, coordinate) in coordinates.iter().enumerate() {
        let coordinate_i64 = format!("%coordinate{axis}_i64");
        let contribution = format!("%contribution{axis}");
        let next = format!("%physical{}", axis + 1);
        body.push_str(&format!(
            "      {coordinate_i64} = arith.index_cast {coordinate} : index to i64\n      {contribution} = arith.muli {coordinate_i64}, %stride{axis} : i64\n      {next} = arith.addi {physical}, {contribution} : i64\n"
        ));
        physical = next;
    }
    let load = match element {
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E4M3Fn | LoweredFloatFormat::Float8E5M2,
        } => format!(
            "      %element_bits = llvm.load %address : !llvm.ptr -> i8\n      %element = arith.bitcast %element_bits : i8 to {scalar}\n"
        ),
        _ => format!("      %element = llvm.load %address : !llvm.ptr -> {scalar}\n"),
    };
    body.push_str(&format!(
        "      %byte_width = arith.constant {byte_width} : i64\n      %byte_offset = arith.muli {physical}, %byte_width : i64\n      %address = llvm.getelementptr %data[%byte_offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n{load}      %next = tensor.insert %element into %current[{}] : {output}\n      scf.yield %next : {output}\n    }}\n    return %filled : {output}\n",
        coordinates.join(", ")
    ));
    Ok(body)
}

fn storage_abi_representation(
    element: LoweredTensorElement,
) -> Result<(u32, u32, u32), CompileError> {
    Ok(match element {
        LoweredTensorElement::Integer { bits, signed: true } => (1, u32::from(bits), 0),
        LoweredTensorElement::Integer {
            bits,
            signed: false,
        } => (2, u32::from(bits), 0),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Ieee(bits),
        } => (3, u32::from(bits), 1),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::BrainFloat16,
        } => (3, 16, 2),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E4M3Fn,
        } => (3, 8, 3),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E5M2,
        } => (3, 8, 4),
        LoweredTensorElement::Boolean => {
            return Err(invalid(
                "SafeTensor StorageView ABI does not define a Boolean element kind",
            ))
        }
    })
}

fn lower_operation(
    operation: tensor::TensorOp,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
    input_spellings: &[String],
    output_spellings: &[String],
    attributes: &Attrs,
) -> Result<String, CompileError> {
    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::FromAbi) {
        return lower_storage_view_from_abi(inputs, outputs, output_spellings);
    }
    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::FromElements) {
        let [output] = output_spellings else {
            return Err(invalid("from_elements requires one tensor result"));
        };
        let [LoweredType::Tensor { element, shape }] = outputs else {
            return Err(invalid("from_elements result must be a structural tensor"));
        };
        if *element
            != (LoweredTensorElement::Float {
                format: LoweredFloatFormat::Ieee(64),
            })
        {
            return Err(invalid(
                "ranked list construction currently consumes list[float]",
            ));
        }
        let count = static_element_count(shape)
            .map_err(|error| invalid(format!("from_elements: {error}")))?;
        let mut body = String::new();
        let mut elements = Vec::with_capacity(count);
        for index in 0..count {
            body.push_str(&format!(
                "    %index{index} = arith.constant {index} : i64\n    %element{index} = func.call @__sev_list_get_f64(%arg0, %index{index}) : (!llvm.ptr, i64) -> f64\n"
            ));
            elements.push(format!("%element{index}"));
        }
        body.push_str(&format!(
            "    %result = tensor.from_elements {} : {output}\n    return %result : {output}\n",
            elements.join(", ")
        ));
        return Ok(body);
    }

    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::Values) {
        let [input] = inputs else {
            return Err(invalid("values requires one tensor operand"));
        };
        let LoweredType::Tensor { element, shape } = input else {
            return Err(invalid("values operand must be a structural tensor"));
        };
        let scalar = tensor_element_spelling(*element)?;
        let list_element = LoweredTensorElement::Float {
            format: LoweredFloatFormat::Ieee(64),
        };
        let shape = effective_shape(shape, attributes);
        let dimensions =
            static_dimensions(&shape).map_err(|error| invalid(format!("values: {error}")))?;
        let coordinates = tensor_coordinates(&dimensions);
        let input_type = input_spellings
            .first()
            .ok_or_else(|| invalid("values input type is missing"))?;
        let ranked_type = type_spelling(&LoweredType::Tensor {
            element: *element,
            shape: shape.clone(),
        })
        .map_err(|error| invalid(error.to_string()))?;
        let (operand, cast) = if *input_type == ranked_type {
            ("%arg0", String::new())
        } else {
            (
                "%ranked",
                format!("    %ranked = tensor.cast %arg0 : {input_type} to {ranked_type}\n"),
            )
        };
        let mut body =
            format!("{cast}    %result = func.call @__sev_list_create() : () -> !llvm.ptr\n");
        for (ordinal, coordinate) in coordinates.iter().enumerate() {
            let mut indices = Vec::with_capacity(coordinate.len());
            for (axis, index) in coordinate.iter().enumerate() {
                let name = format!("%index{ordinal}_{axis}");
                body.push_str(&format!("    {name} = arith.constant {index} : index\n"));
                indices.push(name);
            }
            body.push_str(&format!(
                "    %element{ordinal} = tensor.extract {operand}[{}] : {ranked_type}\n",
                indices.join(", ")
            ));
            body.push_str(&lower_scalar_conversion(
                &format!("%element{ordinal}"),
                *element,
                list_element,
                &scalar,
                "f64",
                &format!("%element_f64{ordinal}"),
            )?);
            body.push_str(&format!(
                "    func.call @__sev_list_push_f64(%result, %element_f64{ordinal}) : (!llvm.ptr, f64) -> ()\n"
            ));
        }
        body.push_str("    return %result : !llvm.ptr\n");
        return Ok(body);
    }

    if matches!(
        operation,
        tensor::TensorOp::StorageView(
            tensor::StorageViewOp::Shape | tensor::StorageViewOp::Strides
        )
    ) {
        let [LoweredType::Tensor { shape, .. }] = inputs else {
            return Err(invalid("shape and strides require one tensor operand"));
        };
        let shape = effective_shape(shape, attributes);
        let dimensions = match static_dimensions(&shape) {
            Ok(dimensions) => dimensions,
            Err(_) if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::Shape) => {
                let input = input_spellings
                    .first()
                    .ok_or_else(|| invalid("shape input type is missing"))?;
                return Ok(format!(
                    "    %result = func.call @__sev_list_create() : () -> !llvm.ptr\n    %rank = tensor.rank %arg0 : {input}\n    %zero = arith.constant 0 : index\n    %one = arith.constant 1 : index\n    scf.for %axis = %zero to %rank step %one {{\n      %dimension = tensor.dim %arg0, %axis : {input}\n      %dimension_i64 = arith.index_cast %dimension : index to i64\n      func.call @__sev_list_push_i64(%result, %dimension_i64) : (!llvm.ptr, i64) -> ()\n    }}\n    return %result : !llvm.ptr\n"
                ));
            }
            Err(error) => return Err(invalid(format!("shape metadata: {error}"))),
        };
        let values = if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::Shape) {
            dimensions
        } else {
            let mut stride = 1usize;
            let mut strides = vec![0; dimensions.len()];
            for axis in (0..dimensions.len()).rev() {
                strides[axis] = stride;
                stride = stride
                    .checked_mul(dimensions[axis])
                    .ok_or_else(|| invalid("tensor stride overflow"))?;
            }
            strides
        };
        let mut body =
            "    %result = func.call @__sev_list_create() : () -> !llvm.ptr\n".to_owned();
        for (index, value) in values.into_iter().enumerate() {
            body.push_str(&format!(
                "    %value{index} = arith.constant {value} : i64\n    func.call @__sev_list_push_i64(%result, %value{index}) : (!llvm.ptr, i64) -> ()\n"
            ));
        }
        body.push_str("    return %result : !llvm.ptr\n");
        return Ok(body);
    }

    let [output] = output_spellings else {
        return Err(invalid("tensor operations currently produce one result"));
    };
    let [result_type] = outputs else {
        return Err(invalid("tensor operations currently produce one result"));
    };
    let result_element = tensor_element(result_type)?;
    if operation == tensor::TensorOp::Convert {
        return lower_tensor_conversion(inputs, result_type, input_spellings, output);
    }
    let binary = if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add) {
        Some(if is_float(result_element) {
            "arith.addf"
        } else {
            "arith.addi"
        })
    } else if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Subtract) {
        Some(if is_float(result_element) {
            "arith.subf"
        } else {
            "arith.subi"
        })
    } else if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Multiply) {
        Some(if is_float(result_element) {
            "arith.mulf"
        } else {
            "arith.muli"
        })
    } else if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Divide) {
        Some(match result_element {
            LoweredTensorElement::Float { .. } => "arith.divf",
            LoweredTensorElement::Integer { signed: true, .. } => "arith.divsi",
            LoweredTensorElement::Integer { signed: false, .. } | LoweredTensorElement::Boolean => {
                "arith.divui"
            }
        })
    } else {
        None
    };
    if let Some(instruction) = binary {
        if inputs.len() != 2 {
            return Err(invalid("binary tensor operations require two operands"));
        }
        if input_spellings != [output.clone(), output.clone()] {
            let [LoweredType::Tensor {
                element: left_element,
                shape: LoweredTensorShape::Ranked(left_shape),
            }, LoweredType::Tensor {
                element: right_element,
                shape: LoweredTensorShape::Ranked(right_shape),
            }] = inputs
            else {
                return Err(invalid(
                    "broadcasting binary operations require statically ranked tensors",
                ));
            };
            let LoweredType::Tensor {
                element: output_element,
                shape: LoweredTensorShape::Ranked(output_shape),
            } = result_type
            else {
                return Err(invalid(
                    "broadcasting binary operations require a statically ranked result",
                ));
            };
            if left_element != output_element || right_element != output_element {
                return Err(invalid("binary tensor operands must have one element type"));
            }
            let scalar = tensor_element_spelling(*output_element)?;
            let loops = (0..output_shape.len())
                .map(|axis| format!("d{axis}"))
                .collect::<Vec<_>>();
            let left_map = broadcast_indexing_map(left_shape, output_shape.len())?;
            let right_map = broadcast_indexing_map(right_shape, output_shape.len())?;
            let output_rank = output_shape.len();
            let (dynamic_sizes, dynamic_arguments) = dynamic_result_dimensions(
                output_shape,
                |axis| {
                    let left = axis
                        .checked_sub(output_rank - left_shape.len())
                        .map(|source_axis| (0, source_axis, &left_shape[source_axis]));
                    let right = axis
                        .checked_sub(output_rank - right_shape.len())
                        .map(|source_axis| (1, source_axis, &right_shape[source_axis]));
                    [left, right]
                        .into_iter()
                        .flatten()
                        .find(|(_, _, dimension)| **dimension != LoweredTensorDimension::Known(1))
                        .or(left)
                        .or(right)
                        .map(|(operand, source_axis, _)| (operand, source_axis))
                },
                inputs,
                input_spellings,
            )?;
            return Ok(format!(
                "{dynamic_sizes}    %empty = tensor.empty({dynamic_arguments}) : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({left_map})>, affine_map<({loops}) -> ({right_map})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterators}]}} ins(%arg0, %arg1 : {left_type}, {right_type}) outs(%empty : {output}) {{\n    ^bb0(%left: {scalar}, %right: {scalar}, %unused: {scalar}):\n      %value = {instruction} %left, %right : {scalar}\n      linalg.yield %value : {scalar}\n    }} -> {output}\n    return %result : {output}\n",
                loops = loops.join(", "),
                left_map = left_map.join(", "),
                right_map = right_map.join(", "),
                iterators = vec!["\"parallel\""; output_shape.len()].join(", "),
                left_type = input_spellings[0],
                right_type = input_spellings[1],
            ));
        }
        return Ok(format!(
            "    %result = {instruction} %arg0, %arg1 : {output}\n    return %result : {output}\n"
        ));
    }

    let unary = if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Exp) {
        Some("math.exp")
    } else if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Log) {
        Some("math.log")
    } else if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Tanh) {
        Some("math.tanh")
    } else if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Rsqrt) {
        Some("math.rsqrt")
    } else {
        None
    };
    if let Some(instruction) = unary {
        if !is_float(result_element) {
            return Err(invalid(
                "floating tensor operation requires a floating element",
            ));
        }
        if input_spellings.len() != 1 || input_spellings.first() != Some(output) {
            return Err(invalid(
                "unary tensor operand and result must have one exact type",
            ));
        }
        return Ok(format!(
            "    %result = {instruction} %arg0 : {output}\n    return %result : {output}\n"
        ));
    }

    if matches!(
        operation,
        tensor::TensorOp::Elementwise(
            tensor::ElementwiseOp::Scale | tensor::ElementwiseOp::AddScalar
        )
    ) {
        let [LoweredType::Tensor {
            element,
            shape: LoweredTensorShape::Ranked(dimensions),
        }, LoweredType::Float {
            format: scalar_format,
        }] = inputs
        else {
            return Err(invalid(
                "tensor-scalar elementwise operations require a ranked tensor and float scalar",
            ));
        };
        if element != &result_element || input_spellings.first() != Some(output) {
            return Err(invalid(
                "tensor-scalar elementwise operations must preserve the tensor type",
            ));
        }
        let LoweredTensorElement::Float { .. } = element else {
            return Err(invalid(
                "tensor-scalar model operations require floating tensors",
            ));
        };
        let tensor_scalar = tensor_element_spelling(*element)?;
        let argument_element = LoweredTensorElement::Float {
            format: *scalar_format,
        };
        let argument_scalar = type_spelling(&LoweredType::Float {
            format: *scalar_format,
        })
        .map_err(|error| invalid(error.to_string()))?;
        let scalar_conversion = lower_scalar_conversion(
            "%arg1",
            argument_element,
            *element,
            &argument_scalar,
            &tensor_scalar,
            "%scalar",
        )?;
        let loops = (0..dimensions.len())
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>()
            .join(", ");
        let instruction =
            if operation == tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Scale) {
                "arith.mulf"
            } else {
                "arith.addf"
            };
        let (dynamic_sizes, dynamic_arguments) =
            dynamic_result_dimensions(dimensions, |axis| Some((0, axis)), inputs, input_spellings)?;
        return Ok(format!(
            "{scalar_conversion}{dynamic_sizes}    %empty = tensor.empty({dynamic_arguments}) : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({loops})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterators}]}} ins(%arg0 : {output}) outs(%empty : {output}) {{\n    ^bb0(%value: {tensor_scalar}, %unused: {tensor_scalar}):\n      %computed = {instruction} %value, %scalar : {tensor_scalar}\n      linalg.yield %computed : {tensor_scalar}\n    }} -> {output}\n    return %result : {output}\n",
            iterators = vec!["\"parallel\""; dimensions.len()].join(", "),
        ));
    }

    if operation == tensor::TensorOp::Reduce(tensor::ReductionOp::MeanLast) {
        let [LoweredType::Tensor {
            element,
            shape: LoweredTensorShape::Ranked(input_dimensions),
        }] = inputs
        else {
            return Err(invalid("last-axis reduction requires one tensor operand"));
        };
        if element != &result_element || !is_float(*element) {
            return Err(invalid(
                "last-axis model reductions require one floating element type",
            ));
        }
        let Some(width) = input_dimensions.last() else {
            return Err(invalid("last-axis reduction requires a non-scalar tensor"));
        };
        if width == &LoweredTensorDimension::Known(0) {
            return Err(invalid(
                "last-axis reduction cannot reduce an empty dimension",
            ));
        }
        let LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(output_dimensions),
            ..
        } = result_type
        else {
            return Err(invalid(
                "last-axis reduction requires a ranked tensor result",
            ));
        };
        let rank = input_dimensions.len();
        let loops = (0..rank).map(|axis| format!("d{axis}")).collect::<Vec<_>>();
        let outer = loops[..rank - 1].join(", ");
        let scalar = tensor_element_spelling(*element)?;
        let (dynamic_sizes, dynamic_arguments) = dynamic_result_dimensions(
            output_dimensions,
            |axis| Some((0, axis)),
            inputs,
            input_spellings,
        )?;
        let width = match width {
            LoweredTensorDimension::Known(width) => {
                format!("    %width = arith.constant {width}.0 : {scalar}\n")
            }
            LoweredTensorDimension::Dynamic => {
                let last = rank - 1;
                format!(
                    "    %width_axis = arith.constant {last} : index\n    %width_index = tensor.dim %arg0, %width_axis : {input}\n    %width_i64 = arith.index_cast %width_index : index to i64\n    %width = arith.uitofp %width_i64 : i64 to {scalar}\n",
                    input = input_spellings[0],
                )
            }
        };
        return Ok(format!(
            "{dynamic_sizes}    %empty = tensor.empty({dynamic_arguments}) : {output}\n    %initial_value = arith.constant 0.0 : {scalar}\n    %initialized = linalg.fill ins(%initial_value : {scalar}) outs(%empty : {output}) -> {output}\n{width}    %reduced = linalg.generic {{indexing_maps = [affine_map<({identity}) -> ({identity})>, affine_map<({identity}) -> ({outer})>], iterator_types = [{iterators}]}} ins(%arg0 : {input}) outs(%initialized : {output}) {{\n    ^bb0(%value: {scalar}, %accumulator: {scalar}):\n      %prepared = arith.divf %value, %width : {scalar}\n      %next = arith.addf %accumulator, %prepared : {scalar}\n      linalg.yield %next : {scalar}\n    }} -> {output}\n    return %reduced : {output}\n",
            identity = loops.join(", "),
            iterators = (0..rank)
                .map(|axis| if axis + 1 == rank { "\"reduction\"" } else { "\"parallel\"" })
                .collect::<Vec<_>>()
                .join(", "),
            input = input_spellings[0],
        ));
    }

    if operation == tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Materialize) {
        if input_spellings.len() != 1 || input_spellings.first() != Some(output) {
            return Err(invalid("materialize preserves the complete tensor type"));
        }
        return Ok(format!("    return %arg0 : {output}\n"));
    }

    if operation == tensor::TensorOp::Permute(tensor::PermuteOp::Reverse) {
        let [input] = input_spellings else {
            return Err(invalid("transpose requires one tensor operand"));
        };
        let LoweredType::Tensor {
            element,
            shape: LoweredTensorShape::Ranked(input_dimensions),
        } = &inputs[0]
        else {
            return Err(invalid("transpose requires a ranked tensor operand"));
        };
        let LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(output_dimensions),
            ..
        } = result_type
        else {
            return Err(invalid("transpose requires a ranked tensor result"));
        };
        if input_dimensions.len() != output_dimensions.len() {
            return Err(invalid("transpose must preserve tensor rank"));
        }
        let rank = input_dimensions.len();
        let scalar = tensor_element_spelling(*element)?;
        let loop_dimensions = (0..rank).map(|axis| format!("d{axis}")).collect::<Vec<_>>();
        let input_map = loop_dimensions.iter().rev().cloned().collect::<Vec<_>>();
        let iterator_types = vec!["\"parallel\""; rank].join(", ");
        let (dynamic_sizes, dynamic_arguments) = dynamic_result_dimensions(
            output_dimensions,
            |axis| Some((0, rank - 1 - axis)),
            inputs,
            input_spellings,
        )?;
        return Ok(format!(
            "{dynamic_sizes}    %empty = tensor.empty({dynamic_arguments}) : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({input_map})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterator_types}]}} ins(%arg0 : {input}) outs(%empty : {output}) {{\n    ^bb0(%element: {scalar}, %unused: {scalar}):\n      linalg.yield %element : {scalar}\n    }} -> {output}\n    return %result : {output}\n",
            loops = loop_dimensions.join(", "),
            input_map = input_map.join(", "),
        ));
    }

    if operation == tensor::TensorOp::Matmul {
        let [LoweredType::Tensor {
            element: left_element,
            shape: left_shape,
        }, LoweredType::Tensor {
            element: right_element,
            shape: right_shape,
        }] = inputs
        else {
            return Err(invalid("matmul requires two tensor operands"));
        };
        if left_element != right_element || left_element != &result_element {
            return Err(invalid(
                "matmul operands and result must have one element type",
            ));
        }
        let left_dimensions = match left_shape {
            LoweredTensorShape::Ranked(dimensions) if dimensions.len() >= 2 => dimensions,
            _ => {
                return Err(invalid(
                    "matmul left operand must have known rank at least two",
                ))
            }
        };
        let right_dimensions = match right_shape {
            LoweredTensorShape::Ranked(dimensions) if dimensions.len() >= 2 => dimensions,
            _ => {
                return Err(invalid(
                    "matmul right operand must have known rank at least two",
                ))
            }
        };
        let LoweredType::Tensor { element, shape } = result_type else {
            unreachable!();
        };
        let shape = effective_shape(shape, attributes);
        let LoweredTensorShape::Ranked(output_dimensions) = &shape else {
            return Err(invalid("matmul result must have known rank"));
        };
        if output_dimensions.len() < 2 {
            return Err(invalid("matmul result rank must be at least two"));
        }
        let scalar = tensor_element_spelling(*element)?;
        let zero = if is_float(*element) { "0.0" } else { "0" };
        let ranked_type = type_spelling(&LoweredType::Tensor {
            element: *element,
            shape: shape.clone(),
        })
        .map_err(|error| invalid(error.to_string()))?;
        let output_rank = output_dimensions.len();
        let batch_rank = output_rank - 2;
        let (dynamic_sizes, dynamic_arguments) = dynamic_result_dimensions(
            output_dimensions,
            |axis| {
                if axis == batch_rank {
                    return Some((0, left_dimensions.len() - 2));
                }
                if axis == batch_rank + 1 {
                    return Some((1, right_dimensions.len() - 1));
                }
                let left_offset = batch_rank - (left_dimensions.len() - 2);
                let right_offset = batch_rank - (right_dimensions.len() - 2);
                let left = axis
                    .checked_sub(left_offset)
                    .map(|source_axis| (0, source_axis, &left_dimensions[source_axis]));
                let right = axis
                    .checked_sub(right_offset)
                    .map(|source_axis| (1, source_axis, &right_dimensions[source_axis]));
                [left, right]
                    .into_iter()
                    .flatten()
                    .find(|(_, _, dimension)| **dimension != LoweredTensorDimension::Known(1))
                    .or(left)
                    .or(right)
                    .map(|(operand, source_axis, _)| (operand, source_axis))
            },
            inputs,
            input_spellings,
        )?;
        let loops = (0..=output_rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let batch_map = |dimensions: &[LoweredTensorDimension]| {
            let source_batch = dimensions.len() - 2;
            let offset = batch_rank - source_batch;
            dimensions[..source_batch]
                .iter()
                .enumerate()
                .map(|(axis, dimension)| match dimension {
                    LoweredTensorDimension::Known(1) => "0".into(),
                    _ => format!("d{}", offset + axis),
                })
                .collect::<Vec<String>>()
        };
        let mut left_map = batch_map(left_dimensions);
        left_map.push(format!("d{batch_rank}"));
        left_map.push(format!("d{output_rank}"));
        let mut right_map = batch_map(right_dimensions);
        right_map.push(format!("d{output_rank}"));
        right_map.push(format!("d{}", batch_rank + 1));
        let output_map = (0..output_rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let mut iterators = vec!["\"parallel\""; output_rank];
        iterators.push("\"reduction\"");
        let multiply = if is_float(*element) {
            "arith.mulf"
        } else {
            "arith.muli"
        };
        let add = if is_float(*element) {
            "arith.addf"
        } else {
            "arith.addi"
        };
        return Ok(format!(
            "{dynamic_sizes}    %empty = tensor.empty({dynamic_arguments}) : {ranked_type}\n    %zero = arith.constant {zero} : {scalar}\n    %initialized = linalg.fill ins(%zero : {scalar}) outs(%empty : {ranked_type}) -> {ranked_type}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({left_map})>, affine_map<({loops}) -> ({right_map})>, affine_map<({loops}) -> ({output_map})>], iterator_types = [{iterators}]}} ins(%arg0, %arg1 : {left_type}, {right_type}) outs(%initialized : {ranked_type}) {{\n    ^bb0(%left: {scalar}, %right: {scalar}, %acc: {scalar}):\n      %product = {multiply} %left, %right : {scalar}\n      %sum = {add} %acc, %product : {scalar}\n      linalg.yield %sum : {scalar}\n    }} -> {ranked_type}\n    return %result : {output}\n",
            left_type = input_spellings[0],
            right_type = input_spellings[1],
            loops = loops.join(", "),
            left_map = left_map.join(", "),
            right_map = right_map.join(", "),
            output_map = output_map.join(", "),
            iterators = iterators.join(", "),
        ));
    }

    Err(invalid(format!(
        "generic MLIR tensor lowering is not implemented for operation {operation:?}"
    )))
}

fn lower_tensor_conversion(
    inputs: &[LoweredType],
    result_type: &LoweredType,
    input_spellings: &[String],
    output: &str,
) -> Result<String, CompileError> {
    let [LoweredType::Tensor {
        element: source_element,
        shape: LoweredTensorShape::Ranked(source_shape),
    }] = inputs
    else {
        return Err(invalid("tensor conversion requires one ranked operand"));
    };
    let LoweredType::Tensor {
        element: target_element,
        shape: LoweredTensorShape::Ranked(target_shape),
    } = result_type
    else {
        return Err(invalid("tensor conversion requires a ranked result"));
    };
    if source_shape.len() != target_shape.len()
        || !source_shape
            .iter()
            .zip(target_shape)
            .all(|(source, target)| dimensions_compatible(source, target))
    {
        return Err(invalid("tensor conversion must preserve shape"));
    }
    let source_scalar = tensor_element_spelling(*source_element)?;
    let target_scalar = tensor_element_spelling(*target_element)?;
    let conversion = lower_scalar_conversion(
        "%value",
        *source_element,
        *target_element,
        &source_scalar,
        &target_scalar,
        "%converted",
    )?;
    let rank = source_shape.len();
    let loops = (0..rank)
        .map(|axis| format!("d{axis}"))
        .collect::<Vec<_>>()
        .join(", ");
    let iterators = vec!["\"parallel\""; rank].join(", ");
    let input = input_spellings
        .first()
        .ok_or_else(|| invalid("tensor conversion input type is missing"))?;
    let (dynamic_sizes, dynamic_arguments) = dynamic_result_dimensions(
        target_shape,
        |axis| Some((0, axis)),
        inputs,
        input_spellings,
    )?;
    Ok(format!(
        "{dynamic_sizes}    %empty = tensor.empty({dynamic_arguments}) : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({loops})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterators}]}} ins(%arg0 : {input}) outs(%empty : {output}) {{\n    ^bb0(%value: {source_scalar}, %unused: {target_scalar}):\n{conversion}      linalg.yield %converted : {target_scalar}\n    }} -> {output}\n    return %result : {output}\n"
    ))
}

fn lower_scalar_conversion(
    value: &str,
    source: LoweredTensorElement,
    target: LoweredTensorElement,
    source_type: &str,
    target_type: &str,
    result: &str,
) -> Result<String, CompileError> {
    if source == target {
        let (zero, add) = if is_float(source) {
            ("0.0", "arith.addf")
        } else {
            ("0", "arith.addi")
        };
        return Ok(format!(
            "    {result}_zero = arith.constant {zero} : {source_type}\n    {result} = {add} {value}, {result}_zero : {source_type}\n"
        ));
    }
    let conversion = match (source, target) {
        (
            LoweredTensorElement::Float { format: source },
            LoweredTensorElement::Float { format: target },
        ) if float_format_bits(source) == float_format_bits(target) => {
            let intermediate = if float_format_bits(source) < 16 {
                "f16"
            } else {
                "f32"
            };
            return Ok(format!(
                "    {result}_wide = arith.extf {value} : {source_type} to {intermediate}\n    {result} = arith.truncf {result}_wide : {intermediate} to {target_type}\n"
            ));
        }
        (
            LoweredTensorElement::Float { format: source },
            LoweredTensorElement::Float { format: target },
        ) => {
            if float_format_bits(source) < float_format_bits(target) {
                "arith.extf"
            } else {
                "arith.truncf"
            }
        }
        (
            LoweredTensorElement::Integer {
                bits: source,
                signed,
            },
            LoweredTensorElement::Integer { bits: target, .. },
        ) => {
            if source < target {
                if signed {
                    "arith.extsi"
                } else {
                    "arith.extui"
                }
            } else if source > target {
                "arith.trunci"
            } else {
                let zero = "0";
                return Ok(format!(
                    "    {result}_zero = arith.constant {zero} : {source_type}\n    {result} = arith.addi {value}, {result}_zero : {source_type}\n"
                ));
            }
        }
        (
            LoweredTensorElement::Integer { signed: true, .. },
            LoweredTensorElement::Float { .. },
        ) => "arith.sitofp",
        (
            LoweredTensorElement::Integer { signed: false, .. },
            LoweredTensorElement::Float { .. },
        ) => "arith.uitofp",
        (
            LoweredTensorElement::Float { .. },
            LoweredTensorElement::Integer { signed: true, .. },
        ) => "arith.fptosi",
        (
            LoweredTensorElement::Float { .. },
            LoweredTensorElement::Integer { signed: false, .. },
        ) => "arith.fptoui",
        _ => return Err(invalid("unsupported scalar tensor conversion")),
    };
    Ok(format!(
        "    {result} = {conversion} {value} : {source_type} to {target_type}\n"
    ))
}

const fn float_format_bits(format: LoweredFloatFormat) -> u16 {
    match format {
        LoweredFloatFormat::Float8E4M3Fn | LoweredFloatFormat::Float8E5M2 => 8,
        LoweredFloatFormat::Ieee(bits) => bits,
        LoweredFloatFormat::BrainFloat16 => 16,
    }
}

fn effective_shape(shape: &LoweredTensorShape, attributes: &Attrs) -> LoweredTensorShape {
    if !matches!(shape, LoweredTensorShape::Unranked) {
        return shape.clone();
    }
    let Some(AttrValue::TensorShape(shape)) = attributes.get(&tensor::RESULT_SHAPE) else {
        return shape.clone();
    };
    match shape {
        TensorShape::Unranked => LoweredTensorShape::Unranked,
        TensorShape::Ranked(dimensions) => LoweredTensorShape::Ranked(
            dimensions
                .iter()
                .map(|dimension| match dimension {
                    TensorDimension::Dynamic => LoweredTensorDimension::Dynamic,
                    TensorDimension::Known(value) => LoweredTensorDimension::Known(*value),
                })
                .collect(),
        ),
    }
}

fn tensor_element_spelling(element: LoweredTensorElement) -> Result<String, CompileError> {
    type_spelling(&match element {
        LoweredTensorElement::Integer { bits, signed } => LoweredType::Integer { bits, signed },
        LoweredTensorElement::Float { format } => LoweredType::Float { format },
        LoweredTensorElement::Boolean => LoweredType::Boolean,
    })
    .map_err(|error| invalid(error.to_string()))
}

fn static_dimensions(shape: &LoweredTensorShape) -> Result<Vec<usize>, CompileError> {
    let LoweredTensorShape::Ranked(dimensions) = shape else {
        return Err(invalid("operation requires a statically ranked tensor"));
    };
    dimensions
        .iter()
        .map(|dimension| match dimension {
            LoweredTensorDimension::Known(value) => {
                usize::try_from(*value).map_err(|_| invalid("tensor dimension does not fit usize"))
            }
            LoweredTensorDimension::Dynamic => Err(invalid(
                "operation requires statically known tensor dimensions",
            )),
        })
        .collect()
}

fn dynamic_result_dimensions(
    dimensions: &[LoweredTensorDimension],
    source_for_axis: impl Fn(usize) -> Option<(usize, usize)>,
    inputs: &[LoweredType],
    input_spellings: &[String],
) -> Result<(String, String), CompileError> {
    let mut definitions = String::new();
    let mut arguments = Vec::new();
    for (axis, dimension) in dimensions.iter().enumerate() {
        if *dimension != LoweredTensorDimension::Dynamic {
            continue;
        }
        let (operand, source_axis) = source_for_axis(axis).ok_or_else(|| {
            invalid(format!(
                "dynamic result dimension {axis} has no runtime shape source"
            ))
        })?;
        let Some(LoweredType::Tensor {
            shape: LoweredTensorShape::Ranked(source_dimensions),
            ..
        }) = inputs.get(operand)
        else {
            return Err(invalid(format!(
                "dynamic result dimension {axis} does not reference a ranked tensor operand"
            )));
        };
        if source_axis >= source_dimensions.len() {
            return Err(invalid(format!(
                "dynamic result dimension {axis} references missing operand axis {source_axis}"
            )));
        }
        let input = input_spellings
            .get(operand)
            .ok_or_else(|| invalid("dynamic result shape source type is missing"))?;
        definitions.push_str(&format!(
            "    %result_dim{axis}_axis = arith.constant {source_axis} : index\n    %result_dim{axis} = tensor.dim %arg{operand}, %result_dim{axis}_axis : {input}\n"
        ));
        arguments.push(format!("%result_dim{axis}"));
    }
    Ok((definitions, arguments.join(", ")))
}

fn static_element_count(shape: &LoweredTensorShape) -> Result<usize, CompileError> {
    static_dimensions(shape)?
        .into_iter()
        .try_fold(1usize, |count, dimension| {
            count
                .checked_mul(dimension)
                .ok_or_else(|| invalid("tensor element count overflow"))
        })
}

fn tensor_coordinates(dimensions: &[usize]) -> Vec<Vec<usize>> {
    let count = dimensions.iter().product();
    (0..count)
        .map(|mut linear| {
            let mut coordinate = vec![0; dimensions.len()];
            for axis in (0..dimensions.len()).rev() {
                coordinate[axis] = linear % dimensions[axis];
                linear /= dimensions[axis];
            }
            coordinate
        })
        .collect()
}

fn broadcast_indexing_map(
    source: &[LoweredTensorDimension],
    result_rank: usize,
) -> Result<Vec<String>, CompileError> {
    if source.len() > result_rank {
        return Err(invalid("broadcast source rank exceeds result rank"));
    }
    let offset = result_rank - source.len();
    Ok(source
        .iter()
        .enumerate()
        .map(|(axis, dimension)| match dimension {
            LoweredTensorDimension::Known(1) => "0".into(),
            LoweredTensorDimension::Known(_) | LoweredTensorDimension::Dynamic => {
                format!("d{}", offset + axis)
            }
        })
        .collect())
}

fn tensor_element(ty: &LoweredType) -> Result<LoweredTensorElement, CompileError> {
    match ty {
        LoweredType::Tensor { element, .. } => Ok(*element),
        _ => Err(invalid(
            "tensor operation result is not a structural tensor type",
        )),
    }
}

const fn is_float(element: LoweredTensorElement) -> bool {
    matches!(element, LoweredTensorElement::Float { .. })
}

fn lower_type(type_id: TypeId, context: &CompileContext<'_>) -> Result<LoweredType, CompileError> {
    if let Some(tensor) = context.types.tensor(type_id) {
        let element = lower_tensor_element(tensor.element, context.types, context)?;
        let shape = match tensor.shape {
            TensorShape::Unranked => LoweredTensorShape::Unranked,
            TensorShape::Ranked(dimensions) => LoweredTensorShape::Ranked(
                dimensions
                    .into_iter()
                    .map(|dimension| match dimension {
                        TensorDimension::Dynamic => LoweredTensorDimension::Dynamic,
                        TensorDimension::Known(value) => LoweredTensorDimension::Known(value),
                    })
                    .collect(),
            ),
        };
        return Ok(LoweredType::Tensor { element, shape });
    }
    lower_primitive(type_id, context.types, context)
}

fn lower_tensor_element(
    type_id: TypeId,
    types: &TypeContext,
    context: &CompileContext<'_>,
) -> Result<LoweredTensorElement, CompileError> {
    match lower_primitive(type_id, types, context)? {
        LoweredType::Integer { bits, signed } => Ok(LoweredTensorElement::Integer { bits, signed }),
        LoweredType::Float { format } => Ok(LoweredTensorElement::Float { format }),
        LoweredType::Boolean => Ok(LoweredTensorElement::Boolean),
        _ => Err(invalid(format!(
            "type {type_id:?} is not a scalar tensor element"
        ))),
    }
}

fn lower_primitive(
    type_id: TypeId,
    types: &TypeContext,
    context: &CompileContext<'_>,
) -> Result<LoweredType, CompileError> {
    let primitive = types
        .primitive(type_id)
        .ok_or_else(|| invalid(format!("type {type_id:?} has no primitive representation")))?;
    Ok(match primitive.representation {
        PrimitiveRepresentation::Integer { bits, signed } => LoweredType::Integer {
            bits: match bits {
                IntegerWidth::Fixed(bits) => bits,
                IntegerWidth::Machine => context.target.machine_integer_bits(),
            },
            signed,
        },
        PrimitiveRepresentation::PointerInteger { signed } => LoweredType::Integer {
            bits: context.target.pointer_bits(),
            signed,
        },
        PrimitiveRepresentation::Float { format } => LoweredType::Float {
            format: match format {
                FloatFormat::Float8E4M3Fn => LoweredFloatFormat::Float8E4M3Fn,
                FloatFormat::Float8E5M2 => LoweredFloatFormat::Float8E5M2,
                FloatFormat::Ieee(bits) => LoweredFloatFormat::Ieee(bits),
                FloatFormat::BrainFloat16 => LoweredFloatFormat::BrainFloat16,
                FloatFormat::Machine => {
                    LoweredFloatFormat::Ieee(context.target.machine_float_bits())
                }
            },
        },
        PrimitiveRepresentation::Boolean => LoweredType::Boolean,
        PrimitiveRepresentation::String => LoweredType::String,
        PrimitiveRepresentation::Bytes => LoweredType::Bytes,
        PrimitiveRepresentation::Character => LoweredType::Integer {
            bits: 32,
            signed: false,
        },
        PrimitiveRepresentation::None => LoweredType::None,
        PrimitiveRepresentation::Unit => LoweredType::Unit,
        PrimitiveRepresentation::Arguments => LoweredType::Arguments,
    })
}

fn invalid(message: impl Into<String>) -> CompileError {
    CompileError::InvalidArtifact(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_artifact::{ArtifactId, CompiledRegionId};
    use severian_compile::{
        CompileOperation, CompileRegionSpecialization, CompilerRegistry, EffectSet,
        RuntimeValueSpecialization, VerifiedCompiledRegionArtifact,
    };
    use severian_mir::{Value, ValueId};
    use severian_target::{Device, DeviceKind, FeatureSet, TargetSpec};
    use severian_universal::{install_primitives, Attrs, TypeContextBuilder};

    const ELEMENTS: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f8e4m3fn", "f8e5m2",
        "f16", "bf16", "f32", "f64", "f80", "f128",
    ];
    const FLOAT_ELEMENTS: &[&str] = &[
        "f8e4m3fn", "f8e5m2", "f16", "bf16", "f32", "f64", "f80", "f128",
    ];
    const NUMERIC_OPERATIONS: &[(tensor::TensorOp, usize)] = &[
        (tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add), 2),
        (
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Subtract),
            2,
        ),
        (
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Multiply),
            2,
        ),
        (
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Divide),
            2,
        ),
        (tensor::TensorOp::Matmul, 2),
    ];
    const FLOAT_OPERATIONS: &[(tensor::TensorOp, usize)] = &[
        (tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Exp), 1),
        (tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Log), 1),
        (
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Tanh),
            1,
        ),
        (
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Rsqrt),
            1,
        ),
    ];

    fn types() -> (TypeContext, TypeId) {
        let mut builder = TypeContextBuilder::new();
        install_primitives(&mut builder).unwrap();
        let mut types = builder.build();
        let constructor = types
            .register_source_declaration("tensor.Tensor", "Tensor", 1)
            .unwrap();
        types.mark_tensor_constructor(constructor).unwrap();
        (types, constructor)
    }

    fn region(tensor_type: TypeId, operation: tensor::TensorOp, operands: usize) -> CompileRegion {
        let mut attributes = Attrs::new();
        let operation_id = operation.apply(&mut attributes);
        CompileRegion {
            id: CompiledRegionId::new(0),
            compiler: tensor::compiler_id(),
            operations: Vec::new(),
            compile_operations: vec![CompileOperation {
                id: operation_id,
                operands: vec![tensor_type; operands],
                results: vec![tensor_type],
                operand_slots: (0..operands as u32).collect(),
                result_slots: vec![operands as u32],
                attributes,
            }],
            output_slots: vec![operands as u32],
            inputs: (0..operands)
                .map(|index| Value {
                    id: ValueId(index as u32),
                    type_id: tensor_type,
                })
                .collect(),
            outputs: vec![Value {
                id: ValueId(operands as u32),
                type_id: tensor_type,
            }],
            value_contracts: Vec::new(),
            effects: EffectSet::default(),
            placement: None,
        }
    }

    fn direct_region(
        input_types: &[TypeId],
        output_type: TypeId,
        operation: tensor::TensorOp,
    ) -> CompileRegion {
        let mut attributes = Attrs::new();
        let operation_id = operation.apply(&mut attributes);
        let result_slot = input_types.len() as u32;
        CompileRegion {
            id: CompiledRegionId::new(0),
            compiler: tensor::compiler_id(),
            operations: Vec::new(),
            compile_operations: vec![CompileOperation {
                id: operation_id,
                operands: input_types.to_vec(),
                results: vec![output_type],
                operand_slots: (0..result_slot).collect(),
                result_slots: vec![result_slot],
                attributes,
            }],
            output_slots: vec![result_slot],
            inputs: input_types
                .iter()
                .enumerate()
                .map(|(index, type_id)| Value {
                    id: ValueId(index as u32),
                    type_id: *type_id,
                })
                .collect(),
            outputs: vec![Value {
                id: ValueId(result_slot),
                type_id: output_type,
            }],
            value_contracts: Vec::new(),
            effects: EffectSet::default(),
            placement: None,
        }
    }

    fn compile_direct_and_verify(types: &TypeContext, region: &CompileRegion) -> MlirArtifact {
        let artifact = TensorCompiler
            .compile(
                region,
                &CompileContext {
                    types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap_or_else(|error| panic!("direct tensor region did not lower: {error}"));
        let CompiledRegionArtifact::CpuMlir(artifact) = artifact else {
            panic!("host tensor region unexpectedly selected the GPU route");
        };
        severian_mlir::verify_artifact(
            ArtifactId::for_region(region.id),
            artifact.clone(),
            &TargetSpec::host(),
        )
        .unwrap_or_else(|error| panic!("direct tensor region emitted invalid MLIR: {error}"));
        assert!(!artifact.module.contains("rank2"));
        assert!(!artifact.module.contains("rank4"));
        assert!(!artifact.module.contains("_bf16"));
        assert!(!artifact.module.contains("_f32"));
        artifact
    }

    #[test]
    fn legal_mlir_boundary_accepts_rank_two_bf16_add() {
        let (mut types, constructor) = types();
        let bf16 = types.resolve_name("bf16").unwrap();
        let matrix = types
            .instantiate_tensor(constructor, bf16, TensorShape::ranked([2, 4]))
            .unwrap();
        let region = direct_region(
            &[matrix, matrix],
            matrix,
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
        );
        let artifact = compile_direct_and_verify(&types, &region);
        assert!(artifact.module.contains("tensor<2x4xbf16>"));
    }

    #[test]
    fn legal_mlir_boundary_accepts_rank_four_f32_add() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3, 4, 5]))
            .unwrap();
        compile_direct_and_verify(
            &types,
            &direct_region(
                &[tensor_type, tensor_type],
                tensor_type,
                tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
            ),
        );
    }

    #[test]
    fn legal_mlir_boundary_accepts_rank_two_dynamic_dimension_add() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, f32, TensorShape::dynamic(2))
            .unwrap();
        let artifact = compile_direct_and_verify(
            &types,
            &direct_region(
                &[tensor_type, tensor_type],
                tensor_type,
                tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
            ),
        );
        assert!(artifact.module.contains("tensor<?x?xf32>"));
    }

    #[test]
    fn structured_elementwise_add_builds_rank_two_dynamic_bf16_mlir() {
        let (mut types, constructor) = types();
        let bf16 = types.resolve_name("bf16").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, bf16, TensorShape::dynamic(2))
            .unwrap();
        let artifact = compile_direct_and_verify(
            &types,
            &direct_region(
                &[tensor_type, tensor_type],
                tensor_type,
                tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
            ),
        );
        assert!(artifact
            .module
            .contains("func.func @entry(%arg0: tensor<?x?xbf16>, %arg1: tensor<?x?xbf16>)"));
        assert_eq!(artifact.module.matches("tensor.dim").count(), 2);
        assert!(artifact.module.contains("tensor.empty(%v2_d0, %v2_d1)"));
        assert_eq!(
            artifact
                .module
                .matches("affine_map<(d0, d1) -> (d0, d1)>")
                .count(),
            3
        );
        assert!(artifact
            .module
            .contains("iterator_types = [\"parallel\", \"parallel\"]"));
        assert!(artifact
            .module
            .contains("^bb0(%lhs: bf16, %rhs: bf16, %unused: bf16)"));
        assert!(artifact
            .module
            .contains("%computed = arith.addf %lhs, %rhs : bf16"));
        assert!(!artifact.module.contains("bf16_add"));
        assert!(!artifact.module.contains("rank2"));
    }

    #[test]
    fn structured_elementwise_add_uses_the_same_emitter_for_rank_four() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, f32, TensorShape::dynamic(4))
            .unwrap();
        let artifact = compile_direct_and_verify(
            &types,
            &direct_region(
                &[tensor_type, tensor_type],
                tensor_type,
                tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
            ),
        );
        assert_eq!(artifact.module.matches("tensor.dim").count(), 4);
        assert!(artifact
            .module
            .contains("iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"parallel\"]"));
        assert!(artifact.module.contains("arith.addf %lhs, %rhs : f32"));
    }

    #[test]
    fn structured_sum_reduces_all_axes_with_one_generic_reduction_builder() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let input = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([1]))
            .unwrap();
        let artifact = compile_direct_and_verify(
            &types,
            &direct_region(
                &[input],
                output,
                tensor::TensorOp::Reduce(tensor::ReductionOp::Sum),
            ),
        );
        assert!(artifact
            .module
            .contains("iterator_types = [\"reduction\", \"reduction\"]"));
        assert!(artifact.module.contains("affine_map<(d0, d1) -> (0)>"));
        assert!(artifact
            .module
            .contains("arith.addf %accumulator, %value : f32"));
    }

    #[test]
    fn structured_max_last_reuses_reduction_maps_for_rank_four_bf16() {
        let (mut types, constructor) = types();
        let bf16 = types.resolve_name("bf16").unwrap();
        let input = types
            .instantiate_tensor(constructor, bf16, TensorShape::ranked([2, 3, 4, 5]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, bf16, TensorShape::ranked([2, 3, 4]))
            .unwrap();
        let artifact = compile_direct_and_verify(
            &types,
            &direct_region(
                &[input],
                output,
                tensor::TensorOp::Reduce(tensor::ReductionOp::MaxLast),
            ),
        );
        assert!(artifact.module.contains("arith.constant 0xFF80 : bf16"));
        assert!(artifact.module.contains(
            "iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"reduction\"]"
        ));
        assert!(artifact
            .module
            .contains("arith.maximumf %accumulator, %value : bf16"));
    }

    #[test]
    fn structured_sum_axis_takes_axis_identity_from_ir_data() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let input = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2]))
            .unwrap();
        let mut region = direct_region(
            &[input],
            output,
            tensor::TensorOp::Reduce(tensor::ReductionOp::SumAxis),
        );
        region.compile_operations[0]
            .attributes
            .insert(tensor::REDUCTION_AXES, AttrValue::Integers(vec![1]));
        let artifact = compile_direct_and_verify(&types, &region);
        assert!(artifact.module.contains("affine_map<(d0, d1) -> (d0)>"));
        assert!(artifact
            .module
            .contains("iterator_types = [\"parallel\", \"reduction\"]"));
        assert!(!artifact.module.contains("sum_axis_f32"));
    }

    #[test]
    fn missing_structural_reduction_axes_fail_before_mlir_printing() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let input = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2]))
            .unwrap();
        let error = TensorCompiler
            .compile(
                &direct_region(
                    &[input],
                    output,
                    tensor::TensorOp::Reduce(tensor::ReductionOp::SumAxis),
                ),
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires structural reduction axis identities"));
        assert!(!error.contains("ParseFailed"));
        assert!(!error.contains("VerificationFailed"));
    }

    #[test]
    fn legal_mlir_boundary_accepts_mean_last_for_rank_two_bf16() {
        let (mut types, constructor) = types();
        let bf16 = types.resolve_name("bf16").unwrap();
        let input = types
            .instantiate_tensor(constructor, bf16, TensorShape::ranked([3, 4]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, bf16, TensorShape::ranked([3]))
            .unwrap();
        compile_direct_and_verify(
            &types,
            &direct_region(
                &[input],
                output,
                tensor::TensorOp::Reduce(tensor::ReductionOp::MeanLast),
            ),
        );
    }

    #[test]
    fn legal_mlir_boundary_accepts_mean_last_for_rank_four_f32() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let input = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3, 4, 5]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3, 4]))
            .unwrap();
        compile_direct_and_verify(
            &types,
            &direct_region(
                &[input],
                output,
                tensor::TensorOp::Reduce(tensor::ReductionOp::MeanLast),
            ),
        );
    }

    #[test]
    fn legal_mlir_boundary_accepts_dynamic_mean_last_dimensions() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let input = types
            .instantiate_tensor(constructor, f32, TensorShape::dynamic(2))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, f32, TensorShape::dynamic(1))
            .unwrap();
        let artifact = compile_direct_and_verify(
            &types,
            &direct_region(
                &[input],
                output,
                tensor::TensorOp::Reduce(tensor::ReductionOp::MeanLast),
            ),
        );
        assert!(artifact.module.contains("width_index = tensor.dim"));
        assert!(artifact.module.contains("tensor.empty(%v1_d0)"));
    }

    #[test]
    fn legal_mlir_boundary_accepts_rank_two_matmul() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let left = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3]))
            .unwrap();
        let right = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([3, 4]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 4]))
            .unwrap();
        compile_direct_and_verify(
            &types,
            &direct_region(&[left, right], output, tensor::TensorOp::Matmul),
        );
    }

    #[test]
    fn legal_mlir_boundary_accepts_rank_four_batched_matmul() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let left = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3, 4, 5]))
            .unwrap();
        let right = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3, 5, 6]))
            .unwrap();
        let output = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3, 4, 6]))
            .unwrap();
        compile_direct_and_verify(
            &types,
            &direct_region(&[left, right], output, tensor::TensorOp::Matmul),
        );
    }

    #[test]
    fn legal_mlir_boundary_accepts_dynamic_batched_matmul_dimensions() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let left = types
            .instantiate_tensor(
                constructor,
                f32,
                TensorShape::Ranked(vec![
                    TensorDimension::Dynamic,
                    TensorDimension::Known(3),
                    TensorDimension::Dynamic,
                    TensorDimension::Known(5),
                ]),
            )
            .unwrap();
        let right = types
            .instantiate_tensor(
                constructor,
                f32,
                TensorShape::Ranked(vec![
                    TensorDimension::Dynamic,
                    TensorDimension::Known(3),
                    TensorDimension::Known(5),
                    TensorDimension::Dynamic,
                ]),
            )
            .unwrap();
        let output = types
            .instantiate_tensor(
                constructor,
                f32,
                TensorShape::Ranked(vec![
                    TensorDimension::Dynamic,
                    TensorDimension::Known(3),
                    TensorDimension::Dynamic,
                    TensorDimension::Dynamic,
                ]),
            )
            .unwrap();
        let artifact = compile_direct_and_verify(
            &types,
            &direct_region(&[left, right], output, tensor::TensorOp::Matmul),
        );
        assert!(artifact
            .module
            .contains("tensor.empty(%result_dim0, %result_dim2, %result_dim3)"));
    }

    #[test]
    fn legal_mlir_boundary_rejects_unranked_elementwise_before_emission() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, f32, TensorShape::Unranked)
            .unwrap();
        let error = TensorCompiler
            .compile(
                &direct_region(
                    &[tensor_type, tensor_type],
                    tensor_type,
                    tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
                ),
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("MLIR legalization failed"), "{error}");
        assert!(error.contains("known tensor rank"), "{error}");
        assert!(!error.contains("lowering is not implemented"), "{error}");
    }

    #[test]
    fn runtime_specialization_ranks_unranked_values_before_mlir_emission() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, f32, TensorShape::Unranked)
            .unwrap();
        let mut region = direct_region(
            &[tensor_type, tensor_type],
            tensor_type,
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
        );
        region.rebuild_value_contracts(&types).unwrap();
        assert!(region.value_contracts.iter().all(|contract| matches!(
            contract.tensor.as_ref().map(|tensor| &tensor.rank),
            Some(severian_fusion::Rank::Unranked)
        )));
        let specialize = |slot| RuntimeValueSpecialization {
            slot,
            dimensions: vec![2, 3],
            strides: vec![3, 1],
            offset: 0,
        };
        let target = TargetSpec::host();
        let artifact = TensorCompiler
            .compile_specialized_cpu(
                &region,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
                &CompileRegionSpecialization {
                    values: vec![specialize(0), specialize(1), specialize(2)],
                },
            )
            .unwrap();
        assert!(artifact.module.contains("tensor<?x?xf32>"));
        assert!(!artifact.module.contains("tensor<*x"));
        assert!(region.value_contracts.iter().all(|contract| matches!(
            contract.tensor.as_ref().map(|tensor| &tensor.rank),
            Some(severian_fusion::Rank::Unranked)
        )));
        severian_mlir::verify_artifact(ArtifactId::for_region(region.id), artifact, &target)
            .unwrap();
    }

    #[test]
    fn compiler_registry_routes_runtime_specialization_without_handler_downcasts() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, f32, TensorShape::Unranked)
            .unwrap();
        let mut region = direct_region(
            &[tensor_type, tensor_type],
            tensor_type,
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
        );
        region.rebuild_value_contracts(&types).unwrap();
        let mut registry = CompilerRegistry::new();
        registry
            .register(tensor::compiler_id(), TensorCompiler)
            .unwrap();
        let target = TargetSpec::host();
        let artifact = registry
            .compile_specialized_region(
                &region,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
                &CompileRegionSpecialization {
                    values: (0..3)
                        .map(|slot| RuntimeValueSpecialization {
                            slot,
                            dimensions: vec![2, 3],
                            strides: vec![3, 1],
                            offset: 0,
                        })
                        .collect(),
                },
            )
            .unwrap();
        assert!(matches!(
            artifact,
            VerifiedCompiledRegionArtifact::CpuMlir(_)
        ));
    }

    #[test]
    fn runtime_specialization_does_not_confuse_rank_zero_with_unranked() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, f32, TensorShape::Unranked)
            .unwrap();
        let region = direct_region(
            &[tensor_type, tensor_type],
            tensor_type,
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
        );
        let (specialized, _) = region
            .specialize_for_emission(
                &types,
                &CompileRegionSpecialization {
                    values: (0..3)
                        .map(|slot| RuntimeValueSpecialization {
                            slot,
                            dimensions: Vec::new(),
                            strides: Vec::new(),
                            offset: 0,
                        })
                        .collect(),
                },
            )
            .unwrap();
        assert!(specialized.value_contracts.iter().all(|contract| {
            matches!(
                contract.tensor.as_ref().map(|tensor| &tensor.rank),
                Some(severian_fusion::Rank::Ranked(dimensions)) if dimensions.is_empty()
            )
        }));
    }

    #[test]
    fn storage_view_abi_is_validated_and_materialized_after_rank_specialization() {
        let (mut types, constructor) = types();
        let bf16 = types.resolve_name("bf16").unwrap();
        let descriptor = types.resolve_name("bytes").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, bf16, TensorShape::Unranked)
            .unwrap();
        let region = direct_region(
            &[descriptor],
            tensor_type,
            tensor::TensorOp::StorageView(tensor::StorageViewOp::FromAbi),
        );
        let target = TargetSpec::host();
        let artifact = TensorCompiler
            .compile_specialized_cpu(
                &region,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
                &CompileRegionSpecialization {
                    values: vec![RuntimeValueSpecialization {
                        slot: 1,
                        dimensions: vec![2, 3],
                        strides: vec![3, 1],
                        offset: 0,
                    }],
                },
            )
            .unwrap();
        assert!(artifact
            .module
            .contains("@__sev_storage_view_validate(!llvm.ptr, i32, i32, i32, i64) -> i32"));
        assert!(artifact
            .module
            .contains("%expected_kind = arith.constant 3 : i32"));
        assert!(artifact
            .module
            .contains("%expected_bits = arith.constant 16 : i32"));
        assert!(artifact
            .module
            .contains("%expected_float_format = arith.constant 2 : i32"));
        assert!(artifact
            .module
            .contains("llvm.load %address : !llvm.ptr -> bf16"));
        assert!(!artifact.module.contains("__sev_safetensor_bf16"));
        severian_mlir::verify_artifact(ArtifactId::for_region(region.id), artifact, &target)
            .unwrap();
    }

    fn compile_and_verify(
        types: &TypeContext,
        tensor_type: TypeId,
        operation: tensor::TensorOp,
        operands: usize,
    ) -> MlirArtifact {
        let artifact = TensorCompiler
            .compile(
                &region(tensor_type, operation, operands),
                &CompileContext {
                    types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap_or_else(|error| panic!("{operation:?} did not lower: {error}"));
        let CompiledRegionArtifact::CpuMlir(artifact) = artifact else {
            panic!("host tensor operation unexpectedly selected the GPU route");
        };
        severian_mlir::verify_artifact(
            ArtifactId::for_region(CompiledRegionId::new(0)),
            artifact.clone(),
            &TargetSpec::host(),
        )
        .unwrap_or_else(|error| panic!("{operation:?} emitted invalid MLIR: {error}"));
        artifact
    }

    #[test]
    fn every_primitive_width_uses_one_generic_numeric_lowering() {
        let (mut types, constructor) = types();
        for name in ELEMENTS {
            let element = types.resolve_name(name).unwrap();
            let tensor_type = types
                .instantiate_tensor(constructor, element, TensorShape::ranked([2, 2]))
                .unwrap();
            let element = match *name {
                "u8" => "i8",
                "u16" => "i16",
                "u32" => "i32",
                "u64" => "i64",
                "u128" => "i128",
                "f8e4m3fn" => "f8E4M3FN",
                "f8e5m2" => "f8E5M2",
                other => other,
            };
            for &(operation, operands) in NUMERIC_OPERATIONS {
                let artifact = compile_and_verify(&types, tensor_type, operation, operands);
                assert!(
                    artifact.module.contains(&format!("tensor<2x2x{element}>")),
                    "{name} was erased in:\n{}",
                    artifact.module
                );
            }
            for operation in [
                tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Materialize),
                tensor::TensorOp::Permute(tensor::PermuteOp::Reverse),
            ] {
                let artifact = compile_and_verify(&types, tensor_type, operation, 1);
                assert!(artifact.module.contains(&format!("tensor<2x2x{element}>")));
            }
        }
    }

    #[test]
    fn every_float_width_uses_one_generic_transcendental_lowering() {
        let (mut types, constructor) = types();
        for name in FLOAT_ELEMENTS {
            let element = types.resolve_name(name).unwrap();
            let tensor_type = types
                .instantiate_tensor(constructor, element, TensorShape::ranked([2, 2]))
                .unwrap();
            for &(operation, operands) in FLOAT_OPERATIONS {
                compile_and_verify(&types, tensor_type, operation, operands);
            }
        }
    }

    #[test]
    fn dynamic_rank_and_dynamic_dimensions_have_distinct_mlir_types() {
        let (mut types, constructor) = types();
        let i32 = types.resolve_name("i32").unwrap();
        let unranked = types
            .instantiate_tensor(constructor, i32, TensorShape::Unranked)
            .unwrap();
        let dynamic_rank_two = types
            .instantiate_tensor(constructor, i32, TensorShape::dynamic(2))
            .unwrap();
        let target = TargetSpec::host();
        let context = CompileContext {
            types: &types,
            target: &target,
        };
        assert_eq!(
            type_spelling(&lower_type(unranked, &context).unwrap()).unwrap(),
            "tensor<*xi32>"
        );
        assert_eq!(
            type_spelling(&lower_type(dynamic_rank_two, &context).unwrap()).unwrap(),
            "tensor<?x?xi32>"
        );
    }

    #[test]
    fn unranked_shape_is_materialized_from_runtime_tensor_rank() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let storage = types.resolve_name("string").unwrap();
        let unranked = types
            .instantiate_tensor(constructor, f32, TensorShape::Unranked)
            .unwrap();
        let mut attributes = Attrs::new();
        let operation = tensor::TensorOp::StorageView(tensor::StorageViewOp::Shape);
        let region = CompileRegion {
            id: CompiledRegionId::new(0),
            compiler: tensor::compiler_id(),
            operations: Vec::new(),
            compile_operations: vec![CompileOperation {
                id: operation.apply(&mut attributes),
                operands: vec![unranked],
                results: vec![storage],
                operand_slots: vec![0],
                result_slots: vec![1],
                attributes,
            }],
            output_slots: vec![1],
            inputs: vec![Value {
                id: ValueId(0),
                type_id: unranked,
            }],
            outputs: vec![Value {
                id: ValueId(1),
                type_id: storage,
            }],
            value_contracts: Vec::new(),
            effects: EffectSet::default(),
            placement: None,
        };
        let artifact = TensorCompiler
            .compile(
                &region,
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap();
        let CompiledRegionArtifact::CpuMlir(artifact) = artifact else {
            panic!("host tensor operation unexpectedly selected the GPU route");
        };
        assert!(artifact.module.contains("tensor.rank %arg0"));
        assert!(artifact.module.contains("tensor.dim %arg0, %axis"));
        assert!(artifact.module.contains("scf.for %axis"));
    }

    #[test]
    fn broadcasting_changes_shape_without_changing_element_type() {
        let (mut types, constructor) = types();
        let f32 = types.resolve_name("f32").unwrap();
        let matrix = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3]))
            .unwrap();
        let row = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([3]))
            .unwrap();
        let result = types
            .instantiate_tensor(constructor, f32, TensorShape::ranked([2, 3]))
            .unwrap();
        let mut operation = region(
            result,
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
            2,
        );
        operation.compile_operations[0].operands = vec![matrix, row];
        operation.inputs[0].type_id = matrix;
        operation.inputs[1].type_id = row;
        let artifact = TensorCompiler
            .compile(
                &operation,
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap();
        let CompiledRegionArtifact::CpuMlir(artifact) = artifact else {
            panic!("host tensor operation unexpectedly selected the GPU route");
        };
        assert!(artifact.module.contains("tensor<2x3xf32>"));
        assert!(artifact.module.contains("tensor<3xf32>"));
        severian_mlir::verify_artifact(
            ArtifactId::for_region(CompiledRegionId::new(0)),
            artifact,
            &TargetSpec::host(),
        )
        .unwrap();
    }

    #[test]
    fn rectangular_matmul_uses_structural_operand_and_result_shapes() {
        let (mut types, constructor) = types();
        let f64 = types.resolve_name("f64").unwrap();
        let left = types
            .instantiate_tensor(constructor, f64, TensorShape::ranked([2, 3]))
            .unwrap();
        let right = types
            .instantiate_tensor(constructor, f64, TensorShape::ranked([3, 4]))
            .unwrap();
        let result = types
            .instantiate_tensor(constructor, f64, TensorShape::ranked([2, 4]))
            .unwrap();
        let mut operation = region(result, tensor::TensorOp::Matmul, 2);
        operation.compile_operations[0].operands = vec![left, right];
        operation.inputs[0].type_id = left;
        operation.inputs[1].type_id = right;
        let artifact = TensorCompiler
            .compile(
                &operation,
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap();
        let CompiledRegionArtifact::CpuMlir(artifact) = artifact else {
            panic!("host tensor operation unexpectedly selected the GPU route");
        };
        for spelling in ["tensor<2x3xf64>", "tensor<3x4xf64>", "tensor<2x4xf64>"] {
            assert!(artifact.module.contains(spelling));
        }
        severian_mlir::verify_artifact(
            ArtifactId::for_region(CompiledRegionId::new(0)),
            artifact,
            &TargetSpec::host(),
        )
        .unwrap();
    }

    #[test]
    fn rms_norm_is_a_graph_of_primitives_not_a_compiler_operation() {
        let (mut types, constructor) = types();
        let element = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, element, TensorShape::ranked([2, 4]))
            .unwrap();
        let mut region = region(
            tensor_type,
            tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Multiply),
            2,
        );
        let mut rsqrt_attributes = Attrs::new();
        let rsqrt = tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Rsqrt);
        region.compile_operations.push(CompileOperation {
            id: rsqrt.apply(&mut rsqrt_attributes),
            operands: vec![tensor_type],
            results: vec![tensor_type],
            operand_slots: vec![2],
            result_slots: vec![3],
            attributes: rsqrt_attributes,
        });
        region.output_slots = vec![3];
        region.outputs[0].type_id = tensor_type;
        let graph = fusion_graph(&region, &types).unwrap();
        assert_eq!(graph.nodes().len(), 4);
        assert_eq!(
            graph.node(severian_fusion::NodeId(0)).shape.element_kind,
            severian_fusion::ElementKind::IeeeFloat
        );
        assert_eq!(graph.node(severian_fusion::NodeId(2)).operation, "multiply");
        assert_eq!(graph.node(severian_fusion::NodeId(3)).operation, "rsqrt");
    }

    #[test]
    fn gpu_unranked_permute_bypasses_the_cpu_mlir_emitter() {
        let (mut types, constructor) = types();
        let element = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, element, TensorShape::Unranked)
            .unwrap();
        let mut region = region(
            tensor_type,
            tensor::TensorOp::Permute(tensor::PermuteOp::Axes),
            1,
        );
        region.placement = Some(ExecutionPlacement::Gpu);
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.devices.push(Device {
            name: "gpu0".into(),
            kind: DeviceKind::Gpu,
            architecture: "gfx1100".into(),
            features: FeatureSet::from_names(["vendor.amd"]),
        });

        let artifact = TensorCompiler
            .compile(
                &region,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
            )
            .unwrap();
        let CompiledRegionArtifact::GpuKernel(bundle) = artifact else {
            panic!("GPU placement entered the CPU MLIR route");
        };
        assert_eq!(bundle.target, GpuTarget::Amd);
        assert_eq!(bundle.graph.nodes().len(), 2);
        assert_eq!(bundle.plan.node_regions.len(), bundle.graph.nodes().len());
        assert_eq!(
            bundle.graph.node(severian_fusion::NodeId(1)).kind,
            severian_fusion::NodeKind::Permute
        );
    }

    #[test]
    fn gpu_storage_descriptor_becomes_a_typed_parameter_not_ttir_parsing() {
        let (mut types, constructor) = types();
        let bf16 = types.resolve_name("bf16").unwrap();
        let descriptor = types.resolve_name("bytes").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, bf16, TensorShape::Unranked)
            .unwrap();
        let mut region = direct_region(
            &[descriptor],
            tensor_type,
            tensor::TensorOp::StorageView(tensor::StorageViewOp::FromAbi),
        );
        let mut attributes = Attrs::new();
        let add = tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add);
        region.compile_operations.push(CompileOperation {
            id: add.apply(&mut attributes),
            operands: vec![tensor_type, tensor_type],
            results: vec![tensor_type],
            operand_slots: vec![1, 1],
            result_slots: vec![2],
            attributes,
        });
        region.output_slots = vec![2];
        region.outputs[0].id = ValueId(2);
        region.placement = Some(ExecutionPlacement::Gpu);
        region.rebuild_value_contracts(&types).unwrap();
        let mut target = TargetSpec::new("x86_64-unknown-linux");
        target.devices.push(Device {
            name: "gpu0".into(),
            kind: DeviceKind::Gpu,
            architecture: "sm_80".into(),
            features: FeatureSet::from_names(["vendor.nvidia"]),
        });
        let artifact = TensorCompiler
            .compile(
                &region,
                &CompileContext {
                    types: &types,
                    target: &target,
                },
            )
            .unwrap();
        let CompiledRegionArtifact::GpuKernel(bundle) = artifact else {
            panic!("GPU storage descriptor entered CPU MLIR")
        };
        assert_eq!(bundle.inputs, [LoweredType::Bytes]);
        assert_eq!(bundle.graph.nodes().len(), 2);
        assert_eq!(
            bundle.value_nodes.get(&0),
            Some(&severian_fusion::NodeId(0))
        );
        assert_eq!(
            bundle.value_nodes.get(&1),
            Some(&severian_fusion::NodeId(0))
        );
        assert_eq!(
            bundle.value_nodes.get(&2),
            Some(&severian_fusion::NodeId(1))
        );
        let parameter = bundle.graph.node(severian_fusion::NodeId(0));
        assert_eq!(parameter.kind, severian_fusion::NodeKind::Parameter);
        assert_eq!(
            parameter.shape.element_kind,
            severian_fusion::ElementKind::BrainFloat
        );
        assert_eq!(parameter.shape.element_bits, 16);
        assert_eq!(parameter.layout, severian_fusion::StorageLayout::Runtime);
        assert_eq!(
            bundle.graph.node(severian_fusion::NodeId(1)).kind,
            severian_fusion::NodeKind::Elementwise
        );
        let specialization = bundle
            .compile_region_specialization(&severian_fusion::KernelSpecialization {
                shapes: vec![
                    severian_fusion::RuntimeShape {
                        node: severian_fusion::NodeId(0),
                        dimensions: vec![2, 4],
                    },
                    severian_fusion::RuntimeShape {
                        node: severian_fusion::NodeId(1),
                        dimensions: vec![2, 4],
                    },
                ],
                strides: vec![
                    severian_fusion::RuntimeStrides {
                        node: severian_fusion::NodeId(0),
                        strides: vec![4, 1],
                        offset: 0,
                    },
                    severian_fusion::RuntimeStrides {
                        node: severian_fusion::NodeId(1),
                        strides: vec![4, 1],
                        offset: 0,
                    },
                ],
                target: severian_fusion::GpuTarget::Nvidia,
            })
            .unwrap();
        assert_eq!(
            specialization
                .values
                .iter()
                .map(|value| (value.slot, value.dimensions.clone()))
                .collect::<Vec<_>>(),
            [(0, vec![2, 4]), (1, vec![2, 4]), (2, vec![2, 4])]
        );
    }

    #[test]
    fn gpu_graph_dtype_is_data_not_an_operation_identity_or_symbol_suffix() {
        let (mut types, constructor) = types();
        let expected = [
            ("f16", severian_fusion::ElementKind::IeeeFloat, 16),
            ("bf16", severian_fusion::ElementKind::BrainFloat, 16),
            ("f32", severian_fusion::ElementKind::IeeeFloat, 32),
        ];
        let mut identities = Vec::new();
        for (name, element_kind, element_bits) in expected {
            let element = types.resolve_name(name).unwrap();
            let tensor_type = types
                .instantiate_tensor(constructor, element, TensorShape::ranked([2, 4]))
                .unwrap();
            let region = region(
                tensor_type,
                tensor::TensorOp::Elementwise(tensor::ElementwiseOp::Add),
                2,
            );
            identities.push(region.compile_operations[0].id);
            let graph = fusion_graph(&region, &types).unwrap();
            let add = graph.node(severian_fusion::NodeId(2));
            assert_eq!(add.operation, "add");
            assert_eq!(add.shape.element_kind, element_kind);
            assert_eq!(add.shape.element_bits, element_bits);
            assert!(!add.operation.contains(name));
        }
        assert!(identities
            .iter()
            .all(|identity| *identity == tensor::ELEMENTWISE));
    }

    #[test]
    fn gpu_graph_matmul_rank_is_data_not_an_operation_identity() {
        let (mut types, constructor) = types();
        let element = types.resolve_name("f16").unwrap();
        let shapes = [
            TensorShape::ranked([4, 4]),
            TensorShape::ranked([2, 3, 4, 4]),
        ];
        let mut identities = Vec::new();
        for (expected_rank, shape) in [2, 4].into_iter().zip(shapes) {
            let tensor_type = types
                .instantiate_tensor(constructor, element, shape)
                .unwrap();
            let region = region(tensor_type, tensor::TensorOp::Matmul, 2);
            identities.push(region.compile_operations[0].id);
            let graph = fusion_graph(&region, &types).unwrap();
            let matmul = graph.node(severian_fusion::NodeId(2));
            assert_eq!(matmul.kind, severian_fusion::NodeKind::Contraction);
            assert_eq!(matmul.operation, "matmul");
            assert_eq!(matmul.shape.dimensions().unwrap().len(), expected_rank);
            assert!(!matmul.operation.contains(&format!("rank{expected_rank}")));
            let contract = matmul.matmul.as_ref().unwrap();
            assert_eq!(contract.contraction_dimensions.len(), 1);
            assert_eq!(
                contract.contraction_dimensions[0],
                severian_fusion::ContractionDimension {
                    lhs: (expected_rank - 1) as u32,
                    rhs: (expected_rank - 2) as u32,
                }
            );
            assert_eq!(contract.batch_dimensions.len(), expected_rank - 2);
            assert!(contract.batch_dimensions.iter().enumerate().all(
                |(axis, dimension)| dimension.result == axis as u32
                    && dimension.lhs == Some(axis as u32)
                    && dimension.rhs == Some(axis as u32)
            ));
        }
        assert_eq!(identities, [tensor::MATMUL, tensor::MATMUL]);
    }

    #[test]
    fn gpu_graph_preserves_runtime_shape_alias_and_mutation_contracts() {
        let (mut types, constructor) = types();
        let element = types.resolve_name("f32").unwrap();
        let tensor_type = types
            .instantiate_tensor(constructor, element, TensorShape::ranked([2, 4]))
            .unwrap();

        let mut reshape_region = region(
            tensor_type,
            tensor::TensorOp::ReshapeView(tensor::ReshapeViewOp::Reshape),
            2,
        );
        reshape_region.rebuild_value_contracts(&types).unwrap();
        let reshape_contract = reshape_region
            .value_contracts
            .iter()
            .find(|contract| contract.slot == 2)
            .and_then(|contract| contract.tensor.as_ref())
            .unwrap();
        assert_eq!(reshape_contract.runtime_shape_operands, [1]);
        assert_eq!(reshape_contract.aliases[0].source_slot, 0);
        let reshape_graph = fusion_graph(&reshape_region, &types).unwrap();
        let reshape = reshape_graph.node(severian_fusion::NodeId(2));
        assert_eq!(
            reshape.operand_roles,
            [
                severian_fusion::OperandRole::Data,
                severian_fusion::OperandRole::RuntimeShape,
            ]
        );
        assert_eq!(
            reshape.runtime_shape_inputs().collect::<Vec<_>>(),
            [severian_fusion::NodeId(1)]
        );
        assert_eq!(
            reshape.aliases,
            [severian_fusion::InputAlias {
                input_index: 0,
                kind: severian_fusion::AliasKind::View,
            }]
        );
        assert_eq!(reshape.layout, severian_fusion::StorageLayout::Runtime);

        let mut slice_region = region(tensor_type, tensor::TensorOp::Slice, 4);
        slice_region.rebuild_value_contracts(&types).unwrap();
        let slice_graph = fusion_graph(&slice_region, &types).unwrap();
        let slice = slice_graph.node(severian_fusion::NodeId(4));
        assert_eq!(
            slice.operand_roles,
            [
                severian_fusion::OperandRole::Data,
                severian_fusion::OperandRole::RuntimeShape,
                severian_fusion::OperandRole::RuntimeShape,
                severian_fusion::OperandRole::RuntimeShape,
            ]
        );
        assert_eq!(slice.aliases[0].kind, severian_fusion::AliasKind::View);
        assert!(matches!(
            &slice.layout,
            severian_fusion::StorageLayout::Strided { strides, offset }
                if strides == &[severian_fusion::Stride::Dynamic; 2]
                    && *offset == severian_fusion::Stride::Dynamic
        ));

        let mut scatter_region = region(tensor_type, tensor::TensorOp::Scatter, 3);
        scatter_region.rebuild_value_contracts(&types).unwrap();
        let scatter_graph = fusion_graph(&scatter_region, &types).unwrap();
        let scatter = scatter_graph.node(severian_fusion::NodeId(3));
        assert_eq!(scatter.aliases[0].kind, severian_fusion::AliasKind::InPlace);
        assert_eq!(scatter.mutation, severian_fusion::Mutation::WritesInput(0));
    }
}
