#![forbid(unsafe_code)]

mod fusion;

pub use fusion::{fusion_graph, FusionGraphError};

use severian_compile::{
    CompileContext, CompileError, CompileHandler, CompileRegion, CompiledRegionArtifact,
    GpuKernelBundle, GpuTarget,
};
use severian_fusion::{plan as plan_fusion, DeviceModel};
use severian_mlir::{
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
}

impl TensorCompiler {
    fn compile_cpu(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
    ) -> Result<MlirArtifact, CompileError> {
        if region.compile_operations.is_empty() {
            return Err(invalid("the tensor compiler requires a non-empty region"));
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
            helpers.push_str(&format!(
                "  func.func private @__sev_tensor_op_{index}({helper_parameters}){helper_result} {{\n{helper_body}  }}\n"
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
                "    {result_names} = func.call @__sev_tensor_op_{index}({arguments}) : ({input_types}){helper_result}\n",
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
        let device = context
            .target
            .triton_gpu()
            .ok_or_else(|| {
                CompileError::Target(
                    "GPU placement requires an AMD or NVIDIA device in the target".into(),
                )
            })?;
        let target = if device.features.contains("vendor.amd")
            || device.architecture.starts_with("gfx")
        {
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
        let graph = fusion_graph(region, context.types)
            .map_err(|error| CompileError::InvalidArtifact(error.to_string()))?;
        let plan = plan_fusion(&graph, DeviceModel::conservative_gpu());
        Ok(GpuKernelBundle {
            target,
            architecture: device.architecture.clone(),
            graph,
            plan,
            inputs,
            outputs,
        })
    }
}

fn operation_declarations(
    operation: tensor::TensorOp,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
) -> Result<String, CompileError> {
    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::FromElements) {
        return Ok("  func.func private @__sev_list_get_f64(!llvm.ptr, i64) -> f64\n".into());
    }
    if operation == tensor::TensorOp::StorageView(tensor::StorageViewOp::Values) {
        let [LoweredType::Tensor { element, .. }] = inputs else {
            return Err(invalid("values requires one structural tensor operand"));
        };
        let scalar = tensor_element_spelling(*element)?;
        return Ok(format!(
            "  func.func private @__sev_list_create() -> !llvm.ptr\n  func.func private @__sev_list_push_f64(!llvm.ptr, {scalar})\n"
        ));
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
    if operation == tensor::TensorOp::Convert {
        return Ok(
            "  func.func private @__sev_f8e4m3fn_to_f32(i8) -> f32\n  func.func private @__sev_f32_to_f8e4m3fn(f32) -> i8\n  func.func private @__sev_f8e5m2_to_f32(i8) -> f32\n  func.func private @__sev_f32_to_f8e5m2(f32) -> i8\n"
                .into(),
        );
    }
    let _ = outputs;
    Ok(String::new())
}

fn lower_operation(
    operation: tensor::TensorOp,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
    input_spellings: &[String],
    output_spellings: &[String],
    attributes: &Attrs,
) -> Result<String, CompileError> {
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
        if scalar != "f64" {
            return Err(invalid(
                "values currently returns list[float] and requires f64",
            ));
        }
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
                "    %element{ordinal} = tensor.extract {operand}[{}] : {ranked_type}\n    func.call @__sev_list_push_f64(%result, %element{ordinal}) : (!llvm.ptr, {scalar}) -> ()\n",
                indices.join(", ")
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
            return Ok(format!(
                "    %empty = tensor.empty() : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({left_map})>, affine_map<({loops}) -> ({right_map})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterators}]}} ins(%arg0, %arg1 : {left_type}, {right_type}) outs(%empty : {output}) {{\n    ^bb0(%left: {scalar}, %right: {scalar}, %unused: {scalar}):\n      %value = {instruction} %left, %right : {scalar}\n      linalg.yield %value : {scalar}\n    }} -> {output}\n    return %result : {output}\n",
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
        return Ok(format!(
            "    %empty = tensor.empty() : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({input_map})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterator_types}]}} ins(%arg0 : {input}) outs(%empty : {output}) {{\n    ^bb0(%element: {scalar}, %unused: {scalar}):\n      linalg.yield %element : {scalar}\n    }} -> {output}\n    return %result : {output}\n",
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
        // Dynamic broadcast selection requires runtime indexing and is a
        // separate lowering concern. Rank and static batch shape are already
        // fully generic here.
        static_dimensions(left_shape).map_err(|error| invalid(format!("matmul: {error}")))?;
        static_dimensions(right_shape).map_err(|error| invalid(format!("matmul: {error}")))?;
        static_dimensions(&shape).map_err(|error| invalid(format!("matmul: {error}")))?;
        let scalar = tensor_element_spelling(*element)?;
        let zero = if is_float(*element) { "0.0" } else { "0" };
        let ranked_type = type_spelling(&LoweredType::Tensor {
            element: *element,
            shape: shape.clone(),
        })
        .map_err(|error| invalid(error.to_string()))?;
        let output_rank = output_dimensions.len();
        let batch_rank = output_rank - 2;
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
            "    %empty = tensor.empty() : {ranked_type}\n    %zero = arith.constant {zero} : {scalar}\n    %initialized = linalg.fill ins(%zero : {scalar}) outs(%empty : {ranked_type}) -> {ranked_type}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({left_map})>, affine_map<({loops}) -> ({right_map})>, affine_map<({loops}) -> ({output_map})>], iterator_types = [{iterators}]}} ins(%arg0, %arg1 : {left_type}, {right_type}) outs(%initialized : {ranked_type}) {{\n    ^bb0(%left: {scalar}, %right: {scalar}, %acc: {scalar}):\n      %product = {multiply} %left, %right : {scalar}\n      %sum = {add} %acc, %product : {scalar}\n      linalg.yield %sum : {scalar}\n    }} -> {ranked_type}\n    return %result : {output}\n",
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
    if source_shape != target_shape {
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
    Ok(format!(
        "    %empty = tensor.empty() : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({loops})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterators}]}} ins(%arg0 : {input}) outs(%empty : {output}) {{\n    ^bb0(%value: {source_scalar}, %unused: {target_scalar}):\n{conversion}      linalg.yield %converted : {target_scalar}\n    }} -> {output}\n    return %result : {output}\n"
    ))
}

#[allow(dead_code)]
fn lower_rms_norm(
    inputs: &[LoweredType],
    result_type: &LoweredType,
    input_spellings: &[String],
    output: &str,
    result_element: LoweredTensorElement,
) -> Result<String, CompileError> {
    let [LoweredType::Tensor {
        element: input_element,
        shape: LoweredTensorShape::Ranked(input_shape),
    }, LoweredType::Tensor {
        element: weight_element,
        shape: LoweredTensorShape::Ranked(weight_shape),
    }, LoweredType::Float {
        format: epsilon_format,
    }] = inputs
    else {
        return Err(invalid(
            "RMSNorm requires a ranked input, ranked weights, and floating epsilon",
        ));
    };
    let LoweredType::Tensor {
        element: output_element,
        shape: LoweredTensorShape::Ranked(output_shape),
    } = result_type
    else {
        return Err(invalid("RMSNorm requires a ranked tensor result"));
    };
    if input_element != weight_element
        || input_element != output_element
        || input_element != &result_element
    {
        return Err(invalid(
            "RMSNorm input, weights, and result must have one element type",
        ));
    }
    if input_shape.is_empty() || output_shape != input_shape {
        return Err(invalid("RMSNorm must preserve a non-scalar input shape"));
    }
    if weight_shape.len() != 1 || weight_shape[0] != input_shape[input_shape.len() - 1] {
        return Err(invalid(
            "RMSNorm weights must match the input's last dimension",
        ));
    }
    let dimensions = static_dimensions(&LoweredTensorShape::Ranked(input_shape.clone()))
        .map_err(|error| invalid(format!("RMSNorm: {error}")))?;
    let width = *dimensions
        .last()
        .ok_or_else(|| invalid("RMSNorm input must have a last dimension"))?;
    if width == 0 {
        return Err(invalid("RMSNorm last dimension must not be empty"));
    }
    let storage_scalar = tensor_element_spelling(result_element)?;
    if matches!(result_element, LoweredTensorElement::Boolean) {
        return Err(invalid("RMSNorm requires a numeric tensor element type"));
    }
    let accumulation_element = rms_accumulation_element(result_element);
    let accumulation_scalar = tensor_element_spelling(accumulation_element)?;
    let outer_shape = LoweredTensorShape::Ranked(input_shape[..input_shape.len() - 1].to_vec());
    let outer_type = type_spelling(&LoweredType::Tensor {
        element: accumulation_element,
        shape: outer_shape,
    })
    .map_err(|error| invalid(error.to_string()))?;
    let rank = input_shape.len();
    let loops = (0..rank).map(|axis| format!("d{axis}")).collect::<Vec<_>>();
    let outer = loops[..rank - 1].join(", ");
    let identity = loops.join(", ");
    let iterators = (0..rank)
        .map(|axis| {
            if axis + 1 == rank {
                "\"reduction\""
            } else {
                "\"parallel\""
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let epsilon_source = input_spellings
        .get(2)
        .ok_or_else(|| invalid("RMSNorm epsilon type is missing"))?;
    let epsilon = lower_scalar_conversion(
        "%arg2",
        LoweredTensorElement::Float {
            format: *epsilon_format,
        },
        accumulation_element,
        epsilon_source,
        &accumulation_scalar,
        "%epsilon",
    )?;
    let value_to_accumulation = lower_scalar_conversion(
        "%value",
        result_element,
        accumulation_element,
        &storage_scalar,
        &accumulation_scalar,
        "%value_compute",
    )?;
    let weight_to_accumulation = lower_scalar_conversion(
        "%weight",
        result_element,
        accumulation_element,
        &storage_scalar,
        &accumulation_scalar,
        "%weight_compute",
    )?;
    let output_from_accumulation = lower_scalar_conversion(
        "%weighted",
        accumulation_element,
        result_element,
        &accumulation_scalar,
        &storage_scalar,
        "%weighted_output",
    )?;
    let input_type = &input_spellings[0];
    let weight_type = &input_spellings[1];
    Ok(format!(
        "    %sum_empty = tensor.empty() : {outer_type}\n    %zero = arith.constant 0.0 : {accumulation_scalar}\n    %sum_initial = linalg.fill ins(%zero : {accumulation_scalar}) outs(%sum_empty : {outer_type}) -> {outer_type}\n    %sum = linalg.generic {{indexing_maps = [affine_map<({identity}) -> ({identity})>, affine_map<({identity}) -> ({outer})>], iterator_types = [{iterators}]}} ins(%arg0 : {input_type}) outs(%sum_initial : {outer_type}) {{\n    ^bb0(%value: {storage_scalar}, %accumulator: {accumulation_scalar}):\n{value_to_accumulation}      %square = arith.mulf %value_compute, %value_compute : {accumulation_scalar}\n      %next = arith.addf %accumulator, %square : {accumulation_scalar}\n      linalg.yield %next : {accumulation_scalar}\n    }} -> {outer_type}\n{epsilon}    %inverse_empty = tensor.empty() : {outer_type}\n    %width = arith.constant {width}.0 : {accumulation_scalar}\n    %inverse = linalg.generic {{indexing_maps = [affine_map<({outer}) -> ({outer})>, affine_map<({outer}) -> ({outer})>], iterator_types = [{outer_iterators}]}} ins(%sum : {outer_type}) outs(%inverse_empty : {outer_type}) {{\n    ^bb0(%total: {accumulation_scalar}, %unused: {accumulation_scalar}):\n      %mean = arith.divf %total, %width : {accumulation_scalar}\n      %stabilized = arith.addf %mean, %epsilon : {accumulation_scalar}\n      %factor = math.rsqrt %stabilized : {accumulation_scalar}\n      linalg.yield %factor : {accumulation_scalar}\n    }} -> {outer_type}\n    %output_empty = tensor.empty() : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({identity}) -> ({identity})>, affine_map<({identity}) -> ({outer})>, affine_map<({identity}) -> (d{last})>, affine_map<({identity}) -> ({identity})>], iterator_types = [{output_iterators}]}} ins(%arg0, %inverse, %arg1 : {input_type}, {outer_type}, {weight_type}) outs(%output_empty : {output}) {{\n    ^bb0(%value: {storage_scalar}, %factor: {accumulation_scalar}, %weight: {storage_scalar}, %unused: {storage_scalar}):\n{value_to_accumulation}{weight_to_accumulation}      %normalized = arith.mulf %value_compute, %factor : {accumulation_scalar}\n      %weighted = arith.mulf %normalized, %weight_compute : {accumulation_scalar}\n{output_from_accumulation}      linalg.yield %weighted_output : {storage_scalar}\n    }} -> {output}\n    return %result : {output}\n",
        outer_iterators = vec!["\"parallel\""; rank - 1].join(", "),
        output_iterators = vec!["\"parallel\""; rank].join(", "),
        last = rank - 1,
    ))
}

#[allow(dead_code)]
const fn rms_accumulation_element(element: LoweredTensorElement) -> LoweredTensorElement {
    match element {
        LoweredTensorElement::Float {
            format:
                LoweredFloatFormat::Float8E4M3Fn
                | LoweredFloatFormat::Float8E5M2
                | LoweredFloatFormat::Ieee(16)
                | LoweredFloatFormat::BrainFloat16,
        } => LoweredTensorElement::Float {
            format: LoweredFloatFormat::Ieee(32),
        },
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Ieee(80 | 128),
        } => LoweredTensorElement::Float {
            format: LoweredFloatFormat::Ieee(64),
        },
        LoweredTensorElement::Integer { .. } => LoweredTensorElement::Float {
            format: LoweredFloatFormat::Ieee(64),
        },
        other => other,
    }
}

fn lower_scalar_conversion(
    value: &str,
    source: LoweredTensorElement,
    target: LoweredTensorElement,
    source_type: &str,
    target_type: &str,
    result: &str,
) -> Result<String, CompileError> {
    if is_fp8(source) || is_fp8(target) {
        return lower_fp8_conversion(value, source, target, source_type, target_type, result);
    }
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

fn lower_fp8_conversion(
    value: &str,
    source: LoweredTensorElement,
    target: LoweredTensorElement,
    source_type: &str,
    target_type: &str,
    result: &str,
) -> Result<String, CompileError> {
    let mut body = String::new();
    let f32_value = if let Some(symbol) = fp8_decode_symbol(source) {
        body.push_str(&format!(
            "    {result}_source_bits = arith.bitcast {value} : {source_type} to i8\n    {result}_f32 = func.call @{symbol}({result}_source_bits) : (i8) -> f32\n"
        ));
        format!("{}_f32", result.trim_start_matches('%'))
    } else {
        let operation = match source {
            LoweredTensorElement::Float { format } => {
                if float_format_bits(format) < 32 {
                    "arith.extf"
                } else if float_format_bits(format) > 32 {
                    "arith.truncf"
                } else {
                    body.push_str(&format!(
                        "    {result}_f32_zero = arith.constant 0.0 : f32\n    {result}_f32 = arith.addf {value}, {result}_f32_zero : f32\n"
                    ));
                    ""
                }
            }
            LoweredTensorElement::Integer { signed: true, .. } => "arith.sitofp",
            LoweredTensorElement::Integer { signed: false, .. } => "arith.uitofp",
            LoweredTensorElement::Boolean => {
                return Err(invalid("boolean cannot be converted through FP8"))
            }
        };
        if !operation.is_empty() {
            body.push_str(&format!(
                "    {result}_f32 = {operation} {value} : {source_type} to f32\n"
            ));
        }
        format!("{}_f32", result.trim_start_matches('%'))
    };

    if let Some(symbol) = fp8_encode_symbol(target) {
        body.push_str(&format!(
            "    {result}_target_bits = func.call @{symbol}(%{f32_value}) : (f32) -> i8\n    {result} = arith.bitcast {result}_target_bits : i8 to {target_type}\n"
        ));
        return Ok(body);
    }
    let conversion = match target {
        LoweredTensorElement::Float { format } => {
            if float_format_bits(format) > 32 {
                "arith.extf"
            } else if float_format_bits(format) < 32 {
                "arith.truncf"
            } else {
                body.push_str(&format!(
                    "    {result}_zero = arith.constant 0.0 : f32\n    {result} = arith.addf %{f32_value}, {result}_zero : f32\n"
                ));
                return Ok(body);
            }
        }
        LoweredTensorElement::Integer { signed: true, .. } => "arith.fptosi",
        LoweredTensorElement::Integer { signed: false, .. } => "arith.fptoui",
        LoweredTensorElement::Boolean => return Err(invalid("FP8 cannot be converted to boolean")),
    };
    body.push_str(&format!(
        "    {result} = {conversion} %{f32_value} : f32 to {target_type}\n"
    ));
    Ok(body)
}

const fn is_fp8(element: LoweredTensorElement) -> bool {
    matches!(
        element,
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E4M3Fn | LoweredFloatFormat::Float8E5M2
        }
    )
}

const fn fp8_decode_symbol(element: LoweredTensorElement) -> Option<&'static str> {
    match element {
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E4M3Fn,
        } => Some("__sev_f8e4m3fn_to_f32"),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E5M2,
        } => Some("__sev_f8e5m2_to_f32"),
        _ => None,
    }
}

const fn fp8_encode_symbol(element: LoweredTensorElement) -> Option<&'static str> {
    match element {
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E4M3Fn,
        } => Some("__sev_f32_to_f8e4m3fn"),
        LoweredTensorElement::Float {
            format: LoweredFloatFormat::Float8E5M2,
        } => Some("__sev_f32_to_f8e5m2"),
        _ => None,
    }
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
    use severian_compile::{CompileOperation, EffectSet};
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
            effects: EffectSet::default(),
            placement: None,
        }
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
}
