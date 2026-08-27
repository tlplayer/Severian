#![forbid(unsafe_code)]

use severian_compile::{CompileContext, CompileError, CompileHandler, CompileRegion};
use severian_mlir::{
    type_spelling, LoweredFloatFormat, LoweredTensorDimension, LoweredTensorElement,
    LoweredTensorShape, LoweredType, MlirArtifact,
};
use severian_universal::{
    tensor, AttrValue, Attrs, FloatFormat, IntegerWidth, OpId, PrimitiveRepresentation,
    TensorDimension, TensorShape, TypeContext, TypeId,
};

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
    ) -> Result<MlirArtifact, CompileError> {
        let [operation] = region.compile_operations.as_slice() else {
            return Err(invalid(
                "the tensor compiler expects one reduced CompileOp per region",
            ));
        };
        let inputs = operation
            .operands
            .iter()
            .map(|ty| lower_type(*ty, context))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = operation
            .results
            .iter()
            .map(|ty| lower_type(*ty, context))
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
        let declarations = operation_declarations(operation.id, &inputs, &outputs)?;
        let body = lower_operation(
            operation.id,
            &inputs,
            &outputs,
            &input_spellings,
            &output_spellings,
            &operation.attributes,
        )?;
        Ok(MlirArtifact {
            module: format!(
                "module {{\n{declarations}  func.func @entry({parameters}){result_signature} {{\n{body}  }}\n}}"
            ),
            inputs,
            outputs,
        })
    }
}

fn operation_declarations(
    operation: OpId,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
) -> Result<String, CompileError> {
    if operation == tensor::FROM_ELEMENTS {
        return Ok("  func.func private @__sev_list_get_f64(!llvm.ptr, i64) -> f64\n".into());
    }
    if operation == tensor::VALUES {
        let [LoweredType::Tensor { element, .. }] = inputs else {
            return Err(invalid("values requires one structural tensor operand"));
        };
        let scalar = tensor_element_spelling(*element)?;
        return Ok(format!(
            "  func.func private @__sev_list_create() -> !llvm.ptr\n  func.func private @__sev_list_push_f64(!llvm.ptr, {scalar})\n"
        ));
    }
    if matches!(operation, tensor::SHAPE | tensor::STRIDES) {
        return Ok(
            "  func.func private @__sev_list_create() -> !llvm.ptr\n  func.func private @__sev_list_push_i64(!llvm.ptr, i64)\n"
                .into(),
        );
    }
    let _ = outputs;
    Ok(String::new())
}

fn lower_operation(
    operation: OpId,
    inputs: &[LoweredType],
    outputs: &[LoweredType],
    input_spellings: &[String],
    output_spellings: &[String],
    attributes: &Attrs,
) -> Result<String, CompileError> {
    if operation == tensor::FROM_ELEMENTS {
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
            return Err(invalid("ranked list construction currently consumes list[float]"));
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

    if operation == tensor::VALUES {
        let [input] = inputs else {
            return Err(invalid("values requires one tensor operand"));
        };
        let LoweredType::Tensor { element, shape } = input else {
            return Err(invalid("values operand must be a structural tensor"));
        };
        let scalar = tensor_element_spelling(*element)?;
        if scalar != "f64" {
            return Err(invalid("values currently returns list[float] and requires f64"));
        }
        let shape = effective_shape(shape, attributes);
        let dimensions = static_dimensions(&shape)
            .map_err(|error| invalid(format!("values: {error}")))?;
        let coordinates = tensor_coordinates(&dimensions);
        let input_type = input_spellings
            .first()
            .ok_or_else(|| invalid("values input type is missing"))?;
        let ranked_type = type_spelling(&LoweredType::Tensor {
            element: *element,
            shape,
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
        let mut body = format!(
            "{cast}    %result = func.call @__sev_list_create() : () -> !llvm.ptr\n"
        );
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

    if matches!(operation, tensor::SHAPE | tensor::STRIDES) {
        let [LoweredType::Tensor { shape, .. }] = inputs else {
            return Err(invalid("shape and strides require one tensor operand"));
        };
        let shape = effective_shape(shape, attributes);
        let dimensions = static_dimensions(&shape)
            .map_err(|error| invalid(format!("shape metadata: {error}")))?;
        let values = if operation == tensor::SHAPE {
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
        let mut body = "    %result = func.call @__sev_list_create() : () -> !llvm.ptr\n".to_owned();
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
    let binary = if operation == tensor::ADD {
        Some(if is_float(result_element) {
            "arith.addf"
        } else {
            "arith.addi"
        })
    } else if operation == tensor::SUBTRACT {
        Some(if is_float(result_element) {
            "arith.subf"
        } else {
            "arith.subi"
        })
    } else if operation == tensor::MULTIPLY {
        Some(if is_float(result_element) {
            "arith.mulf"
        } else {
            "arith.muli"
        })
    } else if operation == tensor::DIVIDE {
        Some(match result_element {
            LoweredTensorElement::Float { .. } => "arith.divf",
            LoweredTensorElement::Integer { signed: true, .. } => "arith.divsi",
            LoweredTensorElement::Integer { signed: false, .. }
            | LoweredTensorElement::Boolean => "arith.divui",
        })
    } else {
        None
    };
    if let Some(instruction) = binary {
        if inputs.len() != 2 {
            return Err(invalid(
                "binary tensor operations require two operands",
            ));
        }
        if input_spellings != [output.clone(), output.clone()] {
            let [
                LoweredType::Tensor {
                    element: left_element,
                    shape: LoweredTensorShape::Ranked(left_shape),
                },
                LoweredType::Tensor {
                    element: right_element,
                    shape: LoweredTensorShape::Ranked(right_shape),
                },
            ] = inputs
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
                return Err(invalid(
                    "binary tensor operands must have one element type",
                ));
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

    let unary = if operation == tensor::EXP {
        Some("math.exp")
    } else if operation == tensor::LOG {
        Some("math.log")
    } else if operation == tensor::TANH {
        Some("math.tanh")
    } else if operation == tensor::RSQRT {
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

    if operation == tensor::MATERIALIZE {
        if input_spellings.len() != 1 || input_spellings.first() != Some(output) {
            return Err(invalid("materialize preserves the complete tensor type"));
        }
        return Ok(format!("    return %arg0 : {output}\n"));
    }

    if operation == tensor::TRANSPOSE {
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
        let loop_dimensions = (0..rank)
            .map(|axis| format!("d{axis}"))
            .collect::<Vec<_>>();
        let input_map = loop_dimensions.iter().rev().cloned().collect::<Vec<_>>();
        let iterator_types = vec!["\"parallel\""; rank].join(", ");
        return Ok(format!(
            "    %empty = tensor.empty() : {output}\n    %result = linalg.generic {{indexing_maps = [affine_map<({loops}) -> ({input_map})>, affine_map<({loops}) -> ({loops})>], iterator_types = [{iterator_types}]}} ins(%arg0 : {input}) outs(%empty : {output}) {{\n    ^bb0(%element: {scalar}, %unused: {scalar}):\n      linalg.yield %element : {scalar}\n    }} -> {output}\n    return %result : {output}\n",
            loops = loop_dimensions.join(", "),
            input_map = input_map.join(", "),
        ));
    }

    if operation == tensor::MATMUL {
        let [
            LoweredType::Tensor {
                element: left_element,
                shape: left_shape,
            },
            LoweredType::Tensor {
                element: right_element,
                shape: right_shape,
            },
        ] = inputs
        else {
            return Err(invalid("matmul requires two tensor operands"));
        };
        if left_element != right_element || left_element != &result_element {
            return Err(invalid("matmul operands and result must have one element type"));
        }
        if static_dimensions(left_shape)
            .and_then(|dimensions| {
                (dimensions.len() == 2)
                    .then_some(dimensions)
                    .ok_or_else(|| invalid("matmul left operand must have rank two"))
            })
            .is_err()
            || static_dimensions(right_shape)
                .and_then(|dimensions| {
                    (dimensions.len() == 2)
                        .then_some(dimensions)
                        .ok_or_else(|| invalid("matmul right operand must have rank two"))
                })
                .is_err()
        {
            return Err(invalid(
                "initial generic linalg.matmul lowering requires statically ranked operands",
            ));
        }
        let LoweredType::Tensor { element, shape } = result_type else {
            unreachable!();
        };
        let shape = effective_shape(shape, attributes);
        if static_dimensions(&shape)
            .map_err(|error| invalid(format!("matmul: {error}")))?
            .len()
            != 2
        {
            return Err(invalid("initial generic linalg.matmul lowering requires rank two"));
        }
        let scalar = tensor_element_spelling(*element)?;
        let zero = if is_float(*element) { "0.0" } else { "0" };
        let ranked_type = type_spelling(&LoweredType::Tensor {
            element: *element,
            shape,
        })
        .map_err(|error| invalid(error.to_string()))?;
        return Ok(format!(
            "    %empty = tensor.empty() : {ranked_type}\n    %zero = arith.constant {zero} : {scalar}\n    %initialized = linalg.fill ins(%zero : {scalar}) outs(%empty : {ranked_type}) -> {ranked_type}\n    %result = linalg.matmul ins(%arg0, %arg1 : {left_type}, {right_type}) outs(%initialized : {ranked_type}) -> {ranked_type}\n    return %result : {output}\n",
            left_type = input_spellings[0],
            right_type = input_spellings[1],
        ));
    }

    Err(invalid(format!(
        "generic MLIR tensor lowering is not implemented for operation {operation:?}"
    )))
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
            LoweredTensorDimension::Known(value) => usize::try_from(*value)
                .map_err(|_| invalid("tensor dimension does not fit usize")),
            LoweredTensorDimension::Dynamic => {
                Err(invalid("operation requires statically known tensor dimensions"))
            }
        })
        .collect()
}

fn static_element_count(shape: &LoweredTensorShape) -> Result<usize, CompileError> {
    static_dimensions(shape)?.into_iter().try_fold(1usize, |count, dimension| {
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
    use severian_target::TargetSpec;
    use severian_universal::{install_primitives, Attrs, TypeContextBuilder};

    const ELEMENTS: &[&str] = &[
        "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128",
        "f8e4m3fn", "f8e5m2", "f16", "bf16", "f32", "f64", "f80", "f128",
    ];
    const FLOAT_ELEMENTS: &[&str] = &[
        "f8e4m3fn", "f8e5m2", "f16", "bf16", "f32", "f64", "f80", "f128",
    ];
    const NUMERIC_OPERATIONS: &[(OpId, usize)] = &[
        (tensor::ADD, 2),
        (tensor::SUBTRACT, 2),
        (tensor::MULTIPLY, 2),
        (tensor::DIVIDE, 2),
        (tensor::MATMUL, 2),
    ];
    const FLOAT_OPERATIONS: &[(OpId, usize)] = &[
        (tensor::EXP, 1),
        (tensor::LOG, 1),
        (tensor::TANH, 1),
        (tensor::RSQRT, 1),
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

    fn region(tensor_type: TypeId, operation: OpId, operands: usize) -> CompileRegion {
        CompileRegion {
            id: CompiledRegionId::new(0),
            compiler: tensor::compiler_id(),
            operations: Vec::new(),
            compile_operations: vec![CompileOperation {
                id: operation,
                operands: vec![tensor_type; operands],
                results: vec![tensor_type],
                attributes: Attrs::new(),
            }],
            inputs: Vec::new(),
            outputs: Vec::new(),
            effects: EffectSet::default(),
        }
    }

    fn compile_and_verify(
        types: &TypeContext,
        tensor_type: TypeId,
        operation: OpId,
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
            for operation in [tensor::MATERIALIZE, tensor::TRANSPOSE] {
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
        let mut operation = region(result, tensor::ADD, 2);
        operation.compile_operations[0].operands = vec![matrix, row];
        let artifact = TensorCompiler
            .compile(
                &operation,
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap();
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
        let mut operation = region(result, tensor::MATMUL, 2);
        operation.compile_operations[0].operands = vec![left, right];
        let artifact = TensorCompiler
            .compile(
                &operation,
                &CompileContext {
                    types: &types,
                    target: &TargetSpec::host(),
                },
            )
            .unwrap();
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
}
