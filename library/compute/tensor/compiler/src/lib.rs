#![forbid(unsafe_code)]

use severian_compile::{CompileContext, CompileError, CompileHandler, CompileRegion};
use severian_mlir::{LoweredType, MlirArtifact};
use severian_universal::{tensor, AttrValue, OpId};

#[derive(Debug, Clone, Copy, Default)]
pub struct TensorCompiler;

impl CompileHandler for TensorCompiler {
    fn compile(
        &self,
        region: &CompileRegion,
        context: &CompileContext<'_>,
    ) -> Result<MlirArtifact, CompileError> {
        let [operation] = region.compile_operations.as_slice() else {
            return Err(CompileError::InvalidArtifact(
                "the tensor compiler expects one reduced CompileOp per region".into(),
            ));
        };
        let symbol = runtime_symbol(operation.id).ok_or_else(|| {
            CompileError::InvalidArtifact(format!(
                "the tensor compiler does not implement operation {:?}",
                operation.id
            ))
        })?;
        let parameters = (0..operation.operands.len())
            .map(|index| format!("%arg{index}: !llvm.ptr"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = (0..operation.operands.len())
            .map(|index| format!("%arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let argument_types = vec!["!llvm.ptr"; operation.operands.len()].join(", ");
        let (runtime_parameters, call_arguments, call_types, setup) =
            if operation.id == tensor::CONVERT {
                let Some(AttrValue::Type(target)) =
                    operation.attributes.get(&tensor::TARGET_ELEMENT_TYPE)
                else {
                    return Err(CompileError::InvalidArtifact(
                        "tensor conversion is missing target element metadata".into(),
                    ));
                };
                let tag = tensor::element_storage_tag(context.types, *target).ok_or_else(|| {
                    CompileError::InvalidArtifact(format!(
                        "type {target:?} is not a supported tensor element"
                    ))
                })?;
                (
                    "!llvm.ptr, i32".to_owned(),
                    format!("{arguments}, %dtype"),
                    "!llvm.ptr, i32".to_owned(),
                    format!("    %dtype = arith.constant {tag} : i32\n"),
                )
            } else {
                (
                    argument_types.clone(),
                    arguments,
                    argument_types.clone(),
                    String::new(),
                )
            };
        let element = operation
            .attributes
            .get(&tensor::TARGET_ELEMENT_TYPE)
            .or_else(|| operation.attributes.get(&tensor::ELEMENT_TYPE))
            .and_then(|value| match value {
                AttrValue::Type(element) => context
                    .types
                    .definition(*element)
                    .map(|definition| definition.name.as_str()),
                _ => None,
            })
            .unwrap_or("dynamic");
        let module = format!(
            "module {{\n  func.func private @{symbol}({runtime_parameters}) -> !llvm.ptr\n  func.func @entry({parameters}) -> !llvm.ptr attributes {{severian.tensor.element_type = \"{element}\"}} {{\n{setup}    %result = func.call @{symbol}({call_arguments}) : ({call_types}) -> !llvm.ptr\n    return %result : !llvm.ptr\n  }}\n}}"
        );
        Ok(MlirArtifact {
            module,
            inputs: vec![LoweredType::Bytes; operation.operands.len()],
            outputs: vec![LoweredType::Bytes; operation.results.len()],
        })
    }
}

fn runtime_symbol(operation: OpId) -> Option<&'static str> {
    [
        (tensor::FROM_ELEMENTS, "__sev_tensor_from_elements"),
        (tensor::CONVERT, "__sev_tensor_convert"),
        (tensor::ADD, "__sev_tensor_add"),
        (tensor::SUBTRACT, "__sev_tensor_subtract"),
        (tensor::MULTIPLY, "__sev_tensor_multiply"),
        (tensor::DIVIDE, "__sev_tensor_divide"),
        (tensor::REDUCE_SUM, "__sev_tensor_sum"),
        (tensor::MATMUL, "__sev_tensor_matmul"),
        (tensor::TRANSPOSE, "__sev_tensor_transpose"),
        (tensor::SLICE, "__sev_tensor_slice"),
        (tensor::MATERIALIZE, "__sev_tensor_materialize"),
        (tensor::SHAPE, "__sev_tensor_shape"),
        (tensor::STRIDES, "__sev_tensor_strides"),
        (tensor::VALUES, "__sev_tensor_values"),
    ]
    .into_iter()
    .find_map(|(known, symbol)| (operation == known).then_some(symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use severian_universal::{install_primitives, TypeContextBuilder};

    #[test]
    fn every_numeric_storage_width_has_one_stable_runtime_tag() {
        let mut builder = TypeContextBuilder::new();
        install_primitives(&mut builder).unwrap();
        let types = builder.build();
        for (expected, name) in [
            "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f8e4m3fn",
            "f8e5m2", "f16", "bf16", "f32", "f64", "f128",
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                tensor::element_storage_tag(&types, types.resolve_name(name).unwrap()),
                Some(expected as u8)
            );
        }
    }
}
