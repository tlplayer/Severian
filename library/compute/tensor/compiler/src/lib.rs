#![forbid(unsafe_code)]

use severian_compile::{CompileContext, CompileError, CompileHandler, CompileRegion};
use severian_mlir::{LoweredType, MlirArtifact};
use severian_target::ExecutionBackend;
use severian_universal::{tensor, AttrValue, ExecutionPlacement, OpId};

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
        let placement = operation
            .attributes
            .get(&severian_universal::EXECUTION_PLACEMENT_ATTRIBUTE)
            .and_then(|value| match value {
                AttrValue::String(value) => ExecutionPlacement::parse(value),
                _ => None,
            })
            .unwrap_or(ExecutionPlacement::Host);
        let backend = context
            .target
            .select_execution_backend(placement)
            .map_err(|error| CompileError::InvalidArtifact(error.to_string()))?;
        if !matches!(
            backend,
            ExecutionBackend::Native | ExecutionBackend::MlirVector
        ) {
            return Err(CompileError::InvalidArtifact(format!(
                "tensor operation {:?} selected `{}` for device `{}`, but that optional lowering plugin is not installed",
                operation.id,
                backend.as_str(),
                context
                    .target
                    .rocm_device()
                    .map(|device| device.name.as_str())
                    .unwrap_or("unspecified")
            )));
        }
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
        let compile_targets = operation
            .attributes
            .get(&severian_universal::COMPILE_TARGETS_ATTRIBUTE)
            .and_then(|value| match value {
                AttrValue::String(value) => Some(value.as_str()),
                _ => None,
            })
            .unwrap_or("mlir");
        let module = format!(
            "module {{\n  func.func private @{symbol}({runtime_parameters}) -> !llvm.ptr\n  func.func @entry({parameters}) -> !llvm.ptr attributes {{severian.compile.targets = \"{compile_targets}\", severian.execution.backend = \"{}\", severian.execution.placement = \"{}\", severian.tensor.element_type = \"{element}\"}} {{\n{setup}    %result = func.call @{symbol}({call_arguments}) : ({call_types}) -> !llvm.ptr\n    return %result : !llvm.ptr\n  }}\n}}",
            backend.as_str(),
            placement.as_str(),
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
    use severian_artifact::CompiledRegionId;
    use severian_compile::{CompileOperation, EffectSet};
    use severian_target::TargetSpec;
    use severian_universal::{install_primitives, Attrs, TypeContextBuilder};

    fn types() -> severian_universal::TypeContext {
        let mut builder = TypeContextBuilder::new();
        install_primitives(&mut builder).unwrap();
        builder.build()
    }

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

    fn region(placement: ExecutionPlacement) -> CompileRegion {
        let mut attributes = Attrs::new();
        attributes.insert(
            severian_universal::EXECUTION_PLACEMENT_ATTRIBUTE,
            AttrValue::String(placement.as_str().into()),
        );
        attributes.insert(
            severian_universal::COMPILE_TARGETS_ATTRIBUTE,
            AttrValue::String("mlir,stablehlo,xla".into()),
        );
        CompileRegion {
            id: CompiledRegionId::new(0),
            compiler: tensor::compiler_id(),
            operations: Vec::new(),
            compile_operations: vec![CompileOperation {
                id: tensor::ADD,
                operands: vec![severian_universal::TypeId(1); 2],
                results: vec![severian_universal::TypeId(1)],
                attributes,
            }],
            inputs: Vec::new(),
            outputs: Vec::new(),
            effects: EffectSet::default(),
        }
    }

    #[test]
    fn simd_is_an_explicit_mlir_vector_route() {
        let types = types();
        let target = TargetSpec::new("x86_64-unknown-linux");
        let artifact = TensorCompiler
            .compile(
                &region(ExecutionPlacement::Simd),
                &CompileContext {
                    types: &types,
                    target: &target,
                },
            )
            .unwrap();
        assert!(artifact
            .module
            .contains("severian.execution.backend = \"mlir-vector\""));
        assert!(artifact
            .module
            .contains("severian.compile.targets = \"mlir,stablehlo,xla\""));
    }

    #[test]
    fn gpu_never_silently_uses_the_native_tensor_runtime() {
        let types = types();
        let target = TargetSpec::new("x86_64-unknown-linux");
        let error = TensorCompiler
            .compile(
                &region(ExecutionPlacement::Gpu),
                &CompileContext {
                    types: &types,
                    target: &target,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("no supported AMD GPU"));
    }
}
