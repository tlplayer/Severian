use severian_driver::compile_dependency_path;
use severian_hir::TensorElementType;
use severian_mir::{ElementwiseKind, ReductionKind, ScalarValue, TensorOp};
use std::path::Path;

#[test]
fn generic_linear_classes_specialize_into_typed_mir() {
    std::thread::Builder::new()
        .name("generic-class-lowering".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let fixture = workspace.join("compiler/driver/tests/fixtures/generic_classes");
            let compilation =
                compile_dependency_path(&fixture.join("linear.sev"), &fixture.join("package.toml"))
                    .unwrap();

            assert!(compilation
                .optimized_hir
                .classes
                .iter()
                .all(|class| class.name != "Linear"));
            for (class_name, element) in [
                ("Linear__f32", TensorElementType::F32),
                ("Linear__bf16", TensorElementType::BF16),
            ] {
                let class = compilation
                    .optimized_hir
                    .classes
                    .iter()
                    .find(|class| class.name == class_name)
                    .unwrap_or_else(|| panic!("missing {class_name}"));
                assert!(matches!(
                    class.field_types.as_slice(),
                    [severian_hir::ValueType::Tensor(tensor)] if tensor.element == element
                ));

                let function_name = format!("{class_name}.forward");
                let forward = compilation
                    .mir
                    .functions
                    .iter()
                    .find(|function| function.name == function_name)
                    .unwrap_or_else(|| panic!("missing MIR function {function_name}"));
                assert!(matches!(
                    forward.tensor_operations.as_slice(),
                    [TensorOp::Transpose(transpose), TensorOp::Matmul(matmul)]
                        if transpose.result.element == element
                            && matmul.result.element == element
                ));
            }

            for (class_name, storage) in [
                ("layers_RMSNorm__f32", TensorElementType::F32),
                ("layers_RMSNorm__bf16", TensorElementType::BF16),
            ] {
                let forward = compilation
                    .mir
                    .functions
                    .iter()
                    .find(|function| function.name == format!("{class_name}.forward"))
                    .unwrap_or_else(|| panic!("missing MIR function {class_name}.forward"));
                assert!(!matches!(
                    forward.return_type,
                    severian_hir::ValueType::TensorAny
                ));
                assert!(forward.parameters.iter().all(|parameter| !matches!(
                    forward.locals[parameter.0 as usize].ty,
                    severian_hir::ValueType::TensorAny
                )));
                assert!(forward.tensor_operations.iter().all(|operation| {
                    !matches!(operation.result().element, TensorElementType::F64)
                }));
                assert!(forward.tensor_operations.iter().any(|operation| matches!(
                    operation,
                    TensorOp::Reduction(reduction)
                        if reduction.kind == ReductionKind::Mean
                            && reduction.last_axis
                            && reduction.accumulation == TensorElementType::F32
                            && reduction.result.element == TensorElementType::F32
                )));
                assert!(forward.tensor_operations.iter().any(|operation| matches!(
                    operation,
                    TensorOp::Scalar(scalar)
                        if matches!(scalar.value, ScalarValue::Operand(_))
                            && scalar.result.element == TensorElementType::F32
                )));
                assert!(forward.tensor_operations.iter().any(|operation| matches!(
                    operation,
                    TensorOp::Elementwise(elementwise)
                        if elementwise.kind == ElementwiseKind::Rsqrt
                            && elementwise.result.element == TensorElementType::F32
                )));
                let final_operation = forward.tensor_operations.last().unwrap();
                assert_eq!(final_operation.result().element, storage);
            }

            for (class_name, storage) in [
                ("layers_Linear__f32", TensorElementType::F32),
                ("layers_Linear__bf16", TensorElementType::BF16),
            ] {
                let forward = compilation
                    .mir
                    .functions
                    .iter()
                    .find(|function| function.name == format!("{class_name}.forward"))
                    .unwrap_or_else(|| panic!("missing MIR function {class_name}.forward"));
                assert!(forward.tensor_operations.iter().all(|operation| {
                    !matches!(operation.result().element, TensorElementType::F64)
                }));
                assert!(forward.tensor_operations.iter().any(|operation| matches!(
                    operation,
                    TensorOp::Matmul(matmul)
                        if matmul.accumulation == TensorElementType::F32
                            && matmul.result.element == TensorElementType::F32
                )));
                assert_eq!(
                    forward.tensor_operations.last().unwrap().result().element,
                    storage
                );
            }

            for (class_name, element, linear_name) in [
                (
                    "Qwen3MLP__f32",
                    TensorElementType::F32,
                    "layers_Linear__f32",
                ),
                (
                    "Qwen3MLP__bf16",
                    TensorElementType::BF16,
                    "layers_Linear__bf16",
                ),
            ] {
                let class = compilation
                    .optimized_hir
                    .classes
                    .iter()
                    .find(|class| class.name == class_name)
                    .unwrap_or_else(|| panic!("missing {class_name}"));
                assert_eq!(class.field_classes, vec![Some(linear_name.into()); 3]);
                let forward = compilation
                    .mir
                    .functions
                    .iter()
                    .find(|function| function.name == format!("{class_name}.forward"))
                    .unwrap_or_else(|| panic!("missing MIR function {class_name}.forward"));
                assert!(
                    matches!(
                        forward.tensor_operations.as_slice(),
                        [TensorOp::Elementwise(silu), TensorOp::Elementwise(multiply)]
                            if silu.kind == ElementwiseKind::Silu
                                && multiply.kind == ElementwiseKind::Multiply
                                && silu.result.element == element
                                && multiply.result.element == element
                    ),
                    "operations: {:?}",
                    forward
                        .tensor_operations
                        .iter()
                        .map(TensorOp::name)
                        .collect::<Vec<_>>()
                );
            }
            assert!(compilation
                .optimized_hir
                .classes
                .iter()
                .all(|class| !class.name.contains("Bf16") && class.name != "Qwen3MLP"));
        })
        .unwrap()
        .join()
        .unwrap();
}
