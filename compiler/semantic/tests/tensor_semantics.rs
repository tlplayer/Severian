use severian_hir::{
    Expression, Instruction, TensorDimension, TensorElementType, TensorType, ValueType,
};
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::analyze;

fn analyze_source(source: &str) -> Result<severian_hir::Program, severian_semantic::SemanticError> {
    analyze(&parse(&lex(source).unwrap()).unwrap())
}

#[test]
fn incompatible_broadcast_is_a_semantic_error() {
    let source = concat!(
        "unsafe:\n",
        "    native(\"__sev_tensor_add\") def tensor_add[T: Numeric](left: Tensor[T], right: Tensor[T]) -> Tensor[T]\n",
        "\n",
        "def combine(left: Tensor[f32, 2, 3], right: Tensor[f32, 4, 3]) -> Tensor[f32]:\n",
        "    return tensor_add(left, right)\n",
    );
    let error = analyze_source(source).unwrap_err();
    assert!(error.message.contains("E002402"));
    assert!(error.message.contains("tensor shapes cannot be broadcast"));
}

#[test]
fn matmul_result_shape_is_resolved_before_mir() {
    let source = concat!(
        "unsafe:\n",
        "    native(\"__sev_tensor_matmul\") def tensor_matmul[T: Numeric](left: Tensor[T], right: Tensor[T]) -> Tensor[T]\n",
        "\n",
        "def project(left: Tensor[f32, 2, 4, 16], right: Tensor[f32, 16, 8]) -> Tensor[f32, 2, 4, 8]:\n",
        "    return tensor_matmul(left, right)\n",
    );
    let program = analyze_source(source).unwrap();
    let function = program
        .functions
        .iter()
        .find(|function| function.name == "project")
        .unwrap();
    let Instruction::Return(Some(value)) = &function.instructions[0] else {
        panic!("expected a returned tensor operation");
    };
    let expected = TensorType::ranked(
        TensorElementType::F32,
        &[
            TensorDimension::Static(2),
            TensorDimension::Static(4),
            TensorDimension::Static(8),
        ],
    )
    .unwrap();
    assert_eq!(value.ty(), Some(ValueType::Tensor(expected)));
    assert!(
        matches!(value.kind(), Expression::Call { target, .. } if target.tensor_intrinsic().is_some())
    );
}

#[test]
fn tensor_operator_resolves_to_the_same_intrinsic_and_shape_rules() {
    let source = concat!(
        "import tensor\n",
        "@tensor(X)\n",
        "def project(left: Tensor[f32, 2, 4, 16], right: Tensor[f32, 16, 8]) -> Tensor[f32, 2, 4, 8]:\n",
        "    return left X right\n",
    );
    let program = analyze_source(source).unwrap();
    let function = program
        .functions
        .iter()
        .find(|function| function.name == "project")
        .unwrap();
    let Instruction::Return(Some(value)) = &function.instructions[0] else {
        panic!("expected a returned tensor operation");
    };
    assert!(matches!(
        value.kind(),
        Expression::Call { target, .. }
            if target.tensor_intrinsic() == Some(severian_hir::TensorIntrinsic::Matmul)
    ));
    assert_eq!(
        value.ty(),
        Some(ValueType::Tensor(
            TensorType::ranked(
                TensorElementType::F32,
                &[
                    TensorDimension::Static(2),
                    TensorDimension::Static(4),
                    TensorDimension::Static(8),
                ],
            )
            .unwrap()
        ))
    );
}
