use severian_hir::{
    Expression, Function, Instruction, Program, TensorDimension, TensorElementType, TensorType,
    ValueType,
};

#[test]
fn finds_the_main_function() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            name: "main".into(),
            native_symbol: None,
            decorators: vec![],
            contract: None,
            params: vec![],
            return_type: ValueType::Unit,
            instructions: vec![Instruction::Print(Expression::String("hello".into()))],
            tests: vec![],
        }],
    };

    assert_eq!(program.main().unwrap().name, "main");
}

#[test]
fn verifies_ranked_tensor_compatibility_and_broadcasting() {
    let matrix = TensorType::ranked(
        TensorElementType::F64,
        &[TensorDimension::Static(2), TensorDimension::Static(3)],
    )
    .unwrap();
    let row = TensorType::ranked(TensorElementType::F64, &[TensorDimension::Static(3)]).unwrap();
    assert_eq!(matrix.broadcast_with(row).unwrap(), matrix);

    let incompatible =
        TensorType::ranked(TensorElementType::F64, &[TensorDimension::Static(4)]).unwrap();
    assert!(matrix.broadcast_with(incompatible).is_err());
}
