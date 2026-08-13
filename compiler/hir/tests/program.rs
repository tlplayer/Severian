use severian_hir::{
    BindingRef, Expression, Function, FunctionId, Instruction, Parameter, Program, TensorDimension,
    TensorElementType, TensorType, ValueType,
};

#[test]
fn tensor_dtype_promotion_is_language_defined() {
    use TensorElementType as T;
    assert_eq!(T::promote(T::F8E4M3FN, T::F8E5M2), Some(T::F16));
    assert_eq!(T::promote(T::F16, T::BF16), Some(T::F32));
    assert_eq!(T::promote(T::F16, T::F32), Some(T::F32));
    assert_eq!(T::promote(T::I8, T::U8), Some(T::I16));
    assert_eq!(T::promote(T::C64, T::F64), Some(T::C128));
}

#[test]
fn finds_the_main_function() {
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            id: FunctionId::from_name("main"),
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
fn namespaces_binding_definitions_and_uses_together() {
    let parameter = BindingRef::source("value", 4, 9);
    let local = BindingRef::source("copy", 20, 24);
    let mut program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            id: FunctionId::from_name("copy"),
            name: "copy".into(),
            native_symbol: None,
            decorators: vec![],
            contract: None,
            params: vec![Parameter {
                name: parameter.clone(),
                ty: ValueType::Int,
                default: None,
                receiver: None,
            }],
            return_type: ValueType::Int,
            instructions: vec![
                Instruction::Let {
                    name: local.clone(),
                    value: Expression::Variable(parameter.clone()),
                },
                Instruction::Return(Some(Expression::Variable(local.clone()))),
            ],
            tests: vec![],
        }],
    };

    program.namespace_bindings("dependency");
    let function = &program.functions[0];
    let Instruction::Let { name, value } = &function.instructions[0] else {
        unreachable!()
    };
    let Expression::Variable(parameter_use) = value else {
        unreachable!()
    };
    let Instruction::Return(Some(Expression::Variable(local_use))) = &function.instructions[1]
    else {
        unreachable!()
    };

    assert_eq!(
        function.params[0].name.id,
        parameter.id.in_namespace("dependency")
    );
    assert_eq!(parameter_use.id, function.params[0].name.id);
    assert_eq!(name.id, local.id.in_namespace("dependency"));
    assert_eq!(local_use.id, name.id);
}

#[test]
fn verifies_ranked_tensor_compatibility_and_broadcasting() {
    let rank_two = TensorType::ranked(
        TensorElementType::F64,
        &[TensorDimension::Static(2), TensorDimension::Static(3)],
    )
    .unwrap();
    let row = TensorType::ranked(TensorElementType::F64, &[TensorDimension::Static(3)]).unwrap();
    assert_eq!(rank_two.broadcast_with(row).unwrap(), rank_two);

    let incompatible =
        TensorType::ranked(TensorElementType::F64, &[TensorDimension::Static(4)]).unwrap();
    assert!(rank_two.broadcast_with(incompatible).is_err());
}
