use severian_hir::{
    BindingRef, Expression, Function, FunctionId, Instruction, Parameter, Program, ValueType,
};

#[test]
fn creates_explicit_branch_blocks_with_typed_value_references() {
    let condition = Expression::Typed {
        id: severian_hir::HirId::from_source_range(0, 4),
        ty: ValueType::Bool,
        any_origin: None,
        expression: Box::new(Expression::Boolean(true)),
    };
    let mut hir = Program {
        metadata: Default::default(),
        globals: Vec::new(),
        classes: Vec::new(),
        functions: vec![Function {
            id: FunctionId::from_name("choose"),
            name: "choose".into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: Vec::new(),
            return_type: ValueType::Unit,
            instructions: vec![Instruction::If {
                condition,
                then_instructions: Vec::new(),
                else_instructions: Vec::new(),
            }],
            tests: Vec::new(),
        }],
    };
    hir.attach_source_file("/workspace/branch.sev", "true");

    let mir = severian_mir::lower(&hir);
    severian_mir::verify(&mir).expect("lowered MIR should satisfy its CFG invariants");
    assert_eq!(mir.functions[0].blocks.len(), 4);
    assert!(matches!(
        mir.functions[0].blocks[0].terminator,
        severian_mir::Terminator::Branch {
            condition: severian_mir::ValueRef {
                ty: Some(ValueType::Bool),
                ..
            },
            ..
        }
    ));
    let severian_mir::Terminator::Branch { condition, .. } = mir.functions[0].blocks[0].terminator
    else {
        unreachable!()
    };
    let span = mir
        .source_span(condition)
        .expect("MIR should retain HIR source metadata");
    assert_eq!(span.range.start, 0);
    assert_eq!(span.range.end, 4);
    assert_eq!(mir.metadata().sources.files().len(), 1);
}

#[test]
fn verifier_rejects_an_out_of_range_successor() {
    let hir = Program {
        metadata: Default::default(),
        globals: Vec::new(),
        classes: Vec::new(),
        functions: vec![Function {
            id: FunctionId::from_name("broken"),
            name: "broken".into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: Vec::new(),
            return_type: ValueType::Unit,
            instructions: Vec::new(),
            tests: Vec::new(),
        }],
    };
    let mut mir = severian_mir::lower(&hir);
    mir.functions[0].blocks[0].terminator =
        severian_mir::Terminator::Goto(severian_mir::BlockId(9));

    let errors = severian_mir::verify(&mir).expect_err("invalid successor must be rejected");
    assert_eq!(errors[0].invariant, "valid-successor");
    assert!(errors[0].to_string().contains("targets block 9"));
}

#[test]
fn resolves_operations_to_dense_local_ids_instead_of_names() {
    let parameter = BindingRef::source("value", 4, 9);
    let local = BindingRef::source("value", 20, 25);
    let hir = Program {
        metadata: Default::default(),
        globals: Vec::new(),
        classes: Vec::new(),
        functions: vec![Function {
            id: FunctionId::from_name("shadow"),
            name: "shadow".into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: vec![Parameter {
                name: parameter.clone(),
                ty: ValueType::Int,
                default: None,
                receiver: None,
            }],
            return_type: ValueType::Int,
            instructions: vec![Instruction::Let {
                name: local.clone(),
                value: Expression::Variable(parameter),
            }],
            tests: Vec::new(),
        }],
    };

    let mir = severian_mir::lower(&hir);
    let function = &mir.functions[0];
    assert_eq!(function.parameters, vec![severian_mir::LocalId(0)]);
    assert_eq!(function.locals[0].binding.name, "value");
    assert_eq!(function.locals[1].binding.name, "value");
    assert_ne!(function.locals[0].binding.id, function.locals[1].binding.id);
    assert!(matches!(
        function.blocks[0].operations[0],
        severian_mir::Operation {
            kind: severian_mir::OperationKind::Bind(severian_mir::LocalId(1)),
            ref operands,
        } if operands[0].local == Some(severian_mir::LocalId(0))
    ));
}

#[test]
fn preserves_resolved_tensor_operations_instead_of_reparsing_names() {
    let tensor = severian_hir::TensorType::ranked(
        severian_hir::TensorElementType::F32,
        &[severian_hir::TensorDimension::Static(8)],
    )
    .unwrap();
    let parameter = BindingRef::synthetic("value");
    let typed = |expression| Expression::Typed {
        id: severian_hir::HirId::synthetic(1),
        ty: ValueType::Tensor(tensor),
        any_origin: None,
        expression: Box::new(expression),
    };
    let program = |target| Program {
        functions: vec![Function {
            id: FunctionId::from_name("kernel"),
            name: "kernel".into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: vec![Parameter {
                name: parameter.clone(),
                ty: ValueType::Tensor(tensor),
                default: None,
                receiver: None,
            }],
            return_type: ValueType::Tensor(tensor),
            instructions: vec![Instruction::Return(Some(typed(Expression::Call {
                target,
                args: vec![typed(Expression::Variable(parameter.clone()))],
            })))],
            tests: Vec::new(),
        }],
        ..Program::default()
    };

    let mut resolved = severian_mir::lower(&program(severian_hir::CallTarget::native(
        "renamed_activation",
        "__sev_tensor_relu",
    )));
    assert!(matches!(
        resolved.functions[0].tensor_operations.as_slice(),
        [severian_mir::TensorOp::Elementwise(operation)]
            if operation.kind == severian_mir::ElementwiseKind::Relu
    ));

    let unresolved = severian_mir::lower(&program(severian_hir::CallTarget::source(
        "__sev_tensor_relu",
    )));
    assert!(unresolved.functions[0].tensor_operations.is_empty());

    let severian_mir::Terminator::Return(Some(value)) =
        &mut resolved.functions[0].blocks[0].terminator
    else {
        unreachable!()
    };
    value.tensor_op = Some(severian_mir::TensorOpId(9));
    let errors = severian_mir::verify(&resolved).unwrap_err();
    assert!(errors
        .iter()
        .any(|error| error.invariant == "valid-tensor-operation"));
}
