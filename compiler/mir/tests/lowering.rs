use severian_hir::{
    BindingRef, Expression, Function, FunctionId, Instruction, Parameter, Program, ScopedBehavior,
    TaskPlacement, ValueType,
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

    let mir = severian_mir::lower(&hir).unwrap();
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
fn scoped_behaviors_exit_in_reverse_order_on_early_and_normal_returns() {
    let time = ScopedBehavior {
        provider: "Time".into(),
        with_member: "Time::with".into(),
        without_member: "Time::without".into(),
    };
    let memory = ScopedBehavior {
        provider: "Memory".into(),
        with_member: "Memory::with".into(),
        without_member: "Memory::without".into(),
    };
    let hir = Program {
        metadata: Default::default(),
        globals: Vec::new(),
        classes: Vec::new(),
        functions: vec![Function {
            id: FunctionId::from_name("measured"),
            name: "measured".into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: Vec::new(),
            return_type: ValueType::Unit,
            instructions: vec![Instruction::With {
                placement: TaskPlacement::Default,
                resources: Vec::new(),
                scoped_behaviors: vec![time.clone(), memory.clone()],
                instructions: vec![Instruction::If {
                    condition: Expression::Boolean(true),
                    then_instructions: vec![Instruction::Return(None)],
                    else_instructions: vec![Instruction::Evaluate(Expression::Integer(1))],
                }],
            }],
            tests: Vec::new(),
        }],
    };

    let mir = severian_mir::lower(&hir).unwrap();
    let function = &mir.functions[0];
    let entered = function.blocks[0]
        .operations
        .iter()
        .filter_map(|operation| {
            let severian_mir::OperationKind::ScopeEnter(behavior) = &operation.kind else {
                return None;
            };
            Some(behavior.provider.as_str())
        })
        .collect::<Vec<_>>();
    assert_eq!(entered, ["Time", "Memory"]);
    let return_blocks = function
        .blocks
        .iter()
        .filter(|block| matches!(block.terminator, severian_mir::Terminator::Return(_)))
        .collect::<Vec<_>>();
    assert_eq!(return_blocks.len(), 2);
    for block in return_blocks {
        assert!(matches!(
            block.operations.as_slice(),
            [.., severian_mir::Operation {
                kind: severian_mir::OperationKind::ScopeExit(memory_exit),
                ..
            }, severian_mir::Operation {
                kind: severian_mir::OperationKind::ScopeExit(time_exit),
                ..
            }] if memory_exit == &memory && time_exit == &time
        ));
    }
}

#[test]
fn loop_control_exits_only_the_scoped_behaviors_it_leaves() {
    let metric = ScopedBehavior {
        provider: "Metric".into(),
        with_member: "Metric::with".into(),
        without_member: "Metric::without".into(),
    };
    let hir = Program {
        metadata: Default::default(),
        globals: Vec::new(),
        classes: Vec::new(),
        functions: vec![Function {
            id: FunctionId::from_name("looping"),
            name: "looping".into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: Vec::new(),
            return_type: ValueType::Unit,
            instructions: vec![Instruction::While {
                setup: None,
                capabilities: Vec::new(),
                condition: Expression::Boolean(true),
                instructions: vec![Instruction::With {
                    placement: TaskPlacement::Default,
                    resources: Vec::new(),
                    scoped_behaviors: vec![metric.clone()],
                    instructions: vec![Instruction::If {
                        condition: Expression::Boolean(true),
                        then_instructions: vec![Instruction::Break],
                        else_instructions: vec![Instruction::Continue],
                    }],
                }],
            }],
            tests: Vec::new(),
        }],
    };

    let mir = severian_mir::lower(&hir).unwrap();
    let loop_exit_blocks = mir.functions[0]
        .blocks
        .iter()
        .filter(|block| {
            matches!(
                block.terminator,
                severian_mir::Terminator::Break | severian_mir::Terminator::Continue
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(loop_exit_blocks.len(), 2);
    for block in loop_exit_blocks {
        assert!(matches!(
            block.operations.last(),
            Some(severian_mir::Operation {
                kind: severian_mir::OperationKind::ScopeExit(behavior),
                ..
            }) if behavior == &metric
        ));
    }
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
    let mut mir = severian_mir::lower(&hir).unwrap();
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

    let mir = severian_mir::lower(&hir).unwrap();
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
    )))
    .unwrap();
    assert!(matches!(
        resolved.functions[0].tensor_operations.as_slice(),
        [severian_mir::TensorOp::Elementwise(operation)]
            if operation.kind == severian_mir::ElementwiseKind::Relu
    ));

    let unresolved = severian_mir::lower(&program(severian_hir::CallTarget::source(
        "__sev_tensor_relu",
    )))
    .unwrap();
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

#[test]
fn recognized_tensor_intrinsic_cannot_fall_back_when_metadata_is_missing() {
    let tensor = severian_hir::TensorType::ranked(
        severian_hir::TensorElementType::F32,
        &[severian_hir::TensorDimension::Static(4)],
    )
    .unwrap();
    let parameter = BindingRef::synthetic("value");
    let typed = |expression| Expression::Typed {
        id: severian_hir::HirId::synthetic(7),
        ty: ValueType::Tensor(tensor),
        any_origin: None,
        expression: Box::new(expression),
    };
    let hir = Program {
        functions: vec![Function {
            id: FunctionId::from_name("missing_axis"),
            name: "missing_axis".into(),
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
                target: severian_hir::CallTarget::native(
                    "softmax_axis",
                    "__sev_tensor_f32_softmax_axis",
                ),
                args: vec![typed(Expression::Variable(parameter))],
            })))],
            tests: Vec::new(),
        }],
        ..Program::default()
    };

    let error = severian_mir::lower(&hir).unwrap_err();
    assert_eq!(error.intrinsic, severian_hir::TensorIntrinsic::SoftmaxAxis);
    assert_eq!(error.function.as_deref(), Some("missing_axis"));
    assert!(error.expression.is_some());
    assert!(error.message.contains("axis argument"));
}

#[test]
fn recognized_tensor_intrinsic_cannot_have_a_non_tensor_result() {
    let tensor = severian_hir::TensorType::ranked(
        severian_hir::TensorElementType::F32,
        &[severian_hir::TensorDimension::Static(4)],
    )
    .unwrap();
    let parameter = BindingRef::synthetic("value");
    let tensor_value = Expression::Typed {
        id: severian_hir::HirId::synthetic(10),
        ty: ValueType::Tensor(tensor),
        any_origin: None,
        expression: Box::new(Expression::Variable(parameter.clone())),
    };
    let hir = Program {
        functions: vec![Function {
            id: FunctionId::from_name("invalid_relu"),
            name: "invalid_relu".into(),
            native_symbol: None,
            decorators: Vec::new(),
            contract: None,
            params: vec![Parameter {
                name: parameter,
                ty: ValueType::Tensor(tensor),
                default: None,
                receiver: None,
            }],
            return_type: ValueType::Int,
            instructions: vec![Instruction::Return(Some(Expression::Typed {
                id: severian_hir::HirId::synthetic(11),
                ty: ValueType::Int,
                any_origin: None,
                expression: Box::new(Expression::Call {
                    target: severian_hir::CallTarget::native("relu", "__sev_tensor_relu"),
                    args: vec![tensor_value],
                }),
            }))],
            tests: Vec::new(),
        }],
        ..Program::default()
    };

    let error = severian_mir::lower(&hir).unwrap_err();
    assert_eq!(error.intrinsic, severian_hir::TensorIntrinsic::Relu);
    assert_eq!(error.function.as_deref(), Some("invalid_relu"));
    assert!(error.expression.is_some());
    assert!(error.message.contains("non-tensor result type"));
}

#[test]
fn verifier_rejects_a_missing_operation_for_a_recognized_intrinsic() {
    let tensor = severian_hir::TensorType::ranked(
        severian_hir::TensorElementType::F32,
        &[severian_hir::TensorDimension::Static(4)],
    )
    .unwrap();
    let parameter = BindingRef::synthetic("value");
    let typed = |expression| Expression::Typed {
        id: severian_hir::HirId::synthetic(9),
        ty: ValueType::Tensor(tensor),
        any_origin: None,
        expression: Box::new(expression),
    };
    let hir = Program {
        functions: vec![Function {
            id: FunctionId::from_name("relu"),
            name: "relu".into(),
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
                target: severian_hir::CallTarget::native("relu", "__sev_tensor_relu"),
                args: vec![typed(Expression::Variable(parameter))],
            })))],
            tests: Vec::new(),
        }],
        ..Program::default()
    };
    let mut mir = severian_mir::lower(&hir).unwrap();
    mir.functions[0].tensor_operations.clear();

    let errors = severian_mir::verify(&mir).unwrap_err();
    assert!(errors
        .iter()
        .any(|error| error.invariant == "complete-tensor-lowering"));
}
