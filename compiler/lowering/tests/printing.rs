use severian_hir::{
    AssignmentOp, BinaryOp, CallTarget, Decorator, Expression, Function, FunctionId, HirId,
    Instruction, MatchPattern, Parameter, Program, SwitchArm, TaskPlacement, TensorElementType,
    TensorType, TypeDefinitionId, ValueType, VariantId,
};

#[test]
fn lowers_primitive_prints_to_native_calls() {
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
            instructions: vec![
                Instruction::Print(Expression::Integer(3)),
                Instruction::Print(Expression::Float(std::f64::consts::PI.to_bits())),
                Instruction::Print(Expression::Boolean(true)),
            ],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    let text = lowered.as_str();

    assert!(text.contains("llvm.call @printf"));
    assert!(text.contains("vararg(!llvm.func<i32 (!llvm.ptr, ...)>)"));
    assert!(text.contains("llvm.select"));
    assert!(text.contains("llvm.call @puts"));
}

#[test]
fn carries_reassigned_values_out_of_switch_arms() {
    let result = Expression::Variant {
        type_id: Some(TypeDefinitionId::from_name("Result")),
        variant_id: VariantId::from_name("ok"),
        name: "ok".into(),
        fields: vec![Expression::String("value".into())],
    };
    let assign = |value: &str| Instruction::Assign {
        target: Expression::Variable("state".into()),
        op: AssignmentOp::Assign,
        value: Expression::String(value.into()),
    };
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
            instructions: vec![
                Instruction::Let {
                    name: "state".into(),
                    value: Expression::String("initial".into()),
                },
                Instruction::Let {
                    name: "result".into(),
                    value: result,
                },
                Instruction::Switch {
                    value: Expression::Variable("result".into()),
                    arms: vec![
                        SwitchArm {
                            source: None,
                            pattern: MatchPattern::Constructor {
                                name: "ok".into(),
                                fields: vec![MatchPattern::Bind("value".into())],
                            },
                            guard: None,
                            instructions: vec![assign("updated")],
                            receivers: Default::default(),
                        },
                        SwitchArm {
                            source: None,
                            pattern: MatchPattern::Constructor {
                                name: "failure".into(),
                                fields: vec![MatchPattern::Bind("error".into())],
                            },
                            guard: None,
                            instructions: vec![assign("failed")],
                            receivers: Default::default(),
                        },
                    ],
                },
                Instruction::Print(Expression::Variable("state".into())),
            ],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    let text = lowered.as_str();
    let exit = text
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with("^bb") && line.contains(": !llvm.ptr"))
        .expect("switch exit must carry pointer variables as block arguments");
    let print = text
        .lines()
        .rev()
        .find(|line| line.contains("llvm.call @puts"))
        .expect("state must be printed after the switch");
    let state = print
        .split("@puts(")
        .nth(1)
        .and_then(|arguments| arguments.split(')').next())
        .expect("the print call must have a state operand");
    assert!(exit.contains(state));
}

#[test]
fn lowers_boolean_and_with_short_circuit_control_flow() {
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
            instructions: vec![Instruction::Print(Expression::Binary {
                left: Box::new(Expression::Boolean(true)),
                op: BinaryOp::And,
                right: Box::new(Expression::Boolean(true)),
            })],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    let text = lowered.as_str();
    assert!(text.contains("llvm.cond_br"));
    assert!(text.contains("llvm.br"));
    assert!(!text.contains("llvm.and"));
}

#[test]
fn lowers_conditional_expressions_to_lazy_control_flow() {
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
            instructions: vec![Instruction::Print(Expression::Conditional {
                condition: Box::new(Expression::Boolean(true)),
                then_expression: Box::new(Expression::Float(1.0f64.to_bits())),
                else_expression: Box::new(Expression::Float(0.0f64.to_bits())),
            })],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    let text = lowered.as_str();
    assert!(text.contains("llvm.cond_br"));
    assert!(text.contains("llvm.br"));
    assert!(!text.contains("llvm.select"));
}

#[test]
fn lowers_map_items_iteration_to_the_map_runtime() {
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
            instructions: vec![Instruction::Evaluate(Expression::MethodCall {
                object: Box::new(Expression::Map(vec![])),
                method: "items".into(),
                args: vec![],
            })],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    assert!(lowered.as_str().contains("call @__sev_map_items"));
}

#[test]
fn lowers_integer_range_for_to_control_flow() {
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
            instructions: vec![Instruction::For {
                setup: None,
                pattern: MatchPattern::Bind("value".into()),
                iterable: Expression::Call {
                    target: CallTarget::source("range"),
                    args: vec![Expression::Integer(0), Expression::Integer(3)],
                },
                instructions: vec![Instruction::Print(Expression::Variable("value".into()))],
            }],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    let text = lowered.as_str();
    assert!(text.contains("llvm.icmp \"slt\""));
    assert!(text.contains("llvm.cond_br"));
    assert!(text.contains("llvm.br"));
}

#[test]
fn lowers_unit_function_calls_without_an_invalid_result() {
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![],
        functions: vec![
            Function {
                id: FunctionId::from_name("consume"),
                name: "consume".into(),
                native_symbol: None,
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Unit,
                instructions: vec![],
                tests: vec![],
            },
            Function {
                id: FunctionId::from_name("main"),
                name: "main".into(),
                native_symbol: None,
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Unit,
                instructions: vec![Instruction::Evaluate(Expression::Call {
                    target: CallTarget::source("consume"),
                    args: vec![],
                })],
                tests: vec![],
            },
        ],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    assert!(lowered
        .as_str()
        .contains("llvm.call @__sev_fn_consume() : () -> ()"));
    assert!(!lowered.as_str().contains("= llvm.call @__sev_fn_consume"));
}

#[test]
fn attaches_local_distribution_to_the_task_spawn_not_the_function() {
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![],
        functions: vec![
            Function {
                id: FunctionId::from_name("work"),
                name: "work".into(),
                native_symbol: None,
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Int,
                instructions: vec![Instruction::Return(Some(Expression::Integer(1)))],
                tests: vec![],
            },
            Function {
                id: FunctionId::from_name("main"),
                name: "main".into(),
                native_symbol: None,
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Unit,
                instructions: vec![Instruction::Let {
                    name: "task".into(),
                    value: Expression::Task {
                        value: Box::new(Expression::Call {
                            target: CallTarget::source("work"),
                            args: vec![],
                        }),
                        placement: TaskPlacement::Local,
                    },
                }],
                tests: vec![],
            },
        ],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    let text = lowered.as_str();
    assert!(text.contains("llvm.call @__sev_task_spawn_work() {severian_distribution = \"local\"}"));
    assert!(!text.contains("llvm.func @main() -> i32 attributes"));
}

#[test]
fn preserves_parallel_placement_on_spawn_calls() {
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![],
        functions: vec![
            Function {
                id: FunctionId::from_name("work"),
                name: "work".into(),
                native_symbol: None,
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Int,
                instructions: vec![Instruction::Return(Some(Expression::Integer(1)))],
                tests: vec![],
            },
            Function {
                id: FunctionId::from_name("main"),
                name: "main".into(),
                native_symbol: None,
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Unit,
                instructions: vec![Instruction::Let {
                    name: "task".into(),
                    value: Expression::Task {
                        value: Box::new(Expression::Call {
                            target: CallTarget::source("work"),
                            args: vec![],
                        }),
                        placement: TaskPlacement::Gpu,
                    },
                }],
                tests: vec![],
            },
        ],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    let text = lowered.as_str();
    assert!(text.contains("severian_parallel = \"gpu\""));
    assert!(text.contains("severian_device_fallback = \"cpu\""));
}

#[test]
fn compares_dynamic_collection_values_by_value() {
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
            instructions: vec![Instruction::Print(Expression::Binary {
                left: Box::new(Expression::Index {
                    object: Box::new(Expression::List(vec![Expression::Float(1.0f64.to_bits())])),
                    index: Box::new(Expression::Integer(0)),
                }),
                op: BinaryOp::Greater,
                right: Box::new(Expression::Index {
                    object: Box::new(Expression::List(vec![Expression::Float(0.0f64.to_bits())])),
                    index: Box::new(Expression::Integer(0)),
                }),
            })],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program));
    assert!(lowered.as_str().contains("llvm.call @__sev_value_less"));
    assert!(!lowered.as_str().contains("llvm.icmp \"sgt\" %v"));
}

#[test]
fn reports_unranked_tensor_regions_instead_of_panicking() {
    let tensor = ValueType::Tensor(TensorType::dynamic(TensorElementType::F64));
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            id: FunctionId::from_name("contract"),
            name: "contract".into(),
            native_symbol: None,
            decorators: vec![Decorator {
                package: "tensor".into(),
                symbols: vec![],
            }],
            contract: None,
            params: vec![
                Parameter {
                    name: "left".into(),
                    ty: tensor,
                    default: None,
                    receiver: None,
                },
                Parameter {
                    name: "right".into(),
                    ty: tensor,
                    default: None,
                    receiver: None,
                },
            ],
            return_type: tensor,
            instructions: vec![Instruction::Return(Some(Expression::Typed {
                id: HirId::synthetic(1),
                ty: tensor,
                expression: Box::new(Expression::Call {
                    target: CallTarget::native("tensor.rankedMatmul", "__sev_tensor_matmul")
                        .with_signature([tensor, tensor], tensor),
                    args: vec![
                        Expression::Typed {
                            id: HirId::synthetic(2),
                            ty: tensor,
                            expression: Box::new(Expression::Variable("left".into())),
                        },
                        Expression::Typed {
                            id: HirId::synthetic(3),
                            ty: tensor,
                            expression: Box::new(Expression::Variable("right".into())),
                        },
                    ],
                }),
            }))],
            tests: vec![],
        }],
    };

    let error = severian_lowering::native_bridge_source(&program).unwrap_err();
    assert!(matches!(
        error,
        severian_lowering::stablehlo::StableHloLoweringError::InvalidRank {
            expected: 2,
            actual: None,
            ..
        }
    ));
}
