use severian_hir::{
    AssignmentOp, BinaryOp, BindingRef, CallTarget, Class, Decorator, Expression, Function,
    FunctionId, HirId, Instruction, MatchPattern, Parameter, Program, ReceiverType, SwitchArm,
    TaskPlacement, TensorElementType, TensorType, TypeDefinitionId, ValueType, VariantId,
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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
fn lowers_string_switch_arms_to_value_comparisons() {
    let arm = |pattern, value: &str| SwitchArm {
        source: None,
        pattern,
        guard: None,
        instructions: vec![Instruction::Print(Expression::String(value.into()))],
        receivers: Default::default(),
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
                    name: "extension".into(),
                    value: Expression::String(".json".into()),
                },
                Instruction::Switch {
                    value: Expression::Variable("extension".into()),
                    arms: vec![
                        arm(MatchPattern::String(".csv".into()), "csv"),
                        arm(MatchPattern::String(".json".into()), "json"),
                        arm(MatchPattern::Wildcard, "binary"),
                    ],
                },
            ],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
    assert_eq!(
        lowered
            .as_str()
            .lines()
            .filter(|line| line.contains("= llvm.call @__sev_string_equal"))
            .count(),
        2
    );
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
    let text = lowered.as_str();
    assert!(text.contains("llvm.icmp \"slt\""));
    assert!(text.contains("llvm.cond_br"));
    assert!(text.contains("llvm.br"));
}

#[test]
fn preserves_abstract_receiver_dispatch_through_for_loop() {
    let predict = Function {
        id: FunctionId::from_name("FixedModel.predict"),
        name: "predict".into(),
        native_symbol: None,
        decorators: vec![],
        contract: None,
        params: vec![Parameter {
            name: "tokens".into(),
            ty: ValueType::List,
            default: None,
            receiver: None,
        }],
        return_type: ValueType::Any,
        instructions: vec![Instruction::Return(Some(Expression::Variable(
            "tokens".into(),
        )))],
        tests: vec![],
    };
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![Class {
            id: TypeDefinitionId::from_name("FixedModel"),
            name: "FixedModel".into(),
            decorators: vec![],
            fields: vec![],
            field_types: vec![],
            field_classes: vec![],
            field_defaults: vec![],
            field_constraints: vec![],
            constructors: vec![],
            methods: vec![predict],
            method_return_classes: vec![None],
        }],
        functions: vec![Function {
            id: FunctionId::from_name("generate"),
            name: "generate".into(),
            native_symbol: None,
            decorators: vec![],
            contract: None,
            params: vec![Parameter {
                name: "model".into(),
                ty: ValueType::Interface(TypeDefinitionId::from_name("MaskedModel")),
                default: None,
                receiver: Some(ReceiverType {
                    name: "MaskedModel".into(),
                    concrete: false,
                    methods: vec!["predict".into()],
                }),
            }],
            return_type: ValueType::Unit,
            instructions: vec![Instruction::For {
                setup: None,
                pattern: MatchPattern::Bind("step".into()),
                iterable: Expression::Call {
                    target: CallTarget::source("range"),
                    args: vec![Expression::Integer(0), Expression::Integer(2)],
                },
                instructions: vec![Instruction::Evaluate(Expression::MethodCall {
                    object: Box::new(Expression::Variable("model".into())),
                    method: "predict".into(),
                    args: vec![Expression::List(vec![])],
                })],
            }],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
    let text = lowered.as_str();
    assert!(text.contains("llvm.call @__sev_dispatch_predict_list_any"));
    assert!(!text.contains("llvm.call @__sev_method_MaskedModel_predict"));
}

#[test]
fn lowers_an_abstract_method_without_a_local_implementation() {
    let predicate = BindingRef::synthetic("predicate");
    let input = BindingRef::synthetic("input");
    let target = BindingRef::synthetic("target");
    let typed_any = |id, value| Expression::Typed {
        id: HirId::synthetic(id),
        ty: ValueType::Any,
        any_origin: None,
        expression: Box::new(value),
    };
    let call = Expression::Typed {
        id: HirId::synthetic(3),
        ty: ValueType::Bool,
        any_origin: None,
        expression: Box::new(Expression::MethodCall {
            object: Box::new(Expression::Variable(predicate.clone())),
            method: "forward".into(),
            args: vec![
                typed_any(1, Expression::Variable(input.clone())),
                typed_any(2, Expression::Variable(target.clone())),
            ],
        }),
    };
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            id: FunctionId::from_name("select"),
            name: "select".into(),
            native_symbol: None,
            decorators: vec![],
            contract: None,
            params: vec![
                Parameter {
                    name: predicate,
                    ty: ValueType::Interface(TypeDefinitionId::from_name("Predicate")),
                    default: None,
                    receiver: Some(ReceiverType {
                        name: "Predicate".into(),
                        concrete: false,
                        methods: vec!["forward".into()],
                    }),
                },
                Parameter {
                    name: input,
                    ty: ValueType::Any,
                    default: None,
                    receiver: None,
                },
                Parameter {
                    name: target,
                    ty: ValueType::Any,
                    default: None,
                    receiver: None,
                },
            ],
            return_type: ValueType::Bool,
            instructions: vec![Instruction::Return(Some(call))],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
    let text = lowered.as_str();
    assert!(text.contains("llvm.func @__sev_dispatch_forward_any_any_bool"));
    assert!(text.contains("llvm.call @__sev_dispatch_forward_any_any_bool"));
}

#[test]
fn class_methods_take_precedence_over_builtin_method_names() {
    let type_id = TypeDefinitionId::from_name("Rows");
    let values_method = Function {
        id: FunctionId::from_name("Rows.values"),
        name: "values".into(),
        native_symbol: None,
        decorators: vec![],
        contract: None,
        params: vec![],
        return_type: ValueType::List,
        instructions: vec![Instruction::Return(Some(Expression::List(vec![
            Expression::Integer(7),
        ])))],
        tests: vec![],
    };
    let program = Program {
        metadata: Default::default(),
        globals: vec![],
        classes: vec![Class {
            id: type_id,
            name: "Rows".into(),
            decorators: vec![],
            fields: vec![],
            field_types: vec![],
            field_classes: vec![],
            field_defaults: vec![],
            field_constraints: vec![],
            constructors: vec![],
            methods: vec![values_method],
            method_return_classes: vec![None],
        }],
        functions: vec![Function {
            id: FunctionId::from_name("main"),
            name: "main".into(),
            native_symbol: None,
            decorators: vec![],
            contract: None,
            params: vec![],
            return_type: ValueType::Unit,
            instructions: vec![Instruction::Evaluate(Expression::MethodCall {
                object: Box::new(Expression::Construct {
                    type_id,
                    class: "Rows".into(),
                    args: vec![],
                }),
                method: "values".into(),
                args: vec![],
            })],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
    let text = lowered.as_str();
    assert!(text.contains("llvm.call @__sev_method_Rows_values"));
    assert!(!text.contains("llvm.call @__sev_map_values"));
}

#[test]
fn lowers_propagated_main_failures_to_a_nonzero_exit_status() {
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
            return_type: ValueType::Result,
            instructions: vec![Instruction::TryLet {
                name: "value".into(),
                value: Expression::Variant {
                    type_id: Some(TypeDefinitionId::from_name("Result")),
                    variant_id: VariantId::from_name("failure"),
                    name: "failure".into(),
                    fields: vec![Expression::String("unavailable".into())],
                },
                payload_type: ValueType::Any,
                receiver: None,
            }],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
    let main = lowered.as_str().split("llvm.func @main(").nth(1).unwrap();
    assert!(main.contains("llvm.mlir.constant(1 : i32) : i32"));
    assert!(main
        .lines()
        .filter(|line| line.trim_start().starts_with("llvm.return"))
        .all(|line| line.ends_with(": i32")));
}

#[test]
fn unboxes_statically_typed_result_payloads_at_the_variant_boundary() {
    let value = BindingRef::synthetic("value");
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
            return_type: ValueType::Result,
            instructions: vec![
                Instruction::TryLet {
                    name: value.clone(),
                    value: Expression::Variant {
                        type_id: Some(TypeDefinitionId::from_name("Result")),
                        variant_id: VariantId::from_name("ok"),
                        name: "ok".into(),
                        fields: vec![Expression::Integer(7)],
                    },
                    payload_type: ValueType::Int,
                    receiver: None,
                },
                Instruction::Print(Expression::Variable(value)),
            ],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
    let text = lowered.as_str();
    assert!(text.contains("llvm.call @__sev_variant_field"));
    assert!(text.contains("llvm.call @__sev_unbox_i64"));
    assert!(text.contains("llvm.call @printf"));
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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

    let lowered = severian_lowering::lower(&severian_mir::lower(&program).unwrap());
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
                any_origin: None,
                expression: Box::new(Expression::Call {
                    target: CallTarget::native("tensor.ranked_matmul", "__sev_tensor_matmul")
                        .with_signature([tensor, tensor], tensor),
                    args: vec![
                        Expression::Typed {
                            id: HirId::synthetic(2),
                            ty: tensor,
                            any_origin: None,
                            expression: Box::new(Expression::Variable("left".into())),
                        },
                        Expression::Typed {
                            id: HirId::synthetic(3),
                            ty: tensor,
                            any_origin: None,
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
