use severian_hir::{
    BinaryOp, Expression, Function, Instruction, MatchPattern, Program, TaskPlacement, ValueType,
};

#[test]
fn lowers_primitive_prints_to_native_calls() {
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
            instructions: vec![
                Instruction::Print(Expression::Integer(3)),
                Instruction::Print(Expression::Float(std::f64::consts::PI.to_bits())),
                Instruction::Print(Expression::Boolean(true)),
            ],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&program);
    let text = lowered.as_str();

    assert!(text.contains("llvm.call @printf"));
    assert!(text.contains("vararg(!llvm.func<i32 (!llvm.ptr, ...)>)"));
    assert!(text.contains("llvm.select"));
    assert!(text.contains("llvm.call @puts"));
}

#[test]
fn lowers_boolean_and_with_short_circuit_control_flow() {
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
            instructions: vec![Instruction::Print(Expression::Binary {
                left: Box::new(Expression::Boolean(true)),
                op: BinaryOp::And,
                right: Box::new(Expression::Boolean(true)),
            })],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&program);
    let text = lowered.as_str();
    assert!(text.contains("llvm.cond_br"));
    assert!(text.contains("llvm.br"));
    assert!(!text.contains("llvm.and"));
}

#[test]
fn lowers_conditional_expressions_to_native_selects() {
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
            instructions: vec![Instruction::Print(Expression::Conditional {
                condition: Box::new(Expression::Boolean(true)),
                then_expression: Box::new(Expression::Float(1.0f64.to_bits())),
                else_expression: Box::new(Expression::Float(0.0f64.to_bits())),
            })],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&program);
    assert!(lowered.as_str().contains("llvm.select"));
    assert!(lowered.as_str().contains(": i1, f64"));
}

#[test]
fn lowers_integer_range_for_to_control_flow() {
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
            instructions: vec![Instruction::For {
                setup: None,
                pattern: MatchPattern::Bind("value".into()),
                iterable: Expression::Call {
                    function: "range".into(),
                    args: vec![Expression::Integer(0), Expression::Integer(3)],
                },
                instructions: vec![Instruction::Print(Expression::Variable("value".into()))],
            }],
            tests: vec![],
        }],
    };

    let lowered = severian_lowering::lower(&program);
    let text = lowered.as_str();
    assert!(text.contains("llvm.icmp \"slt\""));
    assert!(text.contains("llvm.cond_br"));
    assert!(text.contains("llvm.br"));
}

#[test]
fn lowers_unit_function_calls_without_an_invalid_result() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![
            Function {
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
                name: "main".into(),
                native_symbol: None,
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Unit,
                instructions: vec![Instruction::Evaluate(Expression::Call {
                    function: "consume".into(),
                    args: vec![],
                })],
                tests: vec![],
            },
        ],
    };

    let lowered = severian_lowering::lower(&program);
    assert!(lowered
        .as_str()
        .contains("llvm.call @__sev_fn_consume() : () -> ()"));
    assert!(!lowered.as_str().contains("= llvm.call @__sev_fn_consume"));
}

#[test]
fn attaches_local_distribution_to_the_task_spawn_not_the_function() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![
            Function {
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
                            function: "work".into(),
                            args: vec![],
                        }),
                        placement: TaskPlacement::Local,
                    },
                }],
                tests: vec![],
            },
        ],
    };

    let lowered = severian_lowering::lower(&program);
    let text = lowered.as_str();
    assert!(text.contains("llvm.call @__sev_task_spawn_work() {severian_distribution = \"local\"}"));
    assert!(!text.contains("llvm.func @main() -> i32 attributes"));
}

#[test]
fn preserves_parallel_placement_on_spawn_calls() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![
            Function {
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
                            function: "work".into(),
                            args: vec![],
                        }),
                        placement: TaskPlacement::Gpu,
                    },
                }],
                tests: vec![],
            },
        ],
    };

    let lowered = severian_lowering::lower(&program);
    let text = lowered.as_str();
    assert!(text.contains("severian_parallel = \"gpu\""));
    assert!(text.contains("severian_device_fallback = \"cpu\""));
}

#[test]
fn compares_dynamic_collection_values_by_value() {
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

    let lowered = severian_lowering::lower(&program);
    assert!(lowered.as_str().contains("llvm.call @__sev_value_less"));
    assert!(!lowered.as_str().contains("llvm.icmp \"sgt\" %v"));
}
