use severian_hir::{BinaryOp, Expression, Function, Instruction, MatchPattern, Program, ValueType};

#[test]
fn lowers_primitive_prints_to_native_calls() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            name: "main".into(),
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
fn lowers_boolean_and_without_replacing_it_with_false() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            name: "main".into(),
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
    assert!(lowered.as_str().contains("llvm.and"));
}

#[test]
fn lowers_integer_range_for_to_control_flow() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            name: "main".into(),
            decorators: vec![],
            contract: None,
            params: vec![],
            return_type: ValueType::Unit,
            instructions: vec![Instruction::For {
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
                decorators: vec![],
                contract: None,
                params: vec![],
                return_type: ValueType::Unit,
                instructions: vec![],
                tests: vec![],
            },
            Function {
                name: "main".into(),
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
    assert!(lowered.as_str().contains("llvm.call @consume() : () -> ()"));
    assert!(!lowered.as_str().contains("= llvm.call @consume"));
}
