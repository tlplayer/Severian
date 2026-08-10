use severian_hir::{
    CallTarget, Expression, Function, FunctionId, Instruction, OwnershipOp, Parameter, Program,
    ValueType,
};

fn function(name: &str, params: &[&str], instructions: Vec<Instruction>) -> Function {
    Function {
        id: FunctionId::from_name(name),
        name: name.into(),
        native_symbol: None,
        decorators: vec![],
        contract: None,
        params: params
            .iter()
            .map(|name| Parameter {
                name: (*name).into(),
                ty: ValueType::Any,
                default: None,
            })
            .collect(),
        return_type: ValueType::Unit,
        instructions,
        tests: vec![],
    }
}

fn program(instructions: Vec<Instruction>) -> Program {
    Program {
        globals: vec![],
        classes: vec![],
        functions: vec![function("main", &[], instructions)],
    }
}

fn ownership(op: OwnershipOp, name: &str) -> Expression {
    Expression::Ownership {
        op,
        value: Box::new(Expression::Variable(name.into())),
    }
}

#[test]
fn accepts_owned_string_literals() {
    let program = program(vec![Instruction::Print(Expression::String("hello".into()))]);

    severian_ownership::check(&program).unwrap();
}

#[test]
fn rejects_use_after_move() {
    let program = program(vec![
        Instruction::Let {
            name: "value".into(),
            value: Expression::List(vec![]),
        },
        Instruction::Let {
            name: "owned".into(),
            value: ownership(OwnershipOp::Move, "value"),
        },
        Instruction::Print(Expression::Variable("value".into())),
    ]);

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0301"));
    assert!(error.message.contains("value"));
}

#[test]
fn rejects_mutation_while_shared_view_is_live() {
    let program = program(vec![
        Instruction::Let {
            name: "values".into(),
            value: Expression::List(vec![]),
        },
        Instruction::Let {
            name: "values_view".into(),
            value: ownership(OwnershipOp::View, "values"),
        },
        Instruction::Evaluate(Expression::MethodCall {
            object: Box::new(Expression::Variable("values".into())),
            method: "push".into(),
            args: vec![Expression::Integer(1)],
        }),
        Instruction::Print(Expression::Variable("values_view".into())),
    ]);

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0302"));
}

#[test]
fn permits_mutation_after_shared_views_last_use() {
    let program = program(vec![
        Instruction::Let {
            name: "values".into(),
            value: Expression::List(vec![]),
        },
        Instruction::Let {
            name: "values_view".into(),
            value: ownership(OwnershipOp::View, "values"),
        },
        Instruction::Print(Expression::Variable("values_view".into())),
        Instruction::Evaluate(Expression::MethodCall {
            object: Box::new(Expression::Variable("values".into())),
            method: "push".into(),
            args: vec![Expression::Integer(1)],
        }),
    ]);

    severian_ownership::check(&program).unwrap();
}

#[test]
fn rejects_two_live_exclusive_borrows() {
    let program = program(vec![
        Instruction::Let {
            name: "values".into(),
            value: Expression::List(vec![]),
        },
        Instruction::Let {
            name: "first".into(),
            value: ownership(OwnershipOp::Borrow, "values"),
        },
        Instruction::Let {
            name: "second".into(),
            value: ownership(OwnershipOp::Borrow, "values"),
        },
        Instruction::Print(Expression::Variable("first".into())),
        Instruction::Print(Expression::Variable("second".into())),
    ]);

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0303"));
}

#[test]
fn rejects_use_after_move_on_only_one_branch() {
    let program = program(vec![
        Instruction::Let {
            name: "value".into(),
            value: Expression::List(vec![]),
        },
        Instruction::If {
            condition: Expression::Boolean(true),
            then_instructions: vec![Instruction::Let {
                name: "owned".into(),
                value: ownership(OwnershipOp::Move, "value"),
            }],
            else_instructions: vec![],
        },
        Instruction::Print(Expression::Variable("value".into())),
    ]);

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0301"));
}

#[test]
fn infers_that_a_function_consumes_its_parameter() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![
            function(
                "consume",
                &["input"],
                vec![Instruction::Let {
                    name: "owned".into(),
                    value: ownership(OwnershipOp::Move, "input"),
                }],
            ),
            function(
                "main",
                &[],
                vec![
                    Instruction::Let {
                        name: "value".into(),
                        value: Expression::List(vec![]),
                    },
                    Instruction::Evaluate(Expression::Call {
                        target: CallTarget::source("consume"),
                        args: vec![Expression::Variable("value".into())],
                    }),
                    Instruction::Print(Expression::Variable("value".into())),
                ],
            ),
        ],
    };

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0301"));
}

#[test]
fn inferred_mutable_call_conflicts_with_a_live_view() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![
            function(
                "update",
                &["values"],
                vec![Instruction::Evaluate(Expression::MethodCall {
                    object: Box::new(Expression::Variable("values".into())),
                    method: "push".into(),
                    args: vec![Expression::Integer(1)],
                })],
            ),
            function(
                "main",
                &[],
                vec![
                    Instruction::Let {
                        name: "values".into(),
                        value: Expression::List(vec![]),
                    },
                    Instruction::Let {
                        name: "snapshot".into(),
                        value: ownership(OwnershipOp::View, "values"),
                    },
                    Instruction::Evaluate(Expression::Call {
                        target: CallTarget::source("update"),
                        args: vec![Expression::Variable("values".into())],
                    }),
                    Instruction::Print(Expression::Variable("snapshot".into())),
                ],
            ),
        ],
    };

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0303"));
}

#[test]
fn call_argument_loans_overlap_for_the_whole_call() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![
            function("pair", &["left", "right"], vec![]),
            function(
                "main",
                &[],
                vec![
                    Instruction::Let {
                        name: "value".into(),
                        value: Expression::List(vec![]),
                    },
                    Instruction::Evaluate(Expression::Call {
                        target: CallTarget::source("pair"),
                        args: vec![
                            ownership(OwnershipOp::Borrow, "value"),
                            ownership(OwnershipOp::View, "value"),
                        ],
                    }),
                ],
            ),
        ],
    };

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0303"));
}

#[test]
fn borrowed_alias_cannot_escape_through_return() {
    let mut escaping = function(
        "escaping",
        &["value"],
        vec![
            Instruction::Let {
                name: "alias".into(),
                value: ownership(OwnershipOp::View, "value"),
            },
            Instruction::Return(Some(Expression::Variable("alias".into()))),
        ],
    );
    escaping.return_type = ValueType::Any;
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![escaping],
    };

    let error = severian_ownership::check(&program).unwrap_err();
    assert!(error.message.contains("E0305"));
}
