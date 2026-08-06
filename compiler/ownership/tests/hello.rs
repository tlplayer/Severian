use severian_hir::{Expression, Function, Instruction, OwnershipOp, Program, ValueType};

fn program(instructions: Vec<Instruction>) -> Program {
    Program {
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            name: "main".into(),
            native_symbol: None,
            decorators: vec![],
            contract: None,
            params: vec![],
            return_type: ValueType::Unit,
            instructions,
            tests: vec![],
        }],
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
