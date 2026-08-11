use severian_hir::{Expression, Function, FunctionId, Instruction, Program, ValueType};

#[test]
fn creates_explicit_branch_blocks_with_typed_value_references() {
    let condition = Expression::Typed {
        id: severian_hir::HirId::synthetic(1),
        ty: ValueType::Bool,
        expression: Box::new(Expression::Boolean(true)),
    };
    let hir = Program {
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

    let mir = severian_mir::lower(&hir);
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
}
