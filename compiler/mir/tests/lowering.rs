use severian_hir::{Expression, Function, FunctionId, Instruction, Program, ValueType};

#[test]
fn creates_explicit_branch_blocks_with_typed_value_references() {
    let condition = Expression::Typed {
        id: severian_hir::HirId::from_source_range(0, 4),
        ty: ValueType::Bool,
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
