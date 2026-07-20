use severian_hir::{Expression, Function, Instruction, Program, ValueType};

#[test]
fn finds_the_main_function() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            name: "main".into(),
            decorators: vec![],
            contract: None,
            params: vec![],
            return_type: ValueType::Unit,
            instructions: vec![Instruction::Print(Expression::String("hello".into()))],
            tests: vec![],
        }],
    };

    assert_eq!(program.main().unwrap().name, "main");
}
