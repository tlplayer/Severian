use severian_hir::{Expression, Function, Instruction, Program, ValueType};

#[test]
fn accepts_owned_string_literals() {
    let program = Program {
        globals: vec![],
        classes: vec![],
        functions: vec![Function {
            name: "main".into(),
            params: vec![],
            return_type: ValueType::Unit,
            instructions: vec![Instruction::Print(Expression::String("hello".into()))],
            tests: vec![],
        }],
    };

    severian_ownership::check(&program).unwrap();
}
