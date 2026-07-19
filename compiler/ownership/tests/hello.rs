use severian_hir::{Function, Instruction, Program};

#[test]
fn accepts_owned_string_literals() {
    let program = Program {
        functions: vec![Function {
            name: "main".into(),
            instructions: vec![Instruction::Print("hello".into())],
        }],
    };

    severian_ownership::check(&program).unwrap();
}
