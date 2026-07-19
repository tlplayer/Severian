use severian_hir::{Function, Instruction, Program};

#[test]
fn finds_the_main_function() {
    let program = Program {
        functions: vec![Function {
            name: "main".into(),
            instructions: vec![Instruction::Print("hello".into())],
        }],
    };

    assert_eq!(program.main().unwrap().name, "main");
}
