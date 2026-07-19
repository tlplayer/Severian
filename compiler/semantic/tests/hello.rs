use severian_hir::Instruction;
use severian_lexer::lex;
use severian_parser::parse;
use severian_semantic::analyze;

#[test]
fn resolves_print_and_lowers_hello_to_hir() {
    let source = include_str!("../../../docs/examples/00-getting-started/01-hello.sev");
    let ast = parse(&lex(source).unwrap()).unwrap();
    let hir = analyze(&ast).unwrap();

    assert_eq!(
        hir.main().unwrap().instructions,
        vec![Instruction::Print("hello, severian".into())]
    );
}

#[test]
fn rejects_unknown_functions() {
    let ast = parse(&lex("def main():\n    write(\"hello\")\n").unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "unknown function `write`");
}
