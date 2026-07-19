use severian_hir::{Expression, Instruction};
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
        vec![Instruction::Print(Expression::String(
            "hello, severian".into()
        ))]
    );
}

#[test]
fn rejects_unknown_functions() {
    let ast = parse(&lex("def main():\n    write(\"hello\")\n").unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "unknown function `write`");
}

#[test]
fn rejects_snake_case_function_names() {
    let ast = parse(&lex("def bad_name():\n    print(\"hello\")\n").unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "function `bad_name` must use lowerCamelCase");
}

#[test]
fn rejects_a_typed_function_that_does_not_return_on_every_path() {
    let source = "def choose(ready: bool) -> int:\n    if ready:\n        return 1\n";
    let ast = parse(&lex(source).unwrap()).unwrap();
    let error = analyze(&ast).unwrap_err();
    assert_eq!(error.message, "function `choose` must return a value");
}

#[test]
fn type_checks_injected_return_values() {
    let source = concat!(
        "def read() -> int:\n",
        "    return 1\n",
        "\n",
        "test:\n",
        "    chaos.add(when read return \"not an int\")\n",
    );
    let ast = parse(&lex(source).unwrap()).unwrap();

    let error = analyze(&ast).unwrap_err();

    assert_eq!(error.message, "expected Int, found String");
}
