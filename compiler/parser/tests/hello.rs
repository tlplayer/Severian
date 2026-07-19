use severian_ast::{Expr, Item, Literal, Stmt};
use severian_lexer::lex;
use severian_parser::parse;

#[test]
fn parses_the_hello_fixture_into_the_source_ast() {
    let source = include_str!("../../../docs/examples/00-getting-started/01-hello.sev");
    let tokens = lex(source).unwrap();
    let module = parse(&tokens).unwrap();

    let Item::Function(main) = &module.items[0] else {
        panic!("expected a function");
    };
    assert_eq!(main.name.name, "main");

    let Stmt::Expr(Expr::Call(call)) = &main.body.statements[0] else {
        panic!("expected a call statement");
    };
    let Expr::Identifier(callee) = call.callee.as_ref() else {
        panic!("expected an identifier callee");
    };
    assert_eq!(callee.name, "print");
    assert!(matches!(
        &call.args[0].value,
        Expr::Literal(Literal::String { value, .. }) if value == "hello, severian"
    ));
}

#[test]
fn rejects_a_missing_function_body() {
    let tokens = lex("def main():\n").unwrap();
    let error = parse(&tokens).unwrap_err();
    assert!(error.message.contains("indented function body"));
}
