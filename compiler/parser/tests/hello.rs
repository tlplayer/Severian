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

#[test]
fn accepts_chaos_injection_patterns_in_any_test_block() {
    let source = concat!(
        "def read():\n",
        "    return None\n",
        "\n",
        "test:\n",
        "    chaos.add(when read return None)\n",
        "\n",
        "test with chaos \"throws\":\n",
        "    chaos.add(when read throw TimedOut)\n",
    );

    parse(&lex(source).unwrap()).unwrap();
}

#[test]
fn rejects_chaos_injection_patterns_outside_tests() {
    let source = concat!(
        "def read():\n",
        "    return None\n",
        "\n",
        "def main():\n",
        "    chaos.add(when read return None)\n",
    );

    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert_eq!(
        error.message,
        "chaos injection patterns are only valid inside tests"
    );
}
