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
fn parses_integration_tests_with_output_assertions() {
    let source = concat!(
        "def main():\n",
        "    print(\"hello\")\n",
        "\n",
        "test with integration \"native output\":\n",
        "    main()\n",
        "    assert(\"hello\" in stdout)\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    severian_parser::parse(&tokens).unwrap();
}

#[test]
fn parses_imported_decorator_symbol_packs() {
    let source = concat!(
        "import math\n",
        "\n",
        "@math(X, *, ^)\n",
        "def transform(value: int) -> int:\n",
        "    return value\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    let module = severian_parser::parse(&tokens).unwrap();
    let severian_ast::Item::Function(function) = &module.items[1] else {
        panic!("expected decorated function");
    };
    assert_eq!(function.decorators[0].name.segments[0].name, "math");
    assert_eq!(
        function.decorators[0]
            .symbols
            .iter()
            .map(|symbol| symbol.spelling.as_str())
            .collect::<Vec<_>>(),
        ["X", "*", "^"]
    );
}

#[test]
fn parses_server_signatures_destructuring_resources_and_contracts() {
    let source = concat!(
        "import network\n",
        "\n",
        "def serve(\n",
        "    connection: network.TcpConnection,\n",
        ") -> Result[unit, IOError] with {\n",
        "    connection != invalid,\n",
        "    with connection,\n",
        "}:\n",
        "    reader, writer = connection.split()\n",
        "    with connection:\n",
        "        while true with connection:\n",
        "            return\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    let module = severian_parser::parse(&tokens).unwrap();
    let severian_ast::Item::Function(function) = &module.items[1] else {
        panic!("expected function");
    };
    assert_eq!(
        function.params[0].ty.as_ref().unwrap().span().start,
        source.find("network.TcpConnection").unwrap()
    );
    assert!(function.contract.is_some());
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

#[test]
fn restricts_address_of_to_unsafe_blocks() {
    let valid = concat!(
        "def first(values: list[int]) -> int:\n",
        "    unsafe:\n",
        "        pointer = &values\n",
        "        return pointer[0]\n",
    );
    parse(&lex(valid).unwrap()).unwrap();

    let invalid = concat!(
        "def first(values: list[int]) -> int:\n",
        "    pointer = &values\n",
        "    return pointer[0]\n",
    );
    let error = parse(&lex(invalid).unwrap()).unwrap_err();
    assert_eq!(error.message, "address-of is only valid inside `unsafe`");
}
