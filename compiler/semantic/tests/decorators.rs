#[test]
fn retains_decorator_symbol_packs_after_validating_the_import() {
    let source = concat!(
        "import math\n",
        "\n",
        "@math(X, *, ^)\n",
        "def transform(value: int) -> int:\n",
        "    return value\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    let module = severian_parser::parse(&tokens).unwrap();
    let program = severian_semantic::analyze(&module).unwrap();

    assert_eq!(program.functions[0].decorators[0].package, "math");
    assert_eq!(program.functions[0].decorators[0].symbols, ["X", "*", "^"]);
}

#[test]
fn rejects_a_decorator_whose_package_was_not_imported() {
    let source = concat!(
        "@math(X)\n",
        "def transform(value: int) -> int:\n",
        "    return value\n",
    );
    let tokens = severian_lexer::lex(source).unwrap();
    let module = severian_parser::parse(&tokens).unwrap();
    let error = severian_semantic::analyze(&module).unwrap_err();

    assert!(error
        .message
        .contains("decorator package `math` must be imported"));
}
