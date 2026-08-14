use severian_lexer::lex;
use severian_parser::parse;

#[test]
fn typed_bindings_require_an_initializer_at_the_declaration() {
    let source = "def main():\n    value: int\n    print(value)\n";
    let error = parse(&lex(source).unwrap()).unwrap_err();
    assert_eq!(
        error.message,
        "E000205: binding `value` requires an initializer"
    );
    assert_eq!(&source[error.span.start..error.span.end], "value");
}
