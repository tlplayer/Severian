use severian_lexer::{lex, TokenKind};

#[test]
fn lexes_the_hello_fixture() {
    let source = include_str!("../../../docs/examples/00-getting-started/01-hello.sev");
    let kinds: Vec<_> = lex(source)
        .unwrap()
        .into_iter()
        .map(|token| token.kind)
        .collect();

    assert_eq!(
        kinds,
        vec![
            TokenKind::Def,
            TokenKind::Identifier("main".into()),
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::Colon,
            TokenKind::Newline,
            TokenKind::Indent,
            TokenKind::Identifier("print".into()),
            TokenKind::LeftParen,
            TokenKind::String("hello, severian".into()),
            TokenKind::RightParen,
            TokenKind::Newline,
            TokenKind::Dedent,
            TokenKind::Test,
            TokenKind::With,
            TokenKind::Identifier("integration".into()),
            TokenKind::String("hello output".into()),
            TokenKind::Colon,
            TokenKind::Newline,
            TokenKind::Indent,
            TokenKind::Identifier("main".into()),
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::Newline,
            TokenKind::Assert,
            TokenKind::LeftParen,
            TokenKind::String("hello, severian".into()),
            TokenKind::In,
            TokenKind::Identifier("stdout".into()),
            TokenKind::RightParen,
            TokenKind::Newline,
            TokenKind::Dedent,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn rejects_inconsistent_indentation() {
    let error = lex("def main():\n    print(\"a\")\n  print(\"b\")\n").unwrap_err();
    assert!(error.message.contains("indentation"));
}
