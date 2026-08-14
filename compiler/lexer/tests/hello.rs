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

#[test]
fn lexes_power_and_leading_decimal_literals() {
    let kinds = lex("value ** .5\n")
        .unwrap()
        .into_iter()
        .map(|token| token.kind)
        .collect::<Vec<_>>();

    assert_eq!(
        kinds,
        [
            TokenKind::Identifier("value".into()),
            TokenKind::Power,
            TokenKind::Float(0.5_f64.to_bits()),
            TokenKind::Newline,
            TokenKind::Eof,
        ]
    );
}

#[test]
fn lexes_triple_quoted_strings_with_preserved_newlines() {
    let source = "value = \"\"\"first\n  \"quoted\" second\n\"\"\"\n";
    let tokens = lex(source).unwrap();
    assert!(matches!(
        &tokens[2].kind,
        TokenKind::String(value) if value == "first\n  \"quoted\" second\n"
    ));
    assert_eq!(tokens[3].kind, TokenKind::Newline);
}

#[test]
fn lexes_an_empty_triple_quoted_string() {
    let tokens = lex("value = \"\"\"\"\"\"\n").unwrap();
    assert!(matches!(&tokens[2].kind, TokenKind::String(value) if value.is_empty()));
}

#[test]
fn lexes_formatted_triple_quoted_strings_with_preserved_newlines() {
    let source = "value = f\"\"\"model {name}\nversion {version}\n\"\"\"\n";
    let tokens = lex(source).unwrap();
    assert!(matches!(
        &tokens[2].kind,
        TokenKind::FormattedString(value)
            if value == "model {name}\nversion {version}\n"
    ));
    assert_eq!(tokens[3].kind, TokenKind::Newline);
}

#[test]
fn rejects_an_unterminated_formatted_triple_quoted_string() {
    let error = lex("value = f\"\"\"never closed\n").unwrap_err();
    assert!(error.message.contains("unterminated formatted triple-quoted"));
}

#[test]
fn rejects_an_unterminated_triple_quoted_string() {
    let error = lex("value = \"\"\"never closed\n").unwrap_err();
    assert!(error.message.contains("unterminated triple-quoted"));
}
