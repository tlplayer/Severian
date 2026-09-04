#![forbid(unsafe_code)]

mod scanner;
mod token;

pub use scanner::scan;
pub use token::{Token, TokenKind};

#[cfg(test)]
mod tests {
    use super::*;
    use severian_source::SourceFile;

    #[test]
    fn scans_integer_binding_and_addition() {
        let source = SourceFile::virtual_source("test.sev", "b = 2\na = 1 + b\n");
        let tokens = scan(&source).unwrap();
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Plus));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Integer("2".into())));
    }

    #[test]
    fn scans_error_preserving_assignment() {
        let source = SourceFile::virtual_source("result.sev", "result ?= read()\n");
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::QuestionEqual));
    }

    #[test]
    fn scans_raw_address_operator() {
        let source = SourceFile::virtual_source("pointer.sev", "pointer := &value\n");
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Ampersand));
    }

    #[test]
    fn scans_primitive_bitwise_operators() {
        let source =
            SourceFile::virtual_source("bits.sev", "combined = left | right & mask ^ salt\n");
        let tokens = scan(&source).unwrap();
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Pipe));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Ampersand));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Caret));
    }

    #[test]
    fn scans_indented_trait_members_and_typed_operators() {
        let source = SourceFile::virtual_source(
            "primitive.sev",
            "trait i32: Primitive\n    property bits: int = 32\n    operator +(right: i32) -> i32\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Indent));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Arrow));
        assert!(tokens.iter().any(|token| token.kind == TokenKind::Dedent));
    }

    #[test]
    fn scans_unicode_characters_and_normalizes_numeric_separators() {
        let source = SourceFile::virtual_source(
            "literals.sev",
            "letter: char = '\u{03bb}'\nlarge: i64 = 1_000_000\nescaped: char = '\\n'\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Character('\u{03bb}')));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Character('\n')));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Integer("1000000".into())));
    }

    #[test]
    fn preserves_measured_number_magnitudes_and_suffixes() {
        let source =
            SourceFile::virtual_source(
                "sizes.sev",
                "zero = 0B\nbyte = 8B\nbinary = 4KiB\ndecimal = 2MB\n",
            );
        let tokens = scan(&source).unwrap();
        assert!(tokens.iter().any(|token| token.kind
            == TokenKind::MeasuredNumber {
                magnitude: "0".into(),
                suffix: "B".into(),
            }));
        assert!(tokens.iter().any(|token| token.kind
            == TokenKind::MeasuredNumber {
                magnitude: "8".into(),
                suffix: "B".into(),
            }));
        assert!(tokens.iter().any(|token| token.kind
            == TokenKind::MeasuredNumber {
                magnitude: "4".into(),
                suffix: "KiB".into(),
            }));
        assert!(tokens.iter().any(|token| token.kind
            == TokenKind::MeasuredNumber {
                magnitude: "2".into(),
                suffix: "MB".into(),
            }));
    }

    #[test]
    fn scans_scientific_notation_before_unit_suffixes() {
        let source = SourceFile::virtual_source(
            "scientific.sev",
            "small = 1e-8\nlarge = 2.5E+10\nunit = 1eV\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Float("1e-8".into())));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Float("2.5E+10".into())));
        assert!(tokens.iter().any(|token| token.kind
            == TokenKind::MeasuredNumber {
                magnitude: "1".into(),
                suffix: "eV".into(),
            }));
    }

    #[test]
    fn scans_hexadecimal_literals_as_canonical_integers() {
        let source = SourceFile::virtual_source("hex.sev", "0xFF 0xFE00_0000");
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Integer("255".into())));
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Integer("4261412864".into())));
    }

    #[test]
    fn quoted_strings_decode_common_escapes() {
        let source = SourceFile::virtual_source(
            "escaped-string.sev",
            "value = \"{\\\"enabled\\\": true}\\n\"\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| { token.kind == TokenKind::String("{\"enabled\": true}\n".into()) }));
    }

    #[test]
    fn block_strings_remove_structural_indentation_and_preserve_lines() {
        let source = SourceFile::virtual_source(
            "block-string.sev",
            "value = \"\"\"\n    first\n      second\n    third\n    \"\"\"\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| { token.kind == TokenKind::String("first\n  second\nthird\n".into()) }));
    }

    #[test]
    fn unterminated_block_strings_have_the_block_string_diagnostic() {
        let source = SourceFile::virtual_source("block-string.sev", "value = \"\"\"open\n");
        let error = scan(&source).unwrap_err();
        assert_eq!(error.code, "E000101");
        assert!(error.message.contains("block string"));
    }

    #[test]
    fn block_strings_preserve_unicode_and_blank_lines() {
        let source = SourceFile::virtual_source(
            "unicode-block.sev",
            "value = \"\"\"\n    λ\n\n    世界\n    \"\"\"\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::String("λ\n\n世界\n".into())));
    }

    #[test]
    fn block_string_tabs_have_the_same_width_as_source_indentation() {
        let source = SourceFile::virtual_source(
            "tabbed-block.sev",
            "value = \"\"\"\n\tfirst\n\t\tsecond\n    third\n\t\"\"\"\n",
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::String("first\n\tsecond\nthird\n".into())));
    }

    #[test]
    fn formatted_block_strings_are_single_tokens() {
        let source = SourceFile::virtual_source(
            "formatted.sev",
            r#"value = f"""module {{
{body}}}
"""
"#,
        );
        let tokens = scan(&source).unwrap();
        assert!(tokens.iter().any(|token| {
            token.kind == TokenKind::FormattedString("module {{\n{body}}}\n".into())
        }));
    }

    #[test]
    fn multiline_delimiters_do_not_change_the_layout_stack() {
        let source = SourceFile::virtual_source(
            "layout.sev",
            "def update():\n    apply({\n        \"x\": 1,\n        \"y\": 2,\n    })\nnext = 3\n",
        );
        let tokens = scan(&source).unwrap();
        assert_eq!(
            tokens
                .iter()
                .filter(|token| token.kind == TokenKind::Indent)
                .count(),
            1
        );
        assert_eq!(
            tokens
                .iter()
                .filter(|token| token.kind == TokenKind::Dedent)
                .count(),
            1
        );
    }

    #[test]
    fn source_defined_operator_is_one_open_operator_token() {
        let source = SourceFile::virtual_source("operator.sev", "left <<< right\n");
        let tokens = scan(&source).unwrap();
        assert!(tokens
            .iter()
            .any(|token| token.kind == TokenKind::Operator("<<<".into())));
    }
}
