use crate::{Token, TokenKind};
use severian_diagnostics::Diagnostic;
use severian_source::{SourceFile, Span};

pub fn scan(source: &SourceFile) -> Result<Vec<Token>, Diagnostic> {
    let bytes = source.text.as_bytes();
    let mut cursor = 0usize;
    let mut tokens = Vec::new();
    while cursor < bytes.len() {
        let start = cursor;
        let kind = match bytes[cursor] {
            b' ' | b'\t' | b'\r' => {
                cursor += 1;
                continue;
            }
            b'\n' | b',' | b';' => {
                cursor += 1;
                TokenKind::Separator
            }
            b'=' => {
                cursor += 1;
                TokenKind::Equal
            }
            b':' => {
                cursor += 1;
                TokenKind::Colon
            }
            b'+' => {
                cursor += 1;
                TokenKind::Plus
            }
            b'#' => {
                cursor += 1;
                while cursor < bytes.len() && bytes[cursor] != b'\n' {
                    cursor += 1;
                }
                continue;
            }
            byte if byte.is_ascii_digit() => {
                cursor += 1;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }
                let spelling = &source.text[start..cursor];
                let value = spelling.parse::<i64>().map_err(|_| {
                    Diagnostic::new(
                        "E000105",
                        format!("integer literal `{spelling}` is outside the supported int range"),
                        Some(Span::new(source.id, start as u32, cursor as u32)),
                    )
                })?;
                TokenKind::Integer(value)
            }
            byte if byte.is_ascii_alphabetic() || byte == b'_' => {
                cursor += 1;
                while cursor < bytes.len()
                    && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
                {
                    cursor += 1;
                }
                TokenKind::Identifier(source.text[start..cursor].to_owned())
            }
            other => {
                return Err(Diagnostic::new(
                    "E000100",
                    format!("unexpected character `{}`", char::from(other)),
                    Some(Span::new(source.id, start as u32, (start + 1) as u32)),
                ))
            }
        };
        tokens.push(Token {
            kind,
            span: Span::new(source.id, start as u32, cursor as u32),
        });
    }
    let end = source.text.len() as u32;
    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(source.id, end, end),
    });
    Ok(tokens)
}
