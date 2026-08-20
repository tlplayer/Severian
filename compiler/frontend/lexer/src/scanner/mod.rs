use crate::{Token, TokenKind};
use severian_diagnostics::Diagnostic;
use severian_source::{SourceFile, Span};

pub fn scan(source: &SourceFile) -> Result<Vec<Token>, Diagnostic> {
    let bytes = source.text.as_bytes();
    let mut cursor = 0usize;
    let mut line_start = true;
    let mut indents = vec![0usize];
    let mut tokens = Vec::new();

    while cursor < bytes.len() {
        if line_start {
            let start = cursor;
            while cursor < bytes.len() && matches!(bytes[cursor], b' ' | b'\t') {
                cursor += 1;
            }
            if cursor >= bytes.len() {
                break;
            }
            if bytes[cursor] == b'\n' || bytes[cursor] == b'#' {
                if bytes[cursor] == b'#' {
                    while cursor < bytes.len() && bytes[cursor] != b'\n' {
                        cursor += 1;
                    }
                }
                if cursor < bytes.len() {
                    cursor += 1;
                    tokens.push(token(source, TokenKind::Newline, cursor - 1, cursor));
                }
                line_start = true;
                continue;
            }
            let width: usize = source.text[start..cursor]
                .chars()
                .map(|character| if character == '\t' { 4 } else { 1 })
                .sum();
            match width.cmp(indents.last().expect("indent stack is nonempty")) {
                std::cmp::Ordering::Greater => {
                    indents.push(width);
                    tokens.push(token(source, TokenKind::Indent, start, cursor));
                }
                std::cmp::Ordering::Less => {
                    while width < *indents.last().expect("indent stack is nonempty") {
                        indents.pop();
                        tokens.push(token(source, TokenKind::Dedent, start, cursor));
                    }
                    if width != *indents.last().expect("indent stack is nonempty") {
                        return Err(Diagnostic::new(
                            "E000102",
                            "inconsistent indentation",
                            Some(Span::new(source.id, start as u32, cursor as u32)),
                        ));
                    }
                }
                std::cmp::Ordering::Equal => {}
            }
            line_start = false;
        }

        let start = cursor;
        let kind = match bytes[cursor] {
            b' ' | b'\t' | b'\r' => {
                cursor += 1;
                continue;
            }
            b'\n' => {
                cursor += 1;
                line_start = true;
                TokenKind::Newline
            }
            b';' => {
                cursor += 1;
                TokenKind::Newline
            }
            b'@' => one(&mut cursor, TokenKind::At),
            b',' => one(&mut cursor, TokenKind::Comma),
            b':' => one(&mut cursor, TokenKind::Colon),
            b'(' => one(&mut cursor, TokenKind::LeftParen),
            b')' => one(&mut cursor, TokenKind::RightParen),
            b'[' => one(&mut cursor, TokenKind::LeftBracket),
            b']' => one(&mut cursor, TokenKind::RightBracket),
            b'|' => one(&mut cursor, TokenKind::Pipe),
            b'+' => one(&mut cursor, TokenKind::Plus),
            b'%' => one(&mut cursor, TokenKind::Percent),
            b'*' if bytes.get(cursor + 1) == Some(&b'*') => {
                cursor += 2;
                TokenKind::Power
            }
            b'*' => one(&mut cursor, TokenKind::Star),
            b'/' => one(&mut cursor, TokenKind::Slash),
            b'-' if bytes.get(cursor + 1) == Some(&b'>') => {
                cursor += 2;
                TokenKind::Arrow
            }
            b'-' => one(&mut cursor, TokenKind::Minus),
            b'=' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::EqualEqual
            }
            b'=' => one(&mut cursor, TokenKind::Equal),
            b'!' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::NotEqual
            }
            b'<' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::LessEqual
            }
            b'<' => one(&mut cursor, TokenKind::Less),
            b'>' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::GreaterEqual
            }
            b'>' => one(&mut cursor, TokenKind::Greater),
            b'"' => {
                cursor += 1;
                let content_start = cursor;
                while cursor < bytes.len() && bytes[cursor] != b'"' {
                    if bytes[cursor] == b'\n' {
                        return Err(Diagnostic::new(
                            "E000101",
                            "unterminated string literal",
                            Some(Span::new(source.id, start as u32, cursor as u32)),
                        ));
                    }
                    cursor += 1;
                }
                if cursor == bytes.len() {
                    return Err(Diagnostic::new(
                        "E000101",
                        "unterminated string literal",
                        Some(Span::new(source.id, start as u32, cursor as u32)),
                    ));
                }
                let value = source.text[content_start..cursor].to_owned();
                cursor += 1;
                TokenKind::String(value)
            }
            b'#' => {
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
                if bytes.get(cursor) == Some(&b'.')
                    && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit)
                {
                    cursor += 1;
                    while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                        cursor += 1;
                    }
                    TokenKind::Float(source.text[start..cursor].to_owned())
                } else {
                    TokenKind::Integer(source.text[start..cursor].to_owned())
                }
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
        tokens.push(token(source, kind, start, cursor));
    }
    while indents.len() > 1 {
        indents.pop();
        tokens.push(token(source, TokenKind::Dedent, cursor, cursor));
    }
    tokens.push(token(source, TokenKind::Eof, cursor, cursor));
    Ok(tokens)
}

fn one(cursor: &mut usize, kind: TokenKind) -> TokenKind {
    *cursor += 1;
    kind
}

fn token(source: &SourceFile, kind: TokenKind, start: usize, end: usize) -> Token {
    Token {
        kind,
        span: Span::new(source.id, start as u32, end as u32),
    }
}
