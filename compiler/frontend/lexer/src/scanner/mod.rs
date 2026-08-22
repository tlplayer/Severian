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
            b'.' => one(&mut cursor, TokenKind::Dot),
            b':' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::ColonEqual
            }
            b':' => one(&mut cursor, TokenKind::Colon),
            b'(' => one(&mut cursor, TokenKind::LeftParen),
            b')' => one(&mut cursor, TokenKind::RightParen),
            b'[' => one(&mut cursor, TokenKind::LeftBracket),
            b']' => one(&mut cursor, TokenKind::RightBracket),
            b'|' => one(&mut cursor, TokenKind::Pipe),
            b'+' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::PlusEqual
            }
            b'+' => one(&mut cursor, TokenKind::Plus),
            b'%' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::PercentEqual
            }
            b'%' => one(&mut cursor, TokenKind::Percent),
            b'*' if bytes.get(cursor + 1) == Some(&b'*') => {
                cursor += 2;
                TokenKind::Power
            }
            b'*' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::StarEqual
            }
            b'*' => one(&mut cursor, TokenKind::Star),
            b'/' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::SlashEqual
            }
            b'/' => one(&mut cursor, TokenKind::Slash),
            b'-' if bytes.get(cursor + 1) == Some(&b'>') => {
                cursor += 2;
                TokenKind::Arrow
            }
            b'-' if bytes.get(cursor + 1) == Some(&b'=') => {
                cursor += 2;
                TokenKind::MinusEqual
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
            b'f' if bytes.get(cursor + 1..cursor + 4) == Some(b"\"\"\"") => {
                cursor += 4;
                let content_start = cursor;
                while cursor + 2 < bytes.len() && bytes.get(cursor..cursor + 3) != Some(b"\"\"\"") {
                    cursor += 1;
                }
                if cursor + 2 >= bytes.len() {
                    return Err(Diagnostic::new(
                        "E000101",
                        "unterminated formatted block string literal",
                        Some(Span::new(source.id, start as u32, bytes.len() as u32)),
                    ));
                }
                let value = block_string(&source.text[content_start..cursor]);
                cursor += 3;
                TokenKind::FormattedString(value)
            }
            b'"' => {
                let block = bytes.get(cursor..cursor + 3) == Some(b"\"\"\"");
                if block {
                    cursor += 3;
                    let content_start = cursor;
                    while cursor + 2 < bytes.len()
                        && bytes.get(cursor..cursor + 3) != Some(b"\"\"\"")
                    {
                        cursor += 1;
                    }
                    if cursor + 2 >= bytes.len() {
                        return Err(Diagnostic::new(
                            "E000101",
                            "unterminated block string literal",
                            Some(Span::new(source.id, start as u32, bytes.len() as u32)),
                        ));
                    }
                    let value = block_string(&source.text[content_start..cursor]);
                    cursor += 3;
                    tokens.push(token(source, TokenKind::String(value), start, cursor));
                    continue;
                }
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
            b'\'' => {
                cursor += 1;
                let value = if bytes.get(cursor) == Some(&b'\\') {
                    cursor += 1;
                    let escaped = match bytes.get(cursor).copied() {
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'0') => '\0',
                        Some(b'\\') => '\\',
                        Some(b'\'') => '\'',
                        Some(b'"') => '"',
                        _ => return Err(character_error(source, start, cursor)),
                    };
                    cursor += 1;
                    escaped
                } else {
                    let Some(character) = source.text[cursor..].chars().next() else {
                        return Err(character_error(source, start, cursor));
                    };
                    if character == '\n' || character == '\'' {
                        return Err(character_error(source, start, cursor));
                    }
                    cursor += character.len_utf8();
                    character
                };
                if bytes.get(cursor) != Some(&b'\'') {
                    return Err(character_error(source, start, cursor));
                }
                cursor += 1;
                TokenKind::Character(value)
            }
            b'#' => {
                while cursor < bytes.len() && bytes[cursor] != b'\n' {
                    cursor += 1;
                }
                continue;
            }
            byte if byte.is_ascii_digit() => {
                cursor += 1;
                while cursor < bytes.len()
                    && (bytes[cursor].is_ascii_digit()
                        || (bytes[cursor] == b'_'
                            && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit)))
                {
                    cursor += 1;
                }
                if bytes.get(cursor) == Some(&b'.')
                    && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit)
                {
                    cursor += 1;
                    while cursor < bytes.len()
                        && (bytes[cursor].is_ascii_digit()
                            || (bytes[cursor] == b'_'
                                && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit)))
                    {
                        cursor += 1;
                    }
                    TokenKind::Float(source.text[start..cursor].replace('_', ""))
                } else {
                    TokenKind::Integer(source.text[start..cursor].replace('_', ""))
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

fn character_error(source: &SourceFile, start: usize, cursor: usize) -> Diagnostic {
    Diagnostic::new(
        "E000103",
        "character literals contain exactly one Unicode scalar value",
        Some(Span::new(
            source.id,
            start as u32,
            cursor.min(source.text.len()) as u32,
        )),
    )
}

fn block_string(raw: &str) -> String {
    let content = raw.strip_prefix('\n').unwrap_or(raw);
    let common_indent = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(indentation_width)
        .min()
        .unwrap_or(0);
    content
        .split('\n')
        .map(|line| strip_indentation(line, common_indent))
        .collect::<Vec<_>>()
        .join("\n")
}

fn indentation_width(line: &str) -> usize {
    line.chars()
        .take_while(|character| matches!(character, ' ' | '\t'))
        .map(|character| if character == '\t' { 4 } else { 1 })
        .sum()
}

fn strip_indentation(line: &str, width: usize) -> &str {
    let mut consumed = 0usize;
    let mut byte_offset = 0usize;
    for (offset, character) in line.char_indices() {
        if consumed >= width || !matches!(character, ' ' | '\t') {
            byte_offset = offset;
            break;
        }
        consumed += if character == '\t' { 4 } else { 1 };
        byte_offset = offset + character.len_utf8();
    }
    &line[byte_offset..]
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
