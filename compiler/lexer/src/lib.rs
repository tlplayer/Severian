#![forbid(unsafe_code)]

use severian_ast::Span;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Extern,
    Def,
    Class,
    Trait,
    Operator,
    Enum,
    Test,
    If,
    Elif,
    Else,
    Return,
    Throw,
    When,
    Assert,
    Switch,
    View,
    Borrow,
    Clone,
    Move,
    Async,
    Await,
    Send,
    Unsafe,
    Import,
    From,
    As,
    While,
    For,
    Break,
    Continue,
    In,
    With,
    Defer,
    And,
    Or,
    Not,
    True,
    False,
    Identifier(String),
    Integer(i64),
    Quantity(i64, String),
    Float(u64),
    String(String),
    FormattedString(String),
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Colon,
    Comma,
    Dot,
    Pipe,
    Ampersand,
    Arrow,
    Equal,
    ChangeableEqual,
    AddEqual,
    SubEqual,
    MulEqual,
    DivEqual,
    ModEqual,
    TryEqual,
    EqualEqual,
    NotEqual,
    Plus,
    Minus,
    Star,
    Power,
    Slash,
    Percent,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    At,
    Caret,
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for LexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at bytes {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for LexError {}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    Lexer::new(source).lex()
}

struct Lexer<'source> {
    source: &'source str,
    tokens: Vec<Token>,
    indents: Vec<usize>,
    delimiter_depth: usize,
}

impl<'source> Lexer<'source> {
    fn new(source: &'source str) -> Self {
        Self {
            source,
            tokens: Vec::new(),
            indents: vec![0],
            delimiter_depth: 0,
        }
    }

    fn lex(mut self) -> Result<Vec<Token>, LexError> {
        let mut offset = 0;
        while offset < self.source.len() {
            let remaining = &self.source[offset..];
            let segment_length = remaining
                .find('\n')
                .map_or(remaining.len(), |index| index + 1);
            let segment = &remaining[..segment_length];
            let line = segment.strip_suffix('\n').unwrap_or(segment);
            let content = line.trim_start_matches(' ');
            let indentation = line.len() - content.len();

            if content.starts_with('\t') {
                return Err(LexError {
                    span: Span::new(offset + indentation, offset + indentation + 1),
                    message: "E000102: tabs are not allowed for indentation".into(),
                });
            }

            if !content.is_empty() && !content.starts_with('#') {
                if self.delimiter_depth == 0 {
                    self.emit_indentation(indentation, offset)?;
                }
                let consumed = self.lex_line(content, offset + indentation)?;
                let (line_end, next_offset) = if let Some(end) = consumed {
                    let after = &self.source[end..];
                    let trailing_length = after.find('\n').unwrap_or(after.len());
                    let trailing = &after[..trailing_length];
                    if !trailing.trim().is_empty() {
                        self.lex_line(trailing, end)?;
                    }
                    let line_end = end + trailing_length;
                    let next_offset = line_end
                        + usize::from(after.as_bytes().get(trailing_length) == Some(&b'\n'));
                    (line_end, next_offset)
                } else {
                    (offset + line.len(), offset + segment.len())
                };
                if self.delimiter_depth == 0 {
                    self.tokens.push(Token {
                        kind: TokenKind::Newline,
                        span: Span::new(line_end, next_offset),
                    });
                }
                offset = next_offset;
            } else {
                offset += segment.len();
            }
        }

        while self.indents.len() > 1 {
            self.indents.pop();
            self.tokens.push(Token {
                kind: TokenKind::Dedent,
                span: Span::empty(self.source.len()),
            });
        }

        self.tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span::empty(self.source.len()),
        });

        Ok(self.tokens)
    }

    fn emit_indentation(&mut self, indentation: usize, offset: usize) -> Result<(), LexError> {
        let current = *self
            .indents
            .last()
            .expect("indentation stack is never empty");
        if indentation > current {
            self.indents.push(indentation);
            self.tokens.push(Token {
                kind: TokenKind::Indent,
                span: Span::new(offset, offset + indentation),
            });
        } else if indentation < current {
            while indentation < *self.indents.last().unwrap() {
                self.indents.pop();
                self.tokens.push(Token {
                    kind: TokenKind::Dedent,
                    span: Span::empty(offset + indentation),
                });
            }

            if indentation != *self.indents.last().unwrap() {
                return Err(LexError {
                    span: Span::new(offset, offset + indentation),
                    message: "E000102: indentation does not match an outer block".into(),
                });
            }
        }
        Ok(())
    }

    fn lex_line(&mut self, line: &str, base: usize) -> Result<Option<usize>, LexError> {
        let bytes = line.as_bytes();
        let mut index = 0;

        while index < bytes.len() {
            match bytes[index] {
                b' ' | b'\r' => index += 1,
                b'#' => break,
                b'(' => {
                    self.delimiter_depth += 1;
                    self.push_simple(TokenKind::LeftParen, base, &mut index);
                }
                b')' => {
                    self.delimiter_depth = self.delimiter_depth.saturating_sub(1);
                    self.push_simple(TokenKind::RightParen, base, &mut index);
                }
                b'[' => {
                    self.delimiter_depth += 1;
                    self.push_simple(TokenKind::LeftBracket, base, &mut index);
                }
                b']' => {
                    self.delimiter_depth = self.delimiter_depth.saturating_sub(1);
                    self.push_simple(TokenKind::RightBracket, base, &mut index);
                }
                b'{' => {
                    self.delimiter_depth += 1;
                    self.push_simple(TokenKind::LeftBrace, base, &mut index);
                }
                b'}' => {
                    self.delimiter_depth = self.delimiter_depth.saturating_sub(1);
                    self.push_simple(TokenKind::RightBrace, base, &mut index);
                }
                b':' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::ChangeableEqual, base, &mut index)
                }
                b':' => self.push_simple(TokenKind::Colon, base, &mut index),
                b',' => self.push_simple(TokenKind::Comma, base, &mut index),
                b'.' if bytes.get(index + 1).is_some_and(u8::is_ascii_digit) => {
                    let start = index;
                    index += 1;
                    while index < bytes.len() && bytes[index].is_ascii_digit() {
                        index += 1;
                    }
                    let text = &line[start..index];
                    self.tokens.push(Token {
                        kind: TokenKind::Float(
                            text.parse::<f64>().expect("digits form a float").to_bits(),
                        ),
                        span: Span::new(base + start, base + index),
                    });
                }
                b'.' => self.push_simple(TokenKind::Dot, base, &mut index),
                b'|' => self.push_simple(TokenKind::Pipe, base, &mut index),
                b'&' => self.push_simple(TokenKind::Ampersand, base, &mut index),
                b'@' => self.push_simple(TokenKind::At, base, &mut index),
                b'^' => self.push_simple(TokenKind::Caret, base, &mut index),
                b'+' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::AddEqual, base, &mut index)
                }
                b'+' => self.push_simple(TokenKind::Plus, base, &mut index),
                b'*' if bytes.get(index + 1) == Some(&b'*') => {
                    self.push_double(TokenKind::Power, base, &mut index)
                }
                b'*' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::MulEqual, base, &mut index)
                }
                b'*' => self.push_simple(TokenKind::Star, base, &mut index),
                b'/' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::DivEqual, base, &mut index)
                }
                b'/' => self.push_simple(TokenKind::Slash, base, &mut index),
                b'%' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::ModEqual, base, &mut index)
                }
                b'%' => self.push_simple(TokenKind::Percent, base, &mut index),
                b'?' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::TryEqual, base, &mut index)
                }
                b'-' if bytes.get(index + 1) == Some(&b'>') => {
                    self.push_double(TokenKind::Arrow, base, &mut index)
                }
                b'-' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::SubEqual, base, &mut index)
                }
                b'-' => self.push_simple(TokenKind::Minus, base, &mut index),
                b'=' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::EqualEqual, base, &mut index)
                }
                b'=' => self.push_simple(TokenKind::Equal, base, &mut index),
                b'!' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::NotEqual, base, &mut index)
                }
                b'<' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::LessEqual, base, &mut index)
                }
                b'<' => self.push_simple(TokenKind::Less, base, &mut index),
                b'>' if bytes.get(index + 1) == Some(&b'=') => {
                    self.push_double(TokenKind::GreaterEqual, base, &mut index)
                }
                b'>' => self.push_simple(TokenKind::Greater, base, &mut index),
                b'"' if bytes.get(index + 1) == Some(&b'"')
                    && bytes.get(index + 2) == Some(&b'"') =>
                {
                    let absolute_start = base + index;
                    let absolute_end = self.lex_triple_string(absolute_start)?;
                    if absolute_end <= base + line.len() {
                        index = absolute_end - base;
                    } else {
                        return Ok(Some(absolute_end));
                    }
                }
                b'"' => index = self.lex_string(line, base, index)?,
                b'f' if bytes.get(index + 1) == Some(&b'"')
                    && bytes.get(index + 2) == Some(&b'"')
                    && bytes.get(index + 3) == Some(&b'"') =>
                {
                    let absolute_start = base + index;
                    let absolute_end = self.lex_formatted_triple_string(absolute_start)?;
                    if absolute_end <= base + line.len() {
                        index = absolute_end - base;
                    } else {
                        return Ok(Some(absolute_end));
                    }
                }
                b'f' if bytes.get(index + 1) == Some(&b'"') => {
                    index = self.lex_formatted_string(line, base, index)?
                }
                byte if byte.is_ascii_digit() => {
                    let start = index;
                    index += 1;
                    while index < bytes.len() && bytes[index].is_ascii_digit() {
                        index += 1;
                    }
                    let is_float = bytes.get(index) == Some(&b'.')
                        && bytes.get(index + 1).is_some_and(u8::is_ascii_digit);
                    if is_float {
                        index += 1;
                        while index < bytes.len() && bytes[index].is_ascii_digit() {
                            index += 1;
                        }
                    }
                    let number_end = index;
                    while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
                        index += 1;
                    }
                    let text = &line[start..number_end];
                    let unit = &line[number_end..index];
                    self.tokens.push(Token {
                        kind: if !unit.is_empty() {
                            if is_float {
                                return Err(LexError {
                                    span: Span::new(base + start, base + index),
                                    message:
                                        "profile quantities currently require an integer value"
                                            .into(),
                                });
                            }
                            TokenKind::Quantity(
                                text.parse().expect("digits form an integer"),
                                unit.into(),
                            )
                        } else if is_float {
                            TokenKind::Float(
                                text.parse::<f64>().expect("digits form a float").to_bits(),
                            )
                        } else {
                            TokenKind::Integer(text.parse().expect("digits form an integer"))
                        },
                        span: Span::new(base + start, base + index),
                    });
                }
                byte if is_identifier_start(byte) => {
                    let start = index;
                    index += 1;
                    while index < bytes.len() && is_identifier_continue(bytes[index]) {
                        index += 1;
                    }
                    let value = &line[start..index];
                    let kind = match value {
                        "extern" => TokenKind::Extern,
                        "def" => TokenKind::Def,
                        "class" => TokenKind::Class,
                        "trait" => TokenKind::Trait,
                        "operator" => TokenKind::Operator,
                        "enum" => TokenKind::Enum,
                        "test" => TokenKind::Test,
                        "if" => TokenKind::If,
                        "elif" => TokenKind::Elif,
                        "else" => TokenKind::Else,
                        "return" => TokenKind::Return,
                        "throw" => TokenKind::Throw,
                        "when" => TokenKind::When,
                        "assert" => TokenKind::Assert,
                        "switch" => TokenKind::Switch,
                        "view" => TokenKind::View,
                        "borrow" => TokenKind::Borrow,
                        "clone" => TokenKind::Clone,
                        "move" => TokenKind::Move,
                        "async" => TokenKind::Async,
                        "await" => TokenKind::Await,
                        "send" => TokenKind::Send,
                        "unsafe" => TokenKind::Unsafe,
                        "import" => TokenKind::Import,
                        "from" => TokenKind::From,
                        "as" => TokenKind::As,
                        "while" => TokenKind::While,
                        "for" => TokenKind::For,
                        "break" => TokenKind::Break,
                        "continue" => TokenKind::Continue,
                        "in" => TokenKind::In,
                        "with" => TokenKind::With,
                        "defer" => TokenKind::Defer,
                        "and" => TokenKind::And,
                        "or" => TokenKind::Or,
                        "not" => TokenKind::Not,
                        "true" => TokenKind::True,
                        "false" => TokenKind::False,
                        _ => TokenKind::Identifier(value.into()),
                    };
                    self.tokens.push(Token {
                        kind,
                        span: Span::new(base + start, base + index),
                    });
                }
                _ => {
                    return Err(LexError {
                        span: Span::new(base + index, base + index + 1),
                        message: format!(
                            "unexpected character `{}`",
                            line[index..].chars().next().unwrap()
                        ),
                    });
                }
            }
        }

        Ok(None)
    }

    fn lex_triple_string(&mut self, start: usize) -> Result<usize, LexError> {
        let content_start = start + 3;
        let Some(relative_end) = self.source[content_start..].find("\"\"\"") else {
            return Err(LexError {
                span: Span::new(start, self.source.len()),
                message: "E000101: unterminated triple-quoted block string literal".into(),
            });
        };
        let content_end = content_start + relative_end;
        let end = content_end + 3;
        self.tokens.push(Token {
            kind: TokenKind::String(self.source[content_start..content_end].into()),
            span: Span::new(start, end),
        });
        Ok(end)
    }

    fn lex_formatted_triple_string(&mut self, start: usize) -> Result<usize, LexError> {
        let content_start = start + 4;
        let Some(relative_end) = self.source[content_start..].find("\"\"\"") else {
            return Err(LexError {
                span: Span::new(start, self.source.len()),
                message: "E000101: unterminated formatted triple-quoted block string literal"
                    .into(),
            });
        };
        let content_end = content_start + relative_end;
        let end = content_end + 3;
        self.tokens.push(Token {
            kind: TokenKind::FormattedString(self.source[content_start..content_end].into()),
            span: Span::new(start, end),
        });
        Ok(end)
    }

    fn push_simple(&mut self, kind: TokenKind, base: usize, index: &mut usize) {
        self.tokens.push(Token {
            kind,
            span: Span::new(base + *index, base + *index + 1),
        });
        *index += 1;
    }

    fn push_double(&mut self, kind: TokenKind, base: usize, index: &mut usize) {
        self.tokens.push(Token {
            kind,
            span: Span::new(base + *index, base + *index + 2),
        });
        *index += 2;
    }

    fn lex_string(&mut self, line: &str, base: usize, start: usize) -> Result<usize, LexError> {
        let bytes = line.as_bytes();
        let mut index = start + 1;
        let mut value = String::new();

        while index < bytes.len() {
            match bytes[index] {
                b'"' => {
                    index += 1;
                    self.tokens.push(Token {
                        kind: TokenKind::String(value),
                        span: Span::new(base + start, base + index),
                    });
                    return Ok(index);
                }
                b'\\' => {
                    index += 1;
                    if index == bytes.len() {
                        break;
                    }
                    let escaped = match bytes[index] {
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'"' => '"',
                        b'\\' => '\\',
                        other => {
                            return Err(LexError {
                                span: Span::new(base + index - 1, base + index + 1),
                                message: format!("unsupported escape `\\{}`", other as char),
                            });
                        }
                    };
                    value.push(escaped);
                    index += 1;
                }
                _ => {
                    let character = line[index..].chars().next().unwrap();
                    value.push(character);
                    index += character.len_utf8();
                }
            }
        }

        Err(LexError {
            span: Span::new(base + start, base + line.len()),
            message: "unterminated string literal".into(),
        })
    }

    fn lex_formatted_string(
        &mut self,
        line: &str,
        base: usize,
        start: usize,
    ) -> Result<usize, LexError> {
        let bytes = line.as_bytes();
        let content_start = start + 2;
        let mut index = content_start;
        let mut value = String::new();
        while index < bytes.len() {
            match bytes[index] {
                b'"' => {
                    index += 1;
                    self.tokens.push(Token {
                        kind: TokenKind::FormattedString(value),
                        span: Span::new(base + start, base + index),
                    });
                    return Ok(index);
                }
                b'\\' => {
                    index += 1;
                    if index == bytes.len() {
                        break;
                    }
                    let escaped = match bytes[index] {
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'"' => '"',
                        b'\\' => '\\',
                        other => {
                            return Err(LexError {
                                span: Span::new(base + index - 1, base + index + 1),
                                message: format!("unsupported escape `\\{}`", other as char),
                            });
                        }
                    };
                    value.push(escaped);
                    index += 1;
                }
                _ => {
                    let character = line[index..].chars().next().unwrap();
                    value.push(character);
                    index += character.len_utf8();
                }
            }
        }
        Err(LexError {
            span: Span::new(base + start, base + line.len()),
            message: "unterminated formatted string literal".into(),
        })
    }
}

fn is_identifier_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

fn is_identifier_continue(byte: u8) -> bool {
    is_identifier_start(byte) || byte.is_ascii_digit()
}
