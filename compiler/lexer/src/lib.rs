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
    Def,
    Identifier(String),
    String(String),
    LeftParen,
    RightParen,
    Colon,
    Comma,
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
}

impl<'source> Lexer<'source> {
    fn new(source: &'source str) -> Self {
        Self {
            source,
            tokens: Vec::new(),
            indents: vec![0],
        }
    }

    fn lex(mut self) -> Result<Vec<Token>, LexError> {
        let mut offset = 0;

        for segment in self.source.split_inclusive('\n') {
            let line = segment.strip_suffix('\n').unwrap_or(segment);
            let content = line.trim_start_matches(' ');
            let indentation = line.len() - content.len();

            if content.starts_with('\t') {
                return Err(LexError {
                    span: Span::new(offset + indentation, offset + indentation + 1),
                    message: "tabs are not allowed for indentation".into(),
                });
            }

            if !content.is_empty() && !content.starts_with('#') {
                self.emit_indentation(indentation, offset)?;
                self.lex_line(content, offset + indentation)?;
                let end = offset + line.len();
                self.tokens.push(Token {
                    kind: TokenKind::Newline,
                    span: Span::new(end, end + usize::from(segment.ends_with('\n'))),
                });
            }

            offset += segment.len();
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
                    message: "indentation does not match an outer block".into(),
                });
            }
        }
        Ok(())
    }

    fn lex_line(&mut self, line: &str, base: usize) -> Result<(), LexError> {
        let bytes = line.as_bytes();
        let mut index = 0;

        while index < bytes.len() {
            match bytes[index] {
                b' ' | b'\r' => index += 1,
                b'#' => break,
                b'(' => self.push_simple(TokenKind::LeftParen, base, &mut index),
                b')' => self.push_simple(TokenKind::RightParen, base, &mut index),
                b':' => self.push_simple(TokenKind::Colon, base, &mut index),
                b',' => self.push_simple(TokenKind::Comma, base, &mut index),
                b'"' => index = self.lex_string(line, base, index)?,
                byte if is_identifier_start(byte) => {
                    let start = index;
                    index += 1;
                    while index < bytes.len() && is_identifier_continue(bytes[index]) {
                        index += 1;
                    }
                    let value = &line[start..index];
                    let kind = if value == "def" {
                        TokenKind::Def
                    } else {
                        TokenKind::Identifier(value.into())
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

        Ok(())
    }

    fn push_simple(&mut self, kind: TokenKind, base: usize, index: &mut usize) {
        self.tokens.push(Token {
            kind,
            span: Span::new(base + *index, base + *index + 1),
        });
        *index += 1;
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
}

fn is_identifier_start(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

fn is_identifier_continue(byte: u8) -> bool {
    is_identifier_start(byte) || byte.is_ascii_digit()
}
