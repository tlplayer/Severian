"""Syntax-driven tokenization. Recognition does not evaluate literals."""
from dataclasses import dataclass
from py_compiler.frontend.source.source import Diagnostic, SourceFile, Span
from py_compiler.syntax.recognition import Syntax


@dataclass(frozen=True)
class Lexeme:
    text: str
    span: Span


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: Lexeme

    @property
    def text(self):
        return self.lexeme.text

    @property
    def span(self):
        return self.lexeme.span


def lex(source, syntax):
    text, cursor, tokens = source.text, 0, []
    while cursor < len(text):
        start = cursor
        if text[cursor] in " \t":
            while cursor < len(text) and text[cursor] in " \t":
                cursor += 1
            if start == 0 or text[start - 1] in "\r\n":
                tokens.append(Token("INDENT", Lexeme(text[start:cursor], source.span(start, cursor))))
            continue
        if text[cursor] == "#":
            while cursor < len(text) and text[cursor] not in "\r\n":
                cursor += 1
            continue
        if text[cursor] in "\r\n":
            cursor += 2 if text.startswith("\r\n", cursor) else 1
            kind = "NEWLINE"
        else:
            try:
                kind, cursor = syntax.recognize(text, cursor)
            except ValueError as failure:
                raise Diagnostic("lexer", str(failure), source, source.span(start, start + 1)) from failure
        tokens.append(Token(kind, Lexeme(text[start:cursor], source.span(start, cursor))))
    tokens.append(Token("EOF", Lexeme("", source.span(cursor, cursor))))
    return tokens


import unittest


class LexerTests(unittest.TestCase):
    def test_keyword_boundaries_and_longest_symbol(self):
        tokens = lex(SourceFile("x.sev", "if iffy <= 1..2 # comment\n"), Syntax())
        self.assertEqual([(t.kind, t.text) for t in tokens[:6]],
                         [("KEYWORD", "if"), ("IDENTIFIER", "iffy"), ("SYMBOL", "<="),
                          ("NUMBER", "1"), ("SYMBOL", ".."), ("NUMBER", "2")])

    def test_every_keyword_and_unicode(self):
        syntax = Syntax()
        for spelling in syntax.keywords:
            self.assertEqual(lex(SourceFile("x", spelling), syntax)[0].kind, "KEYWORD")
            self.assertEqual(lex(SourceFile("x", spelling + "_x"), syntax)[0].kind, "IDENTIFIER")
        self.assertEqual(lex(SourceFile("x", "'😀'"), syntax)[0].span.end, 3)
