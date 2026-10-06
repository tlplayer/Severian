"""Syntax-driven tokenization. Recognition does not evaluate literals."""
from dataclasses import dataclass
from py_compiler.frontend.source.source import Diagnostic, SourceFile, Span
from py_compiler.syntax.recognition import Syntax


@dataclass(frozen=True)
class Lexeme:
    text: str
    span: Span
    source: object = None


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: Lexeme
    grammar: object = None
    syntax: object = None

    @property
    def text(self):
        return self.lexeme.text

    @property
    def span(self):
        return self.lexeme.span


def lex(source, syntax):
    from py_compiler.syntax.generic.grammar import SourceWindow
    text, cursor, tokens = source.text, 0, []
    registry = syntax.registry
    while cursor < len(text):
        start = cursor
        try:
            match = registry.recognize('Y', SourceWindow(source, cursor, len(text)))
            kind, cursor = match.provider.construct(match)
        except ValueError as failure:
            raise Diagnostic('lexer', str(failure), source, source.span(start, start + 1)) from failure
        if kind != 'TRIVIA':
            tokens.append(Token(kind, Lexeme(text[start:cursor], source.span(start, cursor), source), match.provider, syntax))
    tokens.append(Token('EOF', Lexeme('', source.span(cursor, cursor), source)))
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
