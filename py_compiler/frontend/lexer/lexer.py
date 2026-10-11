"""Syntax-driven tokenization. Recognition does not evaluate literals."""
from dataclasses import dataclass, replace
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
    replacement: str | None = None

    @property
    def text(self):
        return self.lexeme.text if self.replacement is None else self.replacement

    def with_text(self, text, *, kind=None):
        # Specialization changes parser input, not the original source lexeme.
        return replace(self, replacement=text, kind=self.kind if kind is None else kind)

    @property
    def span(self):
        return self.lexeme.span

    @property
    def form(self):
        return getattr(self.grammar, 'form', None)

    @property
    def symbol(self):
        return self.grammar.owner if self.form is not None else None


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
    def test_specialized_token_preserves_original_lexeme_and_span(self):
        source = SourceFile('generic.sev', 'T N')
        type_token, count_token = lex(source, Syntax())[:2]
        specialized = type_token.with_text('i32')
        count = count_token.with_text('2', kind='NUMBER')
        self.assertEqual((specialized.text, count.text, count.kind), ('i32', '2', 'NUMBER'))
        self.assertIs(specialized.lexeme, type_token.lexeme)
        self.assertEqual(specialized.lexeme.text, 'T')
        self.assertEqual(specialized.span, type_token.span)
        self.assertEqual((type_token.text, count_token.text, count_token.kind), ('T', 'N', 'IDENTIFIER'))

    def test_keyword_boundaries_and_longest_symbol(self):
        tokens = lex(SourceFile("x.sev", "if iffy <= 1..2 # comment\n"), Syntax())
        self.assertEqual([(t.kind, t.text) for t in tokens[:6]],
                         [("KEYWORD", "if"), ("IDENTIFIER", "iffy"), ("LESS_EQUAL", "<="),
                          ("NUMBER", "1"), ("RANGE", ".."), ("NUMBER", "2")])

    def test_every_keyword_and_unicode(self):
        syntax = Syntax()
        # Prelude type owners supply these token kinds instead of the fallback
        # Keyword declaration. Their names remain usable in type expressions.
        type_words = {'dynamic', 'union'}
        for spelling in syntax.keywords:
            with self.subTest(spelling=spelling):
                token = lex(SourceFile("x", spelling), syntax)[0]
                self.assertEqual(token.kind, 'IDENTIFIER' if spelling in type_words else 'KEYWORD')
                self.assertEqual(token.grammar.spelling, spelling)
                self.assertEqual(lex(SourceFile("x", spelling + "_x"), syntax)[0].kind, "IDENTIFIER")
        self.assertEqual(lex(SourceFile("x", "'😀'"), syntax)[0].span.end, 3)
