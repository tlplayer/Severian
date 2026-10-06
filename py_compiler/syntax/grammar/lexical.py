"""LexicalTermic syntax providers. The lexer supplies source windows, not spelling rules."""
from py_compiler.syntax.generic.grammar import Grammar, Match


class LexicalTerm(Grammar):
    role = 'Y'

    def construct(self, match):
        return match.captures['kind'], match.end


class Whitespace(LexicalTerm):
    def recognize(self, window):
        text, start = window.source.text, window.start
        if text[start] not in ' \t':
            return None
        end = start + 1
        while end < window.end and text[end] in ' \t':
            end += 1
        kind = 'INDENT' if start == 0 or text[start - 1] in '\r\n' else 'TRIVIA'
        return Match(self, window, end, {'kind': kind})


class Newline(LexicalTerm):
    def recognize(self, window):
        text, start = window.source.text, window.start
        if text[start] not in '\r\n':
            return None
        return Match(self, window, start + (2 if text.startswith('\r\n', start) else 1), {'kind': 'NEWLINE'})


class Comment(LexicalTerm):
    def recognize(self, window):
        if window.source.text[window.start] != '#':
            return None
        end = window.start + 1
        while end < window.end and window.source.text[end] not in '\r\n':
            end += 1
        return Match(self, window, end, {'kind': 'TRIVIA'})


class Word(LexicalTerm):
    def __init__(self, keywords):
        self.keywords = keywords

    def recognize(self, window):
        text, start = window.source.text, window.start
        if not (text[start].isalpha() or text[start] == '_'):
            return None
        end = start + 1
        while end < window.end and (text[end].isalnum() or text[end] == '_'):
            end += 1
        if text[start:end] in self.keywords:
            return None
        return Match(self, window, end, {'kind': 'IDENTIFIER'})


def providers(syntax):
    from py_compiler.syntax.catalog import grammars
    rules = grammars(syntax)
    reserved = {g.spelling for g in rules if g.role == 'Y' and hasattr(g, 'spelling')}
    return (Whitespace(), Newline(), Comment(), Word(reserved), *rules)
