"""Grammars carry their defining object and concrete capture contracts."""
from dataclasses import dataclass
from py_compiler.syntax.generic.grammar import Grammar, Match


@dataclass(frozen=True)
class Capture:
    name: str
    contract: object
    ownership: str = 'view'


class OwnedGrammar(Grammar):
    def __init__(self, owner, role, arrangement, captures=()):
        self.owner, self.role = owner, role
        self.arrangement, self.captures = tuple(arrangement), tuple(captures)

    def construct(self, match):
        return self


class WordGrammar(OwnedGrammar):
    def __init__(self, owner, spelling, kind='KEYWORD'):
        super().__init__(owner, 'Y', (spelling,))
        self.spelling, self.kind = spelling, kind

    def recognize(self, window):
        text, start = window.source.text, window.start
        end = start + len(self.spelling)
        if end > window.end or text[start:end] != self.spelling:
            return None
        if end < window.end and (text[end].isalnum() or text[end] == '_'):
            return None
        return Match(self, window, end, {'kind': self.kind})

    def construct(self, match):
        return match.captures['kind'], match.end


def select(owner, role, spelling):
    matches = [g for g in owner.grammars() if g.role == role and g.spelling == spelling]
    if len(matches) != 1:
        raise ValueError(f'{owner.name} requires exactly one {role} grammar for {spelling!r}; found {len(matches)}')
    return matches[0]
