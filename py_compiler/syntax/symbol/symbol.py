from dataclasses import dataclass
from py_compiler.syntax.generic.owned import OwnedGrammar
from py_compiler.syntax.generic.grammar import Match


@dataclass(frozen=True)
class Symbol:
    name: str
    longer: tuple = ()

    def grammars(self):
        return (SymbolGrammar(self),)


class SymbolGrammar(OwnedGrammar):
    def __init__(self, owner):
        super().__init__(owner, 'Y', (owner.name,))

    def recognize(self, window):
        text, start = window.source.text, window.start
        end = start + len(self.owner.name)
        if end > window.end or text[start:end] != self.owner.name:
            return None
        if any(start + len(s) <= window.end and text.startswith(s, start) for s in self.owner.longer):
            return None
        return Match(self, window, end, {'symbol': self.owner})

    def construct(self, match):
        return 'SYMBOL', match.end
