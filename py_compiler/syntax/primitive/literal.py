from py_compiler.syntax.generic.owned import OwnedGrammar, WordGrammar
from py_compiler.syntax.generic.grammar import Match
from py_compiler.syntax.symbol.forms import quoted_end


class QuotedLiteral(OwnedGrammar):
    def __init__(self, owner, kind):
        super().__init__(owner, 'Y', (kind,))
        self.kind = kind

    def recognize(self, window):
        text, start = window.source.text, window.start
        if text[start] not in "\"'":
            return None
        kind = 'CHAR' if text[start] == "'" and not text.startswith("'''", start) else 'STRING'
        if kind != self.kind:
            return None
        return Match(self, window, quoted_end(text, start), {'kind': kind, 'type': self.owner})

    def construct(self, match):
        return self.kind, match.end
