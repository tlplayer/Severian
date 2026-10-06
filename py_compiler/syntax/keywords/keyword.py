"""Keyword declarations shared by recognition and tooling, not lexer conditionals."""

KEYWORDS = frozenset("""
and as async await borrow break case catch class continue copy def drop dynamic
elif else enum false for from module grammar if import in is local match mirror
move not operator or unimplemented return self static test throw trait true try union view
while with yield None absent extend switch unsafe atomic
""".split())


from dataclasses import dataclass
from py_compiler.syntax.generic.owned import WordGrammar


@dataclass(frozen=True)
class Keyword:
    name: str

    def grammars(self):
        return (WordGrammar(self, self.name),)
