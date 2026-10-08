"""Ordered dictionary storage policy; native nodes use minimum degree eight."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BTree:
    name: str = 'btree'
    family: str = 'container'
    dictionary_kind: int = 0

    def grammars(self):
        from py_compiler.syntax.generic.owned import WordGrammar
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)
