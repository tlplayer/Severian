"""Shared discovery mechanics; concrete type owners declare their own implementation."""
from dataclasses import dataclass
from py_compiler.syntax.generic.owned import WordGrammar


@dataclass(frozen=True)
class TypeDefinition:
    name: str
    family: str = 'unresolved'

    def grammars(self):
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)

    @property
    def mlir(self):
        raise ValueError(f'{self.name} has no declared type/lowering implementation')

    @property
    def binding_ownership(self):
        raise ValueError(f'{self.name} has no declared ownership implementation')

    def lower(self, *args, **kwargs):
        raise ValueError(f'{self.name} has no declared type/lowering implementation')

    def accept_literal(self, owner, value):
        raise ValueError(f'{self.name} has no declared literal conversion from {owner.name}')
