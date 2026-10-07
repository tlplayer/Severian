"""The language type object wraps a resolved declaration, never a host Python type."""
from dataclasses import dataclass
from py_compiler.syntax.generic.definition import TypeDefinition


@dataclass(frozen=True)
class Type(TypeDefinition):
    name: str = 'type'
    declaration: object = None

    @classmethod
    def of(cls, value):
        declaration = getattr(value, 'type', None)
        if declaration is None:
            raise ValueError('type(value) requires a resolved Severian value')
        return cls(declaration=declaration)

    def is_type(self, declaration):
        if self.declaration is None:
            raise ValueError('type identity requires a resolved declaration')
        return self.declaration == declaration
