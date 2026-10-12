"""Typed synchronous iterator contract; direct consumers fuse into MIR CFG."""
from dataclasses import dataclass
from py_compiler.syntax.generic.definition import TypeDefinition


class IteratorType(TypeDefinition):
    def __init__(self):
        super().__init__('iterator')

    def specialize(self, element):
        if element.binding_ownership != 'copy' and element.family != 'record':
            raise ValueError('iterator elements currently require copy value types')
        return Iterator(element)


@dataclass(frozen=True)
class Iterator:
    element: object
    family: str = 'iterator'
    binding_ownership: str = 'view'

    def grammars(self):
        return ()

    @property
    def name(self):
        return 'iterator[' + self.element.name + ']'

    @property
    def mlir(self):
        raise ValueError('iterator values currently require direct for consumption')
