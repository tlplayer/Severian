"""generic owns its type grammar, ownership and target representation contract."""
from py_compiler.syntax.generic.definition import TypeDefinition


class GenericType(TypeDefinition):
    def __init__(self):
        super().__init__('generic')
