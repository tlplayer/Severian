"""any owns its type grammar, ownership and target representation contract."""
from py_compiler.syntax.generic.definition import TypeDefinition


class AnyType(TypeDefinition):
    def __init__(self):
        super().__init__('any')
