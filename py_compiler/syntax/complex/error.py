"""Error owns its nominal type declaration and existing pointer representation."""
from py_compiler.syntax.complex.object import ObjectType, NamedDefinition


class ErrorType(ObjectType):
    def __init__(self):
        super().__init__('Error', 'error', '!llvm.ptr', declaration=NamedDefinition('Error'))
