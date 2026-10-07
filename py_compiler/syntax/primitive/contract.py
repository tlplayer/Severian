from dataclasses import dataclass


@dataclass(frozen=True)
class Primitive:
    name: str
    family: str
    mlir: str
    bits: int = 0
    signed: bool = False
    conversion_sources = frozenset()

    @property
    def no_result(self):
        return False

    @property
    def binding_ownership(self):
        return "copy"

    def grammars(self):
        if self.conversion_sources:
            from py_compiler.syntax.primitive.numeric.conversion import ScalarConstructor
            return (ScalarConstructor(self),)
        return (Constructor(self),)

    def render_constant(self, value, symbol):
        raise ValueError(f'{self.name} does not supply constant lowering')

    def module_storage(self, symbol):
        raise ValueError(f'{self.name} does not supply module storage')

    def accept_literal(self, owner, value):
        if owner != self:
            raise ValueError(f'{owner.name} literal does not satisfy {self.name}')
        return value

    def require_context(self, context):
        pass


from py_compiler.syntax.generic.constructor import Constructor
