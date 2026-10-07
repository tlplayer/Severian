from py_compiler.syntax.primitive.contract import Primitive


class Pointer(Primitive):
    conversion_sources = frozenset(('integer', 'pointer'))

    def grammars(self):
        from py_compiler.syntax.primitive.pointer.operations import grammars
        return (*super().grammars(), *grammars(self))

    def require_context(self, context):
        if not getattr(context, 'unsafe_depth', 0):
            raise ValueError('pointer values require an unsafe block')

    def render_constant(self, value, symbol):
        if value == 0:
            return [], [f'%value = llvm.mlir.zero : {self.mlir}']
        return [], [f'%address = arith.constant {value} : i{self.bits}',
                    f'%value = llvm.inttoptr %address : i{self.bits} to {self.mlir}']

    def accept_literal(self, owner, value):
        if owner.family != 'integer':
            return super().accept_literal(owner, value)
        if not 0 <= value < (1 << self.bits):
            raise ValueError('pointer address outside target width')
        return value


def declaration(pointer_bits):
    return Pointer('pointer', 'pointer', '!llvm.ptr', pointer_bits)
