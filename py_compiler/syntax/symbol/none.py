"""None retains its null-pointer representation independently of absent."""
from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.generic.owned import WordGrammar


class NoneValue(Primitive):
    def grammars(self):
        return (*super().grammars(), WordGrammar(self, self.name))

    def decode(self, spelling):
        if spelling != self.name:
            raise ValueError('invalid None literal')
        return None

    def render_constant(self, value, symbol):
        return [], [f'%value = llvm.mlir.zero : {self.mlir}']


def declaration(pointer_bits):
    return NoneValue('None', 'absence', '!llvm.ptr', pointer_bits)
