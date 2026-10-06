from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.primitive.literal import QuotedLiteral
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars
from py_compiler.syntax.symbol.forms import decode_quoted


class Char(Primitive):
    def render_constant(self, value, symbol):
        return [], [f'%value = arith.constant {int(value)} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0>'

    def grammars(self):
        return (*super().grammars(), QuotedLiteral(self, 'CHAR'), *scalar_grammars(self, arithmetic=False))

    def decode(self, spelling):
        return ord(decode_quoted(spelling))


CHAR = Char('char', 'char', 'i32', 32)
