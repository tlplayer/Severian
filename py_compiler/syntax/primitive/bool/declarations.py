from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.generic.owned import WordGrammar
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars, UnaryOperation


class Bool(Primitive):
    def render_constant(self, value, symbol):
        return [], [f'%value = arith.constant {int(value)} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0>'

    def grammars(self):
        from py_compiler.syntax.keywords.logical import Logical
        return (*super().grammars(), WordGrammar(self, 'true'), WordGrammar(self, 'false'),
                *scalar_grammars(self, arithmetic=False), Logical(self, 'and', 2), Logical(self, 'or', 1), UnaryOperation(self, 'not'))

    def decode(self, spelling):
        if spelling not in ('true', 'false'):
            raise ValueError('invalid Boolean literal')
        return spelling == 'true'


BOOL = Bool('bool', 'bool', 'i1', 1)
