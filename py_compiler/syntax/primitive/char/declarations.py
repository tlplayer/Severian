from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.primitive.literal import QuotedLiteral
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars
from py_compiler.syntax.symbol.forms import decode_quoted


class Char(Primitive):
    conversion_sources = frozenset(('integer', 'char'))
    def render_constant(self, value, symbol):
        return [], [f'%value = arith.constant {int(value)} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0>'

    def grammars(self):
        return (*super().grammars(), QuotedLiteral(self, 'CHAR'), *scalar_grammars(self, arithmetic=False))

    def decode(self, spelling):
        return ord(decode_quoted(spelling))


CHAR = Char('char', 'char', 'i32', 32)


import unittest


class CharTests(unittest.TestCase):
    def test_unicode_scalar_construction_and_order(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('char.sev', 'def ordered(value: u32) -> bool:\n    letter = char(value)\n    return letter >= \'λ\'\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('Unicode scalar', text)
        self.assertIn('arith.cmpi uge', text)
        for value in (-1, 55296, 57343, 1114112):
            self.assertTrue(compile_source('invalid.sev', f'value = char({value})\n', Syntax()).diagnostics)
