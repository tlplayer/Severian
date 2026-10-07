"""Byte quantities own literal grammar, scalar operations and i64 lowering."""
from py_compiler.syntax.primitive.numeric.types import Integer
from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars

from py_compiler.syntax.generic.owned import OwnedGrammar
from py_compiler.syntax.generic.grammar import Match
from py_compiler.syntax.symbol.forms import NUMBER


class ByteLiteral(OwnedGrammar):
    def __init__(self, owner):
        super().__init__(owner, 'Y', ('byte-literal',))

    def recognize(self, window):
        found = NUMBER.match(window.source.text, window.start, window.end)
        if found is None:
            return None
        spelling = found.group()
        if spelling.lower().startswith(('0x', '0o', '0b')) or not spelling.endswith('B'):
            return None
        end = found.end()
        if end < window.end and (window.source.text[end].isalnum() or window.source.text[end] == '_'):
            raise ValueError('invalid byte literal suffix')
        return Match(self, window, end, {'kind': 'NUMBER', 'type': self.owner})

    def construct(self, match):
        return 'NUMBER', match.end


class Byte(Integer):
    def grammars(self):
        return (*Primitive.grammars(self), ByteLiteral(self), *scalar_grammars(self))

    def decode(self, spelling):
        text = spelling[:-1].replace('_', '')
        if any(c in text.lower() for c in '.e'):
            raise ValueError('byte quantities require an integral amount')
        value = int(text, 10)
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError('byte quantity exceeds its signed 64-bit representation')
        return value

    def accept_literal(self, owner, value):
        return Primitive.accept_literal(self, owner, value)


BYTE = Byte("byte", "byte", "i64", 64, True)


import unittest


class ByteTests(unittest.TestCase):
    def test_byte_owner_reaches_hir_and_mlir(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('byte.sev', 'size = 1B\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        constant = result.program.constants[0]
        self.assertEqual((constant.type, constant.value), (BYTE, 1))
        self.assertIn('arith.constant 1 : i64', render(lower(result.program)))

    def test_byte_is_distinct_from_integer_and_rejects_fraction(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for text in ('size: int = 1B\n', 'size = 1.5B\n'):
            result = compile_source('byte.sev', text, Syntax())
            self.assertTrue(result.diagnostics)
