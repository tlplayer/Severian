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
    conversion_sources = frozenset(('integer', 'byte'))
    def grammars(self):
        from py_compiler.syntax.primitive.numeric.grammar import CompoundAssignment
        return (*Primitive.grammars(self), ByteLiteral(self), *scalar_grammars(self),
                ByteRatio(self, '/', 5), ByteRatio(self, '//', 5),
                IntegerOperation(self, '%', 5), CompoundAssignment(self, '%'))

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


from py_compiler.syntax.primitive.int.operations import IntegerOperation


class ByteRatio(IntegerOperation):
    def expand(self, cfg, left, right, span):
        if left.type != self.owner or right.type != self.owner:
            raise ValueError('byte ratio requires two byte quantities')
        return cfg.emit('scalar', cfg.syntax.types['int'], (left, right), self, span)

    def atom(self, operation):
        from dataclasses import replace
        return replace(super().atom(operation), result=operation.result.type)


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


    def test_byte_division_produces_an_integer_ratio(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('ratio.sev', 'def ratio(a: byte, b: byte) -> int:\n    return a / b\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        operations = [op for body in result.program.bodies for block in body.blocks for op in block.operations if op.kind == 'scalar']
        self.assertEqual(operations[0].result.type.name, 'i64')
        self.assertEqual(operations[0].atom.owner, BYTE)
        self.assertIn('arith.divsi', render(lower(result.program)))
