from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.generic.owned import WordGrammar
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars, UnaryOperation


class Bool(Primitive):
    conversion_sources = frozenset(('integer', 'float', 'bool'))
    def render_constant(self, value, symbol):
        return [], [f'%value = arith.constant {int(value)} : {self.mlir}']

    def module_storage(self, symbol):
        return f'memref.global "private" @{symbol} : memref<{self.mlir}> = dense<0>'

    def grammars(self):
        from py_compiler.syntax.operator.logical import Logical
        from py_compiler.syntax.primitive.int.operations import IntegerOperation
        from py_compiler.syntax.primitive.numeric.grammar import CompoundAssignment
        return (*super().grammars(), WordGrammar(self, 'true'), WordGrammar(self, 'false'),
                *(IntegerOperation(self, op, priority) for op, priority in (('|', 3.1), ('^', 3.2), ('&', 3.3))),
                *(CompoundAssignment(self, op) for op in ('&', '|', '^')),
                *scalar_grammars(self, arithmetic=False), Logical(self, 'and', 2), Logical(self, 'or', 1), UnaryOperation(self, 'not'))

    def decode(self, spelling):
        if spelling not in ('true', 'false'):
            raise ValueError('invalid Boolean literal')
        return spelling == 'true'


BOOL = Bool('bool', 'bool', 'i1', 1)


import unittest


class BoolTests(unittest.TestCase):
    def test_bitwise_assignments_are_owned_bool_operations(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('bool.sev', 'def combine(a: bool, b: bool) -> bool:\n    a &= b\n    a |= b\n    a ^= b\n    return not a\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        for opcode in ('arith.andi', 'arith.ori', 'arith.xori'):
            self.assertIn(opcode, text)
        operations = [op for body in result.program.bodies for block in body.blocks for op in block.operations if op.kind == 'scalar']
        self.assertTrue(all(op.atom.owner == BOOL for op in operations))
