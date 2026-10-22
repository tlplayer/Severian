from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.generic.owned import WordGrammar
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars, UnaryOperation


class Bool(Primitive):
    conversion_sources = frozenset(('integer', 'float', 'bool'))

    def condition(self, node, cfg, env):
        """Conditions delegate to the value's declared Boolean grammar."""
        if node.kind in ('list', 'map', 'set'):
            from py_compiler.syntax.complex.collection import lower_literal
            return lower_literal(node, cfg, env, self)
        value = cfg.expr(node, env)
        if value is None:
            raise ValueError('absent has no value for a Boolean condition')
        if value.type == self:
            return value
        from py_compiler.syntax.block.grammar import boolean_grammar
        return boolean_grammar(value, node, cfg, env)

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


class CollectionConditionTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('collection-condition.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result, render(lower(result.program))

    def test_record_conditions_select_the_source_owned_grammar(self):
        result, text = self.compile('''trait C:
    grammar truth["bool", "(", self, ")"]() -> bool
class Items: C
    count: int
    grammar truth["bool", "(", self, ")"]() -> bool:
        return self.count != 0
def ready(values: Items) -> bool:
    if not values:
        return false
    while values:
        return true
    return false
def combine(values: Items, other: Items) -> bool:
    return values and other or not values
def choose(values: Items) -> int:
    return 1 if values else 0
''')
        self.assertIn('arith.cmpi ne', text)
        self.assertIn('arith.xori', text)

    def test_empty_literal_and_primitive_collections(self):
        self.compile('''def empty() -> bool:
    if []:
        return true
    return not []
def sized(values: tensor[int], text: string, pair: tuple[int,2]) -> bool:
    return values and text and pair
''')

    def test_condition_evaluates_a_call_once(self):
        _, text = self.compile('''class Items:
    count: int
    grammar truth["bool", "(", self, ")"]() -> bool:
        return self.count != 0
def make() -> Items:
    return Items(1)
def check() -> bool:
    return not make()
''')
        calls = [line for line in text.splitlines() if 'func.call' in line and 'make' in line]
        self.assertEqual(len(calls), 1, text)

    def test_len_method_alone_does_not_grant_boolean_grammar(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('no-grammar.sev', '''class Pretender:
    count: int
    def len() -> int:
        return count
def check(value: Pretender) -> bool:
    return not value
''', Syntax())
        self.assertIn('Boolean condition grammar', str(result.diagnostics))
