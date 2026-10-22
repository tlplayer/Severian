"""None retains its null-pointer representation independently of absent."""
from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.generic.owned import WordGrammar, OwnedGrammar, Capture


class NoneCondition(OwnedGrammar):
    def __init__(self, owner):
        capture = Capture('self', owner)
        super().__init__(owner, 'F.condition', ('bool', '(', capture, ')'), (capture,))

    def recognize(self, window):
        return None

    def expand(self, value, node, cfg, env):
        from py_compiler.syntax.function.calls import literal_value
        return literal_value(False, cfg.context.type('bool'), cfg, node.token.span)


class NoneValue(Primitive):
    def grammars(self):
        return (*super().grammars(), WordGrammar(self, self.name), NoneCondition(self))

    def decode(self, spelling):
        if spelling != self.name:
            raise ValueError('invalid None literal')
        return None

    def render_constant(self, value, symbol):
        return [], [f'%value = llvm.mlir.zero : {self.mlir}']


def declaration(pointer_bits):
    return NoneValue('None', 'absence', '!llvm.ptr', pointer_bits)


import unittest


class NoneConditionTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('none-condition.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result

    def test_none_owns_one_false_condition_grammar(self):
        result = self.compile('def condition(value: None) -> bool:\n    return value\n')
        constants = [constant.value for constant in result.program.constants
                     if constant.type.family == 'bool']
        self.assertEqual(constants, [False])
        self.assertEqual(len([g for g in declaration(64).grammars()
                              if g.role == 'F.condition']), 1)

    def test_none_conditions_and_negation_lower(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('''def missing() -> bool:
    minimum = None
    if minimum:
        return false
    while minimum:
        return false
    if not minimum or false:
        return not None
    return None and true
''')
        output = render(lower(result.program))
        self.assertIn('arith.xori', output)
        self.assertIn('cf.cond_br', output)

    def test_none_returning_call_is_evaluated_once(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('''def missing() -> None:
    return None
def condition() -> bool:
    return not missing()
''')
        output = render(lower(result.program))
        calls = [line for line in output.splitlines()
                 if 'func.call' in line and 'missing' in line]
        self.assertEqual(len(calls), 1, output)

    def test_none_does_not_make_absent_a_condition(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('absent-condition.sev',
                                'def condition() -> bool:\n    return not absent\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('absent has no value', str(result.diagnostics))

    def test_short_circuited_operand_still_requires_a_valid_type(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('none-invalid-comparison.sev', '''def condition(value: int) -> bool:
    minimum = None
    return not minimum or value < minimum
''', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertNotIn('Boolean condition grammar', str(result.diagnostics))
        self.assertIn('None', str(result.diagnostics))

    def test_c_none_conditions_preserve_short_circuit_effects(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''calls = 0
def missing() -> None:
    module.calls += 1
    return None

test "None is false and not None is true":
    assert(not None)
    if None:
        assert(false)
    while None:
        assert(false)

test "None returning calls execute once and respect short circuiting":
    module.calls = 0
    assert(not missing())
    assert(module.calls == 1)
    assert(not (None and missing()))
    assert(module.calls == 1)
    assert(not (None or missing()))
    assert(module.calls == 2)
    assert(not None or missing())
    assert(module.calls == 2)
'''
        with tempfile.TemporaryDirectory(prefix='sev-none-condition-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/main.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({
                'package': {'name': 'none-condition', 'version': '0.1.0'},
            }))
            result = test_package(root, memory=True, coverage=True)
            report = json.loads(Path(result['report']).read_text()) if result.get('report') else result
            failures = [{key: item.get(key) for key in ('label', 'stderr', 'error', 'exit')}
                        for item in report.get('tests', ()) if not item.get('passed')]
            self.assertTrue(result['success'], failures or result)
