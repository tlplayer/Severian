"""The absent declaration owns the no-value contract and its literal grammar."""
from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.generic.owned import WordGrammar


class Absent(Primitive):
    @property
    def no_result(self):
        return True

    def grammars(self):
        return (WordGrammar(self, self.name),)

    def decode(self, spelling):
        if spelling != self.name:
            raise ValueError('invalid absent literal')
        return None

    def render_constant(self, value, symbol):
        return [], []


def declaration(pointer_bits):
    return Absent('absent', 'absence', '')


import unittest


class AbsentTests(unittest.TestCase):
    def test_absent_replaces_the_no_result_type(self):
        from py_compiler.syntax.recognition import Syntax
        types = Syntax().types
        self.assertNotIn('unit', types)
        self.assertTrue(types['absent'].no_result)
        self.assertEqual(types['absent'].mlir, '')
        self.assertEqual(types['None'].mlir, '!llvm.ptr')

    def test_explicit_and_inferred_absent_returns_lower_without_values(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_native, verifier_path
        import shutil
        result = compile_source('absent.sev', 'def explicit() -> absent:\n    return absent\ndef inferred():\n    explicit()\n    return\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        for body in result.program.bodies:
            self.assertEqual(body.result_type.name, 'absent')
            self.assertIsNone(body.blocks[-1].terminator.value)
        ir = render(lower(result.program))
        self.assertNotIn(' -> !llvm.ptr', ir)
        if shutil.which(verifier_path()):
            verify_native(ir)
