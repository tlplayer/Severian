"""Raw pointer comparisons are available only in an unsafe context."""
from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation


class PointerComparison(ScalarOperation):
    def __init__(self, owner, spelling):
        super().__init__(owner, spelling, 3, comparison=True)

    def expand(self, cfg, left, right, span):
        self.owner.require_context(cfg)
        return super().expand(cfg, left, right, span)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        predicate = {'==': 'eq', '!=': 'ne', '<': 'ult', '<=': 'ule', '>': 'ugt', '>=': 'uge'}[self.spelling]
        left, right = map(name, operation.operands)
        return [f'{name(operation.result)} = llvm.icmp "{predicate}" {left}, {right} : !llvm.ptr']


def grammars(owner):
    return tuple(PointerComparison(owner, spelling) for spelling in ('==', '!=', '<', '<=', '>', '>='))


import unittest


class PointerTests(unittest.TestCase):
    def test_comparison_is_pointer_typed_and_unsafe(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        for spelling, predicate in (('==', 'eq'), ('!=', 'ne'), ('<', 'ult'), ('<=', 'ule'), ('>', 'ugt'), ('>=', 'uge')):
            result = compile_source('pointer.sev', f'def compare(a: pointer, b: pointer) -> bool:\n    unsafe:\n        return a {spelling} b\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertIn(f'llvm.icmp "{predicate}"', render(lower(result.program)))
        result = compile_source('invalid.sev', 'def compare(a: pointer, b: pointer) -> bool:\n    return a == b\n', Syntax())
        self.assertTrue(result.diagnostics)
