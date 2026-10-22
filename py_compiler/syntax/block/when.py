"""Runtime test scenarios use ordinary conditional control flow."""
from importlib import import_module


class When(import_module('py_compiler.syntax.block.if').Conditional):
    spelling = 'when'
    required_scope = 'test'

    def lower(self, node, cfg, env, nodes, index):
        scenarios = getattr(cfg, 'test_scenarios', ())
        cfg.test_scenarios = (*scenarios, ' '.join(token.text for token in node.header))
        try:
            return super().lower(node, cfg, env, nodes, index)
        finally:
            cfg.test_scenarios = scenarios


import unittest


class WhenTests(unittest.TestCase):
    def test_runtime_condition_and_failure_context(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'test "scenario":\n    age = 12\n    when(age < 18):\n        expect(age == 12)\n'
        result = compile_source('when.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.test)
        assertions = [op.payload for block in body.blocks for op in block.operations if op.kind == 'assert']
        self.assertTrue(any('when ( age < 18 )' in repr(item) for item in assertions))
        self.assertTrue(any(block.terminator.kind == 'conditional' for block in body.blocks))
