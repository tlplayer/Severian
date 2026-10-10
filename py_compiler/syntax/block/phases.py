"""With contract phases, applied by the owning block's semantic provider."""
from py_compiler.syntax.complex.object import Assert


class ContractPhases:
    def __init__(self, entry, cfg, env):
        self.entry, self.cfg, self.env = entry, cfg, env
        self.checking = False

    def check(self, expressions, phase):
        if self.checking:
            return
        self.checking = True
        try:
            for term in expressions:
                value = self.cfg.expr(term, self.env, self.cfg.syntax.types['bool'])
                action = next((action for condition, action in self.entry.failure_actions
                               if condition is term), None)
                if action is not None:
                    self.failure(value, action)
                    continue
                line, column = self.cfg.source.position(term.token.span.start)
                message = f'{phase} contract failed in {self.entry.name} at {self.cfg.source.path}:{line}:{column}'
                self.cfg.effect('assert', (value,), Assert(message), term.token.span)
        finally:
            self.checking = False

    def failure(self, condition, action):
        from py_compiler.mir.cfg.cfg import Edge, Terminator
        from py_compiler.mir.ownership.flow import FlowState
        cfg = self.cfg
        failed, following = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator(
            'conditional', (Edge(following.identity), Edge(failed.identity)), condition)
        initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
        cfg.current = failed
        callee = action.operands[0]
        owner = cfg.syntax.types.get(callee.token.text) if callee.kind == 'name' else None
        if getattr(owner, 'family', None) == 'error':
            owner.raise_failure(action.operands[1:], cfg, action.token.span)
        else:
            result = cfg.expr(action, self.env)
            if result is not None and callable(getattr(result.type, 'release', None)):
                raise ValueError('contract failure action cannot discard owning memory')
        if cfg.current.terminator is None:
            cfg.current.terminator = Terminator('jump', (Edge(following.identity),))
            cfg.flow = FlowState.join([initial_flow, cfg.flow])
        else:
            cfg.flow = initial_flow
        cfg.values = initial_values
        cfg.current = following

    def enter(self):
        self.check(self.entry.guards, 'prefix')
        self.check(self.entry.obligations, 'fix')

    def during(self):
        self.check(self.entry.obligations, 'fix')

    def leave(self):
        self.during()
        self.check(self.entry.suffix, 'suffix')


import unittest


class ContractPhaseTests(unittest.TestCase):
    def compile(self, source, path='contract-phases.sev'):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source(path, source, Syntax(include_tests=True))
        self.assertFalse(result.diagnostics, '\n'.join(map(str, result.diagnostics)))
        return result.program

    def example(self, relative):
        from pathlib import Path
        path = Path(__file__).resolve().parents[2] / 'examples/test/src' / relative
        return self.compile(path.read_text(), str(path))

    def test_example_callbacks_use_only_the_existing_phase_points(self):
        program = self.example('blocks/phases.sev')
        expected = {'default_contract': 1, 'prefix_contract': 1,
                    'fix_contract': 6, 'fix_entry_exit': 2,
                    'suffix_contract': 1, 'deferred_contract': 1,
                    'transient_contract_change': 0, 'lasting_contract_change': 0}
        for name, count in expected.items():
            with self.subTest(function=name):
                body = next(body for body in program.bodies if body.declaration == name)
                calls = [(block, op) for block in body.blocks for op in block.operations
                         if op.kind == 'call' and op.payload.symbol.startswith('__sev_fn_contract_failed[')]
                self.assertEqual(len(calls), count)
                for block, _ in calls:
                    # A successful predicate bypasses the callback entirely.
                    predecessors = [other.terminator for other in body.blocks
                                    if any(edge.target == block.identity for edge in other.terminator.edges)]
                    self.assertEqual(len(predecessors), 1)
                    self.assertEqual(predecessors[0].kind, 'conditional')
                    self.assertEqual(predecessors[0].edges[1].target, block.identity)
        self.assertGreaterEqual(sum(body.test is not None for body in program.bodies), 7)

    def test_example_errors_lower_to_terminating_failure_edges(self):
        program = self.example('complex/error.sev')
        body = next(body for body in program.bodies if body.declaration == 'contract_error_phases')
        failures = [block for block in body.blocks if block.terminator.kind == 'panic']
        self.assertEqual(len(failures), 6)
        for block in failures:
            self.assertIn('error.sev:', block.terminator.message)
            predecessors = [other.terminator for other in body.blocks
                            if any(edge.target == block.identity for edge in other.terminator.edges)]
            self.assertEqual(len(predecessors), 1)
            self.assertEqual(predecessors[0].kind, 'conditional')
            self.assertEqual(predecessors[0].edges[1].target, block.identity)
        from py_compiler.mlir.src.cfg import render_body
        rendered = render_body(body)
        self.assertIn('outside bounds during execution', rendered)
        self.assertEqual(rendered.count('llvm.unreachable'), 6)

    def test_callbacks_can_take_the_current_value(self):
        program = self.compile('def failed(value: int):\n    assert(value < 0)\n'
                               'def work(x: int) -> int with {fix x >= 0 -> failed(x)}:\n'
                               '    x = -1\n    x = 1\n    return x\n')
        body = next(body for body in program.bodies if body.declaration == 'work')
        calls = [op for block in body.blocks for op in block.operations if op.kind == 'call']
        self.assertEqual(len(calls), 4)
        constants = {op.result.identity: op.payload.value for block in body.blocks
                     for op in block.operations if op.kind == 'constant'}
        self.assertEqual(constants[calls[1].operands[0].identity], -1)
