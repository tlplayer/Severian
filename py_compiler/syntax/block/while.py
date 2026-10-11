"""While loops carry typed bindings on explicit CFG backedges."""
from dataclasses import dataclass, field
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.mir.cfg.cfg import Edge, Terminator
from py_compiler.mir.ownership.flow import FlowState
from py_compiler.syntax.keywords.keyword import Keyword


@dataclass
class Loop:
    header: object
    following: object
    carried: tuple
    initial_flow: object
    live: set
    scope_depth: int
    exits: list = field(default_factory=list)

    def jump(self, cfg, span, *, exit_loop=False):
        locals_ = [binding for scope in cfg.active_scopes[self.scope_depth:] for binding in scope.values()
                   if binding.identity not in self.live]
        cfg.cleanup_memory(locals_, span)
        cfg.flow.end_scope(self.live)
        # The condition can exit before the first iteration. Ownership on every
        # continuing edge must therefore agree with the entry state.
        if cfg.flow.moved != self.initial_flow.moved:
            raise ValueError('loop cannot consume an outer binding without replacing it')
        if cfg.flow.views != self.initial_flow.views:
            raise ValueError('loop cannot retarget an outer view')
        if cfg.flow.loans != self.initial_flow.loans:
            raise ValueError('loop cannot change outer borrow lifetimes')
        values = tuple(cfg.values[b.identity] for b in self.carried)
        destination = self.following if exit_loop else self.header
        for value, parameter in zip(values, destination.parameters):
            if value.type != parameter.type:
                raise ValueError('loop cannot change a carried binding type')
            cfg.value_sources.setdefault(parameter.identity, set()).update(cfg.value_sources.get(value.identity, ()))
            if value.identity in cfg.stack_values:
                cfg.stack_values.add(parameter.identity)
        if exit_loop:
            self.exits.append(cfg.flow.fork())
        cfg.current.terminator = Terminator('jump', (Edge(destination.identity, values),))


class LoopJump(Keyword):
    def matches(self, node):
        return bool(node.tokens) and node.tokens[0].text == self.name

    def lower(self, node, cfg, env, local_names):
        if len(node.tokens) != 1:
            raise ValueError(f'{self.name} does not accept a value or label')
        if not cfg.loops:
            raise ValueError(f'{self.name} requires an enclosing while loop in this callable')
        if cfg.phases:
            cfg.phases.during()
        cfg.loops[-1].jump(cfg, node.span, exit_loop=self.name == 'break')


class Break(LoopJump):
    def __init__(self):
        super().__init__('break')


class Continue(LoopJump):
    def __init__(self):
        super().__init__('continue')


class While(BlockProvider):
    spelling = 'while'

    def parse_header(self, items):
        header = super().parse_header(items)
        expression(header)
        return header

    def lower(self, node, cfg, env, nodes, index):
        carried = tuple(binding for binding in env.values()
                        if binding.identity not in cfg.storage and binding.identity not in cfg.flow.moved)
        initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
        header = cfg.block(tuple(binding.type for binding in carried))
        cfg.current.terminator = Terminator('jump', (Edge(header.identity, tuple(cfg.values[b.identity] for b in carried)),))
        cfg.current = header
        for binding, value in zip(carried, header.parameters):
            original = initial_values[binding.identity]
            cfg.values[binding.identity] = value
            cfg.value_sources[value.identity] = set(cfg.value_sources.get(original.identity, ()))
            if original.identity in cfg.stack_values:
                cfg.stack_values.add(value.identity)
        header_values = dict(cfg.values)
        condition = cfg.expr(expression(node.header), env, cfg.context.type('bool'))
        body, following = cfg.block(), cfg.block(tuple(binding.type for binding in carried))
        cfg.current.terminator = Terminator('conditional',
                                           (Edge(body.identity), Edge(following.identity, header.parameters)), condition)
        live = cfg.live_binding_ids(env) | {b.identity for scope in cfg.active_scopes for b in scope.values()}
        loop = Loop(header, following, carried, initial_flow, live, len(cfg.active_scopes))
        cfg.current = body
        cfg.loops.append(loop)
        try:
            cfg.scope(node.children, dict(env))
            if cfg.current.terminator is None:
                loop.jump(cfg, node.span)
        finally:
            cfg.loops.pop()
        cfg.current, cfg.values = following, header_values
        cfg.flow = FlowState.join([initial_flow, *loop.exits])
        for binding, source, value in zip(carried, header.parameters, following.parameters):
            cfg.values[binding.identity] = value
            cfg.value_sources.setdefault(value.identity, set()).update(cfg.value_sources.get(source.identity, ()))
            if source.identity in cfg.stack_values:
                cfg.stack_values.add(value.identity)
        return 1


import unittest


class WhileTests(unittest.TestCase):
    def test_jumps_release_locals_before_their_edges(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_native
        for jump in ('break', 'continue'):
            with self.subTest(jump=jump):
                source = ('def work(n: int) -> int:\n    i = 0\n    while i < n:\n'
                          '        scratch = allocate[int, location=memory.heap](1)\n'
                          '        i += 1\n        if i > 0:\n'
                          '            nested = allocate[int, location=memory.heap](1)\n'
                          f'            {jump}\n    return i\n')
                result = compile_source('jumps.sev', source, Syntax())
                self.assertFalse(result.diagnostics, str(result.diagnostics))
                body = next(body for body in result.program.bodies if body.declaration == 'work')
                blocks = [block for block in body.blocks if sum(op.kind == 'deallocate' for op in block.operations) == 2]
                self.assertEqual(len(blocks), 1)
                self.assertEqual(blocks[0].terminator.kind, 'jump')
                verify_native(render(lower(result.program)))

    def test_invalid_jump_contexts_are_diagnostics(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for jump in ('break', 'continue'):
            for source, expected in (
                (f'def work():\n    {jump}\n', 'requires an enclosing while loop'),
                (f'def work():\n    while true:\n        {jump} 1\n', 'does not accept a value or label'),
                (f'def work():\n    while true:\n        def helper():\n            {jump}\n', 'requires an enclosing while loop'),
                (f'def work():\n    while true:\n        {jump}\n        value = 1\n', 'unreachable sentence'),
                ('def work():\n    owner = allocate[int](1)\n    while true:\n'
                 f'        drop owner\n        {jump}\n', 'cannot consume an outer binding'),
            ):
                with self.subTest(jump=jump, source=source):
                    result = compile_source('invalid-jump.sev', source, Syntax())
                    self.assertTrue(result.diagnostics)
                    self.assertIn(expected, str(result.diagnostics[0]))

    def test_loop_carries_scalar_and_cleans_local_memory(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_native
        result = compile_source('while.sev', 'def sum_to(n: int) -> int:\n    i = 0\n    total = 0\n    while i < n:\n        value = allocate[int]()\n        value[0] = i\n        total += value[0]\n        i += 1\n    return total\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        verify_native(render(lower(result.program)))
