"""While loops carry typed bindings on explicit CFG backedges."""
from dataclasses import dataclass, field, replace
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.hir.thir.src.model import Edge, Terminator
from py_compiler.mir.ownership.flow import FlowState
from py_compiler.syntax.keywords.keyword import Keyword


@dataclass(frozen=True)
class LoopClauses:
    initializers: tuple
    suffix: tuple
    fix: tuple = ()
    bindings: tuple = ()


def split_loop_initializer(tokens, spelling, *, clauses=False):
    """A loop owns the meaning of its optional, once-only with initializer."""
    depth = 0
    for index, token in enumerate(tokens):
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
        if depth == 0 and token.text == 'with':
            initializer = tuple(tokens[index + 1:])
            if clauses:
                from importlib import import_module
                parts = import_module('py_compiler.syntax.block.with').clause_groups(initializer, tuple_bindings=True)
                initializers, suffix, obligations, bindings = [], [], [], []
                for part in parts:
                    if len(part) > 1 and part[0].kind == 'IDENTIFIER' and part[1].text == ',':
                        bindings.append(part)
                        continue
                    if part[0].text == 'prefix':
                        part = part[1:]
                        if not part:
                            raise ValueError('prefix requires a loop clause')
                    elif part[0].text == 'fix':
                        if len(part) < 2:
                            raise ValueError('fix requires a condition')
                        obligations.append(part[1:])
                        continue
                    if part[0].text in ('suffix', 'defer'):
                        if len(part) < 2 or part[1].text in ('return', 'break', 'continue'):
                            raise ValueError('loop suffix requires a non-transferring sentence')
                        suffix.append(part[1:])
                    else:
                        target = tuple(t.text for t in part[:initializer_assignment(part)]) if is_initializer(part) else ()
                        if target and any(is_initializer(previous) and
                                          tuple(t.text for t in previous[:initializer_assignment(previous)]) == target
                                          for previous in initializers):
                            raise ValueError('duplicate loop-local initializer')
                        initializers.append(part)
                return tuple(tokens[:index]), LoopClauses(tuple(initializers), tuple(suffix), tuple(obligations), tuple(bindings))
            validate_initializer(initializer, spelling)
            return tuple(tokens[:index]), initializer
    return tuple(tokens), ()


def validate_initializer(initializer, spelling):
    position = initializer_assignment(initializer)
    if position is None:
        raise ValueError(f'{spelling} with requires a loop-local initializer')
    expression(initializer[position + 1:])


def initializer_assignment(tokens):
    if len(tokens) >= 3 and tokens[0].kind == 'IDENTIFIER' and tokens[1].text in ('=', ':='):
        return 1
    if (len(tokens) >= 5 and tokens[0].text in ('local', 'module', 'self')
            and tokens[1].text == '.' and tokens[2].kind == 'IDENTIFIER'
            and tokens[3].text in ('=', ':=')):
        return 3
    return None


def is_initializer(tokens):
    return initializer_assignment(tokens) is not None


def emit_clause(node, cfg, env, phase):
    from py_compiler.syntax.catalog import assignment_spellings
    depth, assignment = 0, False
    assignments = assignment_spellings(cfg.syntax)
    for token in node.tokens:
        if depth == 0 and token.text in assignments:
            assignment = True
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if assignment:
        cfg.scope((node,), dict(env))
        return
    from py_compiler.syntax.block.phases import ContractPhases
    from py_compiler.syntax.function.contracts import Callable, add_clause
    guards, obligations, suffix, actions = [], [], [], []
    add_clause(node.tokens, guards, obligations, {}, suffix, actions)
    entry = Callable(node, 'while', (), None, tuple(guards), failure_actions=tuple(actions))
    checker = ContractPhases(entry, cfg, env)
    term = guards[0]
    # An action call may return absent; predicates must produce bool.
    if term.kind == 'call' and not actions:
        value = cfg.expr(term, env)
        if value is None:
            return
        if value.type != cfg.syntax.types['bool']:
            raise ValueError('loop clause requires a condition or a no-result action')
        from py_compiler.syntax.complex.object import Assert
        cfg.effect('assert', (value,), Assert(f'{phase} contract failed in while'), node.span)
    else:
        checker.check(entry.guards, phase)


def initialize_loop(node, initializer, cfg, env):
    from py_compiler.syntax.grammar.assignment import Assignment
    loop_env, local_names = dict(env), set()
    initializers = initializer.initializers if isinstance(initializer, LoopClauses) else ((initializer,) if initializer else ())
    for index, initializer in enumerate(initializers):
        init = replace(node, identity=node.identity + ':with:' + str(index), tokens=initializer,
                       children=[], provider=None)
        if not is_initializer(initializer):
            emit_clause(init, cfg, loop_env, 'prefix')
            continue
        # A prefix assignment updates an existing binding, just like the body.
        # Only an absent name introduces a new binding.
        position = initializer_assignment(initializer)
        existing = cfg.lookup(expression(initializer[:position]), loop_env)
        name = initializer[position - 1].text
        if existing is not None and initializer[position].text == ':=':
            raise ValueError(f'binding {name!r} already exists in this scope')
        surrounding = position == 3 and initializer[0].text == 'local'
        Assignment().lower(init, cfg, loop_env, cfg.local_names if surrounding else local_names,
                           declare_local=existing is None)
        if surrounding:
            env[name] = loop_env[name]
    return loop_env, local_names


@dataclass
class Loop:
    header: object
    following: object
    carried: tuple
    initial_flow: object
    live: set
    scope_depth: int
    exits: list = field(default_factory=list)
    suffix: tuple = ()
    env: object = None
    contract: object = None
    controls: tuple = ()
    builder: object = None
    checking: bool = False
    leaving: bool = False

    def ready(self, term):
        if term.kind == 'name':
            return self.builder.lookup(term, self.env) is not None
        operands = term.operands
        if term.kind == 'call' and operands[0].kind == 'name':
            operands = operands[1:]
        return all(self.ready(child) for child in operands if hasattr(child, 'operands'))

    def during(self):
        cfg = self.builder
        if self.contract:
            self.contract.during()
        if self.checking or self.leaving:
            return
        self.checking = True
        try:
            for term, action in self.controls:
                if not self.ready(term):
                    break
                condition = cfg.expr(term, self.env, cfg.syntax.types['bool'])
                taken, following = cfg.block(), cfg.block()
                cfg.current.terminator = Terminator('conditional',
                    (Edge(following.identity), Edge(taken.identity)), condition)
                values, flow = dict(cfg.values), cfg.flow.fork()
                cfg.current = taken
                # A guard belongs to its declaring loop, including when checked
                # inside nested blocks. Exit any entered inner loops first.
                for inner in reversed(cfg.loops):
                    if inner is self:
                        break
                    leave = getattr(inner, 'leave', None)
                    if leave:
                        leave(cfg)
                self.jump(cfg, term.token.span, exit_loop=action == 'break')
                cfg.current, cfg.values, cfg.flow = following, values, flow
        finally:
            self.checking = False

    def leave(self, cfg):
        previous = self.leaving
        self.leaving = True
        try:
            if self.contract:
                self.contract.during()
            for node in self.suffix:
                emit_clause(node, cfg, self.env, 'suffix')
                if self.contract:
                    self.contract.during()
        finally:
            self.leaving = previous

    def jump(self, cfg, span, *, exit_loop=False):
        cfg.run_defers(self.scope_depth)
        locals_ = [binding for scope in cfg.active_scopes[self.scope_depth:] for binding in scope.values()
                   if binding.identity not in self.live]
        cfg.cleanup_memory(locals_, span)
        cfg.flow.end_scope(self.live)
        self.leave(cfg)
        # The condition can exit before the first iteration. Ownership on every
        # continuing edge must therefore agree with the entry state.
        if not exit_loop and cfg.flow.moved & self.live != self.initial_flow.moved & self.live:
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
            self.exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
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
        condition, initializer = split_loop_initializer(super().parse_header(items), self.spelling, clauses=True)
        expression(condition)
        return condition, initializer

    def lower(self, node, cfg, env, nodes, index):
        condition, initializer = node.header
        if not initializer:
            return self.lower_loop(replace(node, header=condition), cfg, env)
        loop_env, local_names = initialize_loop(node, initializer, cfg, env)
        suffix = tuple(replace(node, identity=node.identity + ':suffix:' + str(index),
                               tokens=tokens, children=[], provider=None)
                       for index, tokens in enumerate(initializer.suffix)) if isinstance(initializer, LoopClauses) else ()
        cfg.active_scopes.append(loop_env)
        try:
            result = self.lower_loop(replace(node, header=condition), cfg, loop_env, suffix,
                                     initializer.fix if isinstance(initializer, LoopClauses) else (),
                                     initializer.bindings if isinstance(initializer, LoopClauses) else ())
            if cfg.current.terminator is None:
                cfg.cleanup_memory((loop_env[name] for name in local_names), node.span)
        finally:
            cfg.active_scopes.pop()
        cfg.flow.end_scope(cfg.live_binding_ids(env) |
                           {b.identity for scope in cfg.active_scopes for b in scope.values()})
        return result

    def lower_loop(self, node, cfg, env, suffix=(), obligations=(), bindings=()):
        from py_compiler.syntax.block.phases import ContractPhases, NestedPhases
        from py_compiler.syntax.function.contracts import Callable, add_clause
        conditions, actions, controls = [], [], []
        for tokens in obligations:
            if len(tokens) >= 3 and tokens[-2].text == '->' and tokens[-1].text in ('continue', 'break'):
                controls.append((expression(tokens[:-2]), tokens[-1].text))
            else:
                add_clause(tokens, conditions, [], {}, [], actions)
        entry = Callable(node, 'while', (), None, (), obligations=tuple(conditions), failure_actions=tuple(actions))
        contract = ContractPhases(entry, cfg, env)
        contract.enter()
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
        condition = cfg.context.type('bool').condition(expression(node.header), cfg, env)
        condition_flow = cfg.flow.fork()
        header_values = dict(cfg.values)
        body, following = cfg.block(), cfg.block(tuple(binding.type for binding in carried))
        exhausted = cfg.block()
        infinite = len(node.header) == 1 and node.header[0].text == 'true'
        cfg.current.terminator = (Terminator('jump', (Edge(body.identity),)) if infinite else
            Terminator('conditional', (Edge(body.identity), Edge(exhausted.identity)), condition))
        live = cfg.live_binding_ids(env) | {b.identity for scope in cfg.active_scopes for b in scope.values()}
        body_env = dict(env)
        loop = Loop(header, following, carried, initial_flow, live, len(cfg.active_scopes),
                    suffix=suffix, env=body_env, contract=contract, controls=tuple(controls), builder=cfg)
        cfg.current = body
        cfg.loops.append(loop)
        previous_phases = cfg.phases
        cfg.phases = NestedPhases(previous_phases, loop)
        try:
            loop.during()
            inputs = tuple(replace(node, identity=node.identity + ':input:' + str(index),
                                   tokens=tokens, children=[], provider=None)
                           for index, tokens in enumerate(bindings))
            cfg.scope((*inputs, *node.children), body_env)
            if any(not loop.ready(term) for term, _ in controls):
                raise ValueError('loop fix condition refers to an unbound name')
            if cfg.current.terminator is None:
                loop.jump(cfg, node.span)
        finally:
            cfg.phases = previous_phases
            cfg.loops.pop()
        if infinite:
            exhausted.terminator = Terminator('unreachable', span=node.span)
        exits = [*(() if infinite else ((exhausted, dict(header_values), condition_flow),)), *loop.exits]
        if not exits:
            following.terminator = Terminator('unreachable', span=node.span)
            cfg.current, cfg.values = following, header_values
            return 1
        from py_compiler.mir.ownership.flow import reconcile_owners
        reconcile_owners(cfg, exits, carried, node.span)
        retained = [(b, h, v) for b, h, v in zip(carried, header.parameters, following.parameters)
                    if not (callable(getattr(b.type, 'release', None)) and b.identity in exits[0][2].moved)]
        following.parameters = tuple(v for _, _, v in retained)
        for block, values, _ in exits:
            block.terminator = Terminator('jump', (Edge(following.identity, tuple(values[b.identity] for b, _, _ in retained)),))
        cfg.current, cfg.values = following, header_values
        cfg.flow = FlowState.join([flow for _, _, flow in exits])
        for binding, source, value in retained:
            cfg.values[binding.identity] = value
            cfg.value_sources.setdefault(value.identity, set()).update(cfg.value_sources.get(source.identity, ()))
            if source.identity in cfg.stack_values:
                cfg.stack_values.add(value.identity)
        contract.during()
        return 1


import unittest


class WhileTests(unittest.TestCase):
    def test_qualified_local_initializers_survive_the_loop(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for header in ('local.index = 0', '{local.index = 0, local.total = 7,}'):
            result = compile_source('qualified-loop-local.sev',
                                    'def work() -> int:\n'
                                    f'    while index < 3 with {header}:\n'
                                    '        index += 1\n    return index\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            for body in result.program.bodies:
                body.verify()

    def test_destructured_loop_inputs_and_failed_fix_guards(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''class Queue:
    count: int
    def len(self: view Self) -> int:
        return self.count
    grammar truth["bool", "(", self, ")"]() -> bool:
        return self.count != 0
    def pop(self: borrow Self) -> tuple[int,2]:
        if self.count <= 0:
            throw Error("pop on empty queue")
        self.count -= 1
        return (self.count, 0)
test "acquire after the condition, then check guards in order":
    queue = Queue(4)
    total = 0
    while queue with
    {
        i, j = queue.pop(),
        fix i != 2 -> continue,
        fix 1 <= i < 4 -> break,
    }:
        total += i + j
    assert(total == 4)
    assert(queue.count == 0)
test "empty input never calls pop":
    queue = Queue(0)
    while queue with {i, j = queue.pop(), fix i >= 0 -> continue,}:
        assert(false)
'''
        with tempfile.TemporaryDirectory(prefix='sev-loop-inputs-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'loop-inputs', 'version': '1'}}))
            result = test_package(root)
            detail = Path(result['report']).read_text() if 'report' in result else result
            self.assertTrue(result['success'], detail)

    def test_infinite_loop_has_no_fallthrough_result_obligation(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        source = 'def value(flag: bool) -> int:\n    while true:\n        if flag:\n            return 42\n'
        result = compile_source('infinite.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        verify_c(render(lower(result.program)))
        missing = compile_source('break.sev', source.replace('return 42', 'break'), Syntax())
        self.assertIn('not every path returns', str(missing.diagnostics))

    def test_braced_loop_clause_requires_a_final_comma(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('missing-comma.sev',
                                'def work():\n    while false with {i = 0}:\n        assert(false)\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn("with clauses must end with ','", str(result.diagnostics[0]))

    def test_fix_control_guards_compile_to_loop_edges(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        source = ('def work() -> int:\n    i = 0\n    while i < 5 with\n'
                  '    {fix i != 1 -> continue, fix i != 3 -> break, suffix i += 1,}:\n'
                  '        if i == 1:\n            throw Error("guard failed")\n    return i\n')
        result = compile_source('loop-guards.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        render(lower(result.program))

    def test_prefix_reuses_the_existing_binding(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for header in ('i = 0', '{i = 0, suffix i += 1,}'):
            result = compile_source('existing-loop-variable.sev',
                                    'def work() -> int:\n    i = 99\n'
                                    f'    while i < 3 with {header}:\n'
                                    '        i += 1\n    return i\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            body = next(body for body in result.program.bodies if body.declaration == 'work')
            self.assertEqual(len([binding for binding in body.bindings if binding.name == 'i']), 1)
            returned = [block.terminator.value for block in body.blocks if block.terminator.kind == 'finish']
            self.assertIn(returned[0], [value for block in body.blocks for value in block.parameters])

    def test_prefix_cannot_hide_or_rebind_an_existing_constant(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for declaration, initializer, diagnostic in (
            ('i := 99', 'i = 0', 'cannot rebind constant'),
            ('i = 99', 'i := 0', 'already exists'),
        ):
            result = compile_source('constant-loop-variable.sev',
                                    f'def work():\n    {declaration}\n'
                                    f'    while false with {initializer}:\n        assert(false)\n', Syntax())
            self.assertTrue(result.diagnostics)
            self.assertIn(diagnostic, str(result.diagnostics[0]))

    def test_braced_suffix_is_emitted_on_continue_break_and_return(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        for end in ('continue', 'break', 'return i'):
            source = ('def tick(value: int):\n    return\n'
                      'def work() -> int:\n    while i < 3 with\n'
                      '    {i = 0, suffix tick(i), suffix i += 1,}:\n'
                      f'        {end}\n    return 9\n')
            result = compile_source('suffix.sev', source, Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            body = next(b for b in result.program.bodies if b.declaration == 'work')
            calls = [op for block in body.blocks for op in block.operations if op.kind == 'call']
            self.assertEqual(len(calls), 1)
            render(lower(result.program))

    def test_jumps_release_locals_before_their_edges(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        for jump in ('break', 'continue'):
            with self.subTest(jump=jump):
                source = (f'def work(n: int) -> int:\n    unsafe:\n        i = 0\n        while i < n:\n            scratch = allocate[int, location=memory.heap](1)\n            i += 1\n            if i > 0:\n                nested = allocate[int, location=memory.heap](1)\n                {jump}\n        return i\n')
                result = compile_source('jumps.sev', source, Syntax())
                self.assertFalse(result.diagnostics, str(result.diagnostics))
                body = next(body for body in result.program.bodies if body.declaration == 'work')
                blocks = [block for block in body.blocks if sum(op.kind == 'deallocate' for op in block.operations) == 2]
                self.assertEqual(len(blocks), 1)
                self.assertEqual(blocks[0].terminator.kind, 'jump')
                verify_c(render(lower(result.program)))

    def test_invalid_jump_contexts_are_diagnostics(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for jump in ('break', 'continue'):
            for source, expected in (
                (f'def work():\n    {jump}\n', 'requires an enclosing while loop'),
                (f'def work():\n    while true:\n        {jump} 1\n', 'does not accept a value or label'),
                (f'def work():\n    while true:\n        def helper():\n            {jump}\n', 'requires an enclosing while loop'),
                (f'def work():\n    while true:\n        {jump}\n        value = 1\n', 'unreachable sentence'),
                *([('def work():\n    unsafe:\n        owner = allocate[int](1)\n        while true:\n            drop owner\n            continue\n', 'cannot consume an outer binding')] if jump == 'continue' else []),
            ):
                with self.subTest(jump=jump, source=source):
                    result = compile_source('invalid-jump.sev', source, Syntax())
                    self.assertTrue(result.diagnostics)
                    self.assertIn(expected, str(result.diagnostics[0]))

    def test_loop_carries_scalar_and_cleans_local_memory(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        result = compile_source('while.sev', 'def sum_to(n: int) -> int:\n    unsafe:\n        i = 0\n        total = 0\n        while i < n:\n            value = allocate[int]()\n            value[0] = i\n            total += value[0]\n            i += 1\n        return total\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        verify_c(render(lower(result.program)))


class LoopOwnershipTests(unittest.TestCase):
    def test_consumption_replacement_and_exit_paths(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for tail, accepted in (
            ('owner = allocate[int](1)\n            continue', True),
            ('break', True),
            ('continue', False),
            ('if flag:\n                owner = allocate[int](1)\n            continue', False),
        ):
            with self.subTest(tail=tail):
                source = ('def work(flag: bool):\n    unsafe:\n        owner = allocate[int](1)\n'
                          '        while flag:\n            drop owner\n'
                          f'            {tail}\n')
                result = compile_source('loop-ownership.sev', source, Syntax())
                self.assertEqual(not result.diagnostics, accepted, str(result.diagnostics))
