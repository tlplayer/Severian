from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.mir.cfg.cfg import Edge, Terminator
from py_compiler.mir.ownership.flow import FlowState


class Conditional(BlockProvider):
    spelling = "if"
    has_condition = True
    attachment = "condition-chain"

    def parse_header(self, items):
        header = super().parse_header(items)
        expression(header)
        return header

    def attach(self, previous, parent):
        super().attach(previous, parent)
        if self.continuation and not previous.provider.has_condition:
            raise ValueError(f"{self.spelling} cannot follow a terminal branch")

    def lower(self, node, cfg, env, nodes, index):
        chain = [node]
        while index + len(chain) < len(nodes):
            following = nodes[index + len(chain)]
            if not getattr(following.provider, "continuation", False):
                break
            chain.append(following)
        self.lower_chain(cfg, chain, env)
        return len(chain)

    def lower_chain(self, cfg, chain, env):
        outer = tuple(b for b in env.values() if b.identity not in cfg.storage)
        initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
        exits = []
        for node in chain:
            cfg.values, cfg.flow = dict(initial_values), initial_flow.fork()
            validate = getattr(node.provider, 'validate_captures', None)
            if validate:
                validate(cfg, env)
            if node.provider.has_condition:
                condition = cfg.expr(expression(node.header), env, cfg.syntax.types["bool"])
                if condition.type.name != "bool":
                    raise ValueError("if/elif condition must satisfy bool")
                body, following = cfg.block(), cfg.block()
                cfg.current.terminator = Terminator("conditional", (Edge(body.identity), Edge(following.identity)), condition)
                cfg.current = body
            cfg.scope(node.children, dict(env))
            cfg.flow.end_scope(cfg.live_binding_ids(env) | {b.identity for scope in cfg.active_scopes for b in scope.values()})
            if cfg.current.terminator is None:
                exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
            if node.provider.has_condition:
                cfg.current = following
        if chain[-1].provider.has_condition:
            exits.append((cfg.current, initial_values, initial_flow))
        if not exits:
            return
        from py_compiler.mir.ownership.flow import reconcile_owners
        reconcile_owners(cfg, exits, outer, chain[-1].span)
        outer = tuple(b for b in outer if not (callable(getattr(b.type, 'release', None)) and b.identity in exits[0][2].moved))
        join = cfg.block(tuple(b.type for b in outer))
        for block, values, _ in exits:
            block.terminator = Terminator("jump", (Edge(join.identity, tuple(values[b.identity] for b in outer)),))
        for binding, value in zip(outer, join.parameters):
            cfg.value_sources[value.identity] = set().union(*(cfg.value_sources.get(values[binding.identity].identity, set()) for _, values, _ in exits))
            if any(values[binding.identity].identity in cfg.stack_values for _, values, _ in exits):
                cfg.stack_values.add(value.identity)
        cfg.current = join
        cfg.values = dict(initial_values)
        cfg.values.update({b.identity: value for b, value in zip(outer, join.parameters)})
        cfg.flow = FlowState.join([flow for _, _, flow in exits])


import unittest


class ConditionalTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('conditions.sev', source, Syntax())

    def native(self, source):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-conditions-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'conditions', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_native_boolean_truth_tables_and_precedence(self):
        from itertools import product
        cases = (
            ('a and b and c', lambda a, b, c: a and b and c),
            ('a or b or c', lambda a, b, c: a or b or c),
            ('a or b and c', lambda a, b, c: a or (b and c)),
            ('a and b or c', lambda a, b, c: (a and b) or c),
            ('(a or b) and c', lambda a, b, c: (a or b) and c),
            ('a and (b or c)', lambda a, b, c: a and (b or c)),
            ('not a and b or c', lambda a, b, c: ((not a) and b) or c),
            ('not (a and b) and not c', lambda a, b, c: not (a and b) and not c),
            ('not not a or b and c', lambda a, b, c: a or (b and c)),
        )
        declarations, assertions = [], []
        for index, (condition, expected) in enumerate(cases):
            declarations.append(f'def condition_{index}(a: bool, b: bool, c: bool) -> bool:\n'
                                f'    if {condition}:\n        return true\n    else:\n        return false\n')
            for values in product((False, True), repeat=3):
                arguments = ', '.join(str(value).lower() for value in values)
                assertions.append(f'    assert(condition_{index}({arguments}) == {str(expected(*values)).lower()})')
        self.native('\n'.join(declarations) + '\ntest "all boolean combinations":\n' + '\n'.join(assertions) + '\n')

    def test_native_comparisons_negation_and_multiline_conditions(self):
        self.native('''def classify(x: int, y: int, enabled: bool) -> int:
    if (
        not x == y
        and enabled
        and 0 < x < 10
        and y >= 0
    ):
        return 1
    elif not (x < y) and (x != 0 or not enabled):
        return 2
    else:
        return 3

test "comparison negation binds before boolean combinations":
    assert(classify(4, 5, true) == 1)
    assert(classify(5, 5, true) == 2)
    assert(classify(4, 5, false) == 3)
    assert(classify(0, 0, true) == 3)
    assert(classify(0, 0, false) == 2)
''')

    def test_native_short_circuit_order_and_branch_selection(self):
        self.native('''trace = 0
def probe(label: int, answer: bool) -> bool:
    module.trace = module.trace * 10 + label
    return answer
def number(value: int) -> int:
    module.trace = module.trace * 10 + value
    return value
def must_not_run() -> bool:
    assert(false)
    return true

test "and or skip unused operands and preserve left to right order":
    module.trace = 0
    selected = 0
    if probe(1, false) and must_not_run() or probe(2, true):
        selected = 1
    elif must_not_run():
        selected = 2
    else:
        selected = 3
    assert(module.trace == 12)
    assert(selected == 1)
    if probe(3, true) or must_not_run():
        selected = 4
    assert(module.trace == 123)
    assert(selected == 4)

test "elif evaluates only after earlier conditions fail":
    module.trace = 0
    selected = 0
    if probe(1, false):
        selected = 1
    elif probe(2, true) and not probe(3, false):
        selected = 2
    elif must_not_run():
        selected = 3
    else:
        selected = 4
    assert(module.trace == 123)
    assert(selected == 2)
    if false:
        selected = 5
    elif false or not true:
        selected = 6
    else:
        if selected == 2 and true:
            selected = 7
    assert(selected == 7)

test "comparison chains evaluate middle expressions once and short circuit":
    module.trace = 0
    selected = false
    if number(1) < number(2) < number(3):
        selected = true
    assert(selected)
    assert(module.trace == 123)
    module.trace = 0
    if number(3) < number(2) < number(9):
        assert(false)
    assert(module.trace == 32)
''')

    def test_invalid_conditions_and_orphan_branches_are_rejected(self):
        for condition in ('"text"', 'true and "text"', 'false or "text"'):
            with self.subTest(condition=condition):
                result = self.compile(f'def work():\n    if {condition}:\n        return\n')
                self.assertTrue(result.diagnostics)
        for source in (
            'def work():\n    else:\n        return\n',
            'def work():\n    elif true:\n        return\n',
            'def work():\n    if true:\n        return\n    else true:\n        return\n',
            'def work():\n    if true:\n        return\n    else:\n        return\n    elif false:\n        return\n',
        ):
            with self.subTest(source=source):
                self.assertTrue(self.compile(source).diagnostics)

    @unittest.expectedFailure
    def test_is_type_predicate_in_compound_condition(self):
        result = self.compile('def work(value: int | bool, enabled: bool) -> bool:\n'
                              '    if value is int and enabled:\n        return true\n'
                              '    else:\n        return false\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    @unittest.expectedFailure
    def test_is_not_type_predicate_in_compound_condition(self):
        result = self.compile('def work(value: int | bool, enabled: bool) -> bool:\n'
                              '    if value is not int or not enabled:\n        return true\n'
                              '    else:\n        return false\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
