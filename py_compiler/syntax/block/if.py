from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.hir.thir.src.model import Edge, Terminator
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
        next_values, next_flow = initial_values, initial_flow
        for node in chain:
            cfg.values, cfg.flow = dict(next_values), next_flow.fork()
            validate = getattr(node.provider, 'validate_captures', None)
            if validate:
                validate(cfg, env)
            if node.provider.has_condition:
                evaluate = getattr(node.provider, 'condition_value', None)
                condition = evaluate(cfg, node, env) if evaluate else cfg.expr(expression(node.header), env, cfg.syntax.types["bool"])
                if condition.type.name != "bool":
                    raise ValueError("if/elif condition must satisfy bool")
                body, following = cfg.block(), cfg.block()
                cfg.current.terminator = Terminator("conditional", (Edge(body.identity), Edge(following.identity)), condition)
                next_values, next_flow = dict(cfg.values), cfg.flow.fork()
                cfg.current = body
            if node.provider.has_condition and evaluate is None:
                from py_compiler.syntax.operator.logical import refinement
                with refinement(cfg, expression(node.header), env):
                    cfg.scope(node.children, dict(env))
            else:
                cfg.scope(node.children, dict(env))
            cfg.flow.end_scope(cfg.live_binding_ids(env) | {b.identity for scope in cfg.active_scopes for b in scope.values()})
            if cfg.current.terminator is None:
                exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
            if node.provider.has_condition:
                cfg.current = following
        if chain[-1].provider.has_condition:
            exits.append((cfg.current, next_values, next_flow))
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

    def c(self, source):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-conditions-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'conditions', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_c_boolean_truth_tables_and_precedence(self):
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
        self.c('\n'.join(declarations) + '\ntest "all boolean combinations":\n' + '\n'.join(assertions) + '\n')

    def test_c_comparisons_negation_and_multiline_conditions(self):
        self.c('''def classify(x: int, y: int, enabled: bool) -> int:
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

    def test_c_short_circuit_order_and_branch_selection(self):
        self.c('''trace = 0
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
        for condition in ('42', 'true and 42', 'false or 42'):
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


class ConditionalOwnershipTests(unittest.TestCase):
    def test_condition_consumption_reaches_else_and_fallthrough(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for tail in ('    observed = owner\n',
                     '    else:\n        observed = owner\n',
                     '    elif consume(owner):\n        return\n'):
            for mode in ('view', 'move'):
                with self.subTest(tail=tail, mode=mode):
                    source = (f'def consume(value: {mode} string) -> bool:\n    return true\n'
                              'def work():\n    owner = "owned"\n'
                              '    if consume(owner):\n        return\n' + tail)
                    result = compile_source('condition-ownership.sev', source, Syntax())
                    if mode == 'move':
                        self.assertTrue(result.diagnostics)
                        self.assertIn('moved', str(result.diagnostics))
                    else:
                        self.assertFalse(result.diagnostics, str(result.diagnostics))


class ExpressionEffectTests(unittest.TestCase):
    compile = ConditionalTests.compile
    c = ConditionalTests.c

    def test_c_short_circuit_mutation_borrow_and_consumption(self):
        self.c('''trace = 0
consumed = 0
class Cell:
    value: int

def touch(cell: borrow Cell, label: int, answer: bool) -> bool:
    cell.value += 1
    module.trace = module.trace * 10 + label
    return answer

def number(cell: borrow Cell, value: int) -> int:
    cell.value += 1
    module.trace = module.trace * 10 + value
    return value

def consume(value: move string) -> bool:
    module.consumed += 1
    return true

def consume_number(value: move string) -> int:
    module.consumed += 1
    return 3

test "skipped RHS has no mutation or call effects":
    cell = Cell(0)
    module.trace = 0
    assert(not (false and touch(cell, 1, true)))
    assert(true or touch(cell, 2, false))
    assert(cell.value == 0)
    assert(module.trace == 0)
    assert(true and touch(cell, 3, true))
    assert(false or touch(cell, 4, true))
    assert(cell.value == 2)
    assert(module.trace == 34)
    cell.value = 9
    drop cell

test "nested short circuit observes prior storage mutations":
    cell = Cell(0)
    module.trace = 0
    selected = touch(cell, 1, false) and touch(cell, 2, true) or touch(cell, 3, true)
    assert(selected)
    assert(module.trace == 13)
    assert(cell.value == 2)
    assert(touch(cell, 4, true) and cell.value == 3)
    assert(touch(cell, 5, false) or cell.value == 4)

test "negation and comparison chains preserve borrowed effects":
    cell = Cell(0)
    module.trace = 0
    assert(not number(cell, 3) == number(cell, 4))
    assert(module.trace == 34)
    assert(cell.value == 2)
    module.trace = 0
    assert(number(cell, 1) < number(cell, 2) < number(cell, 3))
    assert(module.trace == 123)
    assert(cell.value == 5)
    module.trace = 0
    assert(not (number(cell, 3) < number(cell, 2) < number(cell, 9)))
    assert(module.trace == 32)
    assert(cell.value == 7)

test "consuming calls run only on reached RHS edges":
    module.consumed = 0
    skipped_and = "one"
    assert(not (false and consume(skipped_and)))
    skipped_or = "two"
    assert(true or consume(skipped_or))
    assert(module.consumed == 0)
    reached_and = "three"
    assert(true and consume(reached_and))
    reached_or = "four"
    assert(false or consume(reached_or))
    assert(module.consumed == 2)

test "comparison chains skip or reach consuming final operands":
    module.consumed = 0
    skipped = "skipped"
    assert(not (2 < 1 < consume_number(skipped)))
    assert(module.consumed == 0)
    reached = "reached"
    assert(0 < 1 < consume_number(reached))
    assert(module.consumed == 1)
''')

    def test_live_borrow_rejects_mutating_rhs_and_dead_borrow_allows_it(self):
        base = '''class Cell:
    value: int
def touch(cell: borrow Cell) -> bool:
    cell.value += 1
    return true
'''
        for operator in ('and', 'or'):
            for live in (False, True):
                with self.subTest(operator=operator, live=live):
                    source = base + ('def work(flag: bool):\n    cell = Cell(0)\n'
                                     '    alias = borrow cell\n'
                                     '    before = alias.value\n'
                                     f'    result = flag {operator} touch(cell)\n')
                    if live:
                        source += '    after = alias.value\n'
                    result = self.compile(source)
                    if live:
                        self.assertTrue(result.diagnostics)
                        self.assertIn('borrow', str(result.diagnostics))
                    else:
                        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_skipped_rhs_still_requires_bool_and_valid_types(self):
        for condition in ('false and 1', 'true or 1', 'false and missing()', 'true or missing()'):
            with self.subTest(condition=condition):
                result = self.compile(f'def work():\n    result = {condition}\n')
                self.assertTrue(result.diagnostics)

    def test_comparison_chain_possible_consumption_rejects_joined_read(self):
        for use_after in (False, True):
            with self.subTest(use_after=use_after):
                source = ('def consume(value: move string) -> int:\n    return 3\n'
                          'def work(limit: int):\n    owner = "owned"\n'
                          '    selected = 0 < limit < consume(owner)\n')
                if use_after:
                    source += '    observed = owner\n'
                result = self.compile(source)
                if use_after:
                    self.assertTrue(result.diagnostics)
                    self.assertIn('moved', str(result.diagnostics))
                else:
                    self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_borrow_retained_by_rhs_pins_source_after_join(self):
        base = '''class Holder:
    value: pointer[int]
def retain(holder: borrow Holder, source: array[int]) -> bool:
    unsafe:
        holder.value = pointer[int](source)
    return true
def work(flag: bool):
    unsafe:
        original = array[int,1]()
        pointer = pointer[int](original)
        holder = Holder(pointer)
        source = array[int,1]()
'''
        for operator in ('and', 'or'):
            for release_holder in (False, True):
                with self.subTest(operator=operator, release_holder=release_holder):
                    source = base + f'        selected = flag {operator} retain(holder, source)\n'
                    if release_holder:
                        source += '        drop holder\n        drop source\n'
                    else:
                        source += '        drop source\n        observed = holder.value[0]\n'
                    result = self.compile(source)
                    if release_holder:
                        self.assertFalse(result.diagnostics, str(result.diagnostics))
                    else:
                        self.assertTrue(result.diagnostics)
                        self.assertIn('borrow', str(result.diagnostics))


class TypePredicateContractTests(unittest.TestCase):
    """Pending acceptance tests, not evidence that predicate lowering exists.

    Every rejection case first requires a supported positive predicate, so a
    parser rejecting `is` cannot masquerade as a successful refinement check.
    """
    compile = ConditionalTests.compile
    c = ConditionalTests.c

    def require_predicates(self):
        result = self.compile('def work(value: int | string) -> bool:\n'
                              '    return value is int\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    @unittest.expectedFailure
    def test_and_rhs_receives_true_type_and_else_merges_both_false_paths(self):
        self.c('''def classify(value: int | string) -> int:
    if value is int and value > 0:
        return value + 1
    else:
        if value is int:
            return value - 1
        else:
            return len(value)
test:
    assert(classify(2) == 3)
    assert(classify(0) == -1)
    assert(classify(-2) == -3)
    assert(classify("text") == 4)
''')

    @unittest.expectedFailure
    def test_or_rhs_receives_false_type_and_not_swaps_edges(self):
        self.c('''def accepted(value: int | string) -> bool:
    return value is not int or value > 0
def positive(value: int | string) -> bool:
    return not value is not int and value > 0
test:
    assert(accepted("text"))
    assert(accepted(1))
    assert(not accepted(0))
    assert(not positive("text"))
    assert(positive(1))
    assert(not positive(0))
''')

    @unittest.expectedFailure
    def test_false_join_and_post_condition_do_not_keep_true_only_type(self):
        self.require_predicates()
        for source in (
            'def work(value: int | string) -> int:\n'
            '    if value is int and value > 0:\n        return 1\n'
            '    else:\n        return value + 1\n',
            'def work(value: int | string) -> int:\n'
            '    selected = value is int and value > 0\n    return value + 1\n',
            'def work(value: int | string) -> bool:\n'
            '    return value is int or value > 0\n',
        ):
            result = self.compile(source)
            self.assertTrue(result.diagnostics, source)
            self.assertNotIn('unsupported expression suffix', str(result.diagnostics))

    @unittest.expectedFailure
    def test_terminating_edge_does_not_widen_continuing_type(self):
        self.c('''def increment(value: int | string) -> int:
    if value is not int:
        return 0
    return value + 1
test:
    assert(increment("text") == 0)
    assert(increment(4) == 5)
''')

    @unittest.expectedFailure
    def test_type_test_evaluates_effectful_operand_once(self):
        self.c('''calls = 0
def sample() -> int | string:
    module.calls += 1
    return 3
test:
    module.calls = 0
    assert(sample() is int)
    assert(module.calls == 1)
''')

    @unittest.expectedFailure
    def test_mutating_alias_invalidates_refinement_before_rhs_read(self):
        self.require_predicates()
        source = '''class Cell:
    value: int | string
def replace_value(cell: borrow Cell) -> bool:
    cell.value = "changed"
    return true
def work(cell: borrow Cell) -> bool:
    return cell.value is int and replace_value(cell) and cell.value > 0
'''
        result = self.compile(source)
        self.assertTrue(result.diagnostics)
        self.assertNotIn('unsupported expression suffix', str(result.diagnostics))

    @unittest.expectedFailure
    def test_refined_read_preserves_ownership_and_consume_invalidates_it(self):
        self.require_predicates()
        base = 'def consume(value: move string) -> bool:\n    return true\n'
        valid = base + ('def work(value: move int | string) -> bool:\n'
                        '    return value is string and consume(value)\n')
        result = self.compile(valid)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        invalid = base + ('def work(value: move int | string) -> bool:\n'
                          '    return value is string and consume(value) and len(value) > 0\n')
        result = self.compile(invalid)
        self.assertTrue(result.diagnostics)
        self.assertIn('moved', str(result.diagnostics))
