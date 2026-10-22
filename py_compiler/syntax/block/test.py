"""Tests are lexical declarations; imported test capabilities stay in the test body."""
import ast
from dataclasses import dataclass, replace
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.function.contracts import Callable
from py_compiler.syntax.block.function import Function


@dataclass(frozen=True)
class TestHeader:
    mode: str = 'unit'
    label: str = ''
    clauses: tuple = ()
    cases: tuple = ()
    options: tuple = ()

    def __iter__(self):
        return iter((self.mode, self.label, self.clauses))

    def __getitem__(self, index):
        return (self.mode, self.label, self.clauses)[index]

    def __len__(self):
        return 3


def groups(tokens):
    depth, start, result = 0, 0, []
    for index, token in enumerate(tokens):
        depth += (token.text in ('[', '(', '{')) - (token.text in (']', ')', '}'))
        if token.text == ',' and depth == 0:
            if index > start:
                result.append(tuple(tokens[start:index]))
            start = index + 1
    if start < len(tokens):
        result.append(tuple(tokens[start:]))
    return tuple(result)


def runner_options(tokens):
    import math
    from py_compiler.syntax.complex.units import quantity, match_literal, UNITS
    options, position = {}, 0
    while position < len(tokens):
        key = tokens[position].text
        if key not in ('timeout', 'repeat', 'skip', 'parallel'):
            break
        if key in options:
            raise ValueError('duplicate test runner option: ' + key)
        position += 1
        if key == 'parallel':
            options[key] = True
        else:
            if position >= len(tokens) or tokens[position].text != '(':
                raise ValueError(key + ' requires a parenthesized argument')
            end = position + 1
            while end < len(tokens) and tokens[end].text != ')':
                end += 1
            argument = tokens[position + 1:end]
            if end == len(tokens) or len(argument) != 1:
                raise ValueError(key + ' requires one literal argument')
            spelling = argument[0].text
            if key == 'skip':
                if argument[0].kind != 'STRING' or not ast.literal_eval(spelling).strip():
                    raise ValueError('skip requires a nonempty reason')
                value = ast.literal_eval(spelling)
            elif key == 'repeat':
                if not spelling.isdecimal() or int(spelling) < 1:
                    raise ValueError('repeat requires a positive integer')
                value = int(spelling)
            else:
                match = match_literal(spelling)
                if not match or match.end() != len(spelling) or UNITS[match[1]][0] != (0, 1, 0):
                    raise ValueError('timeout requires a positive duration')
                value = float(quantity((0, 1, 0)).decode(spelling))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('timeout requires a positive finite duration')
            options[key] = value
            position = end + 1
        if position < len(tokens) and tokens[position].text == 'and':
            position += 1
            if position == len(tokens) or tokens[position].text not in ('timeout', 'repeat', 'skip', 'parallel'):
                raise ValueError('and requires another test runner option')
        else:
            break
    return tuple(options.items()), tuple(tokens[position:])


def case_header(header):
    from itertools import product
    from py_compiler.syntax.grammar.expression import expression
    position = next((i for i, token in enumerate(header) if token.text == 'with' and i + 1 < len(header) and header[i + 1].text == 'cases'), None)
    if position is None:
        return None
    binding, tail = header[:position], list(header[position + 2:])
    if not binding:
        raise ValueError('test cases require parameter bindings')
    label = ast.literal_eval(tail.pop(0).text) if tail and tail[0].kind == 'STRING' else ''
    if binding[0].text == '[' and binding[-1].text == ']':
        if tail:
            raise ValueError('Cartesian cases take their values from the binding lists')
        names, choices = [], []
        for group in groups(binding[1:-1]):
            if len(group) < 3 or group[0].kind != 'IDENTIFIER' or group[1].text != 'in':
                raise ValueError('Cartesian cases require name in [values]')
            values = expression(group[2:])
            if values.kind not in ('list', 'tuple'):
                raise ValueError('Cartesian cases require an explicit value list')
            names.append(group[0])
            choices.append(values.operands)
        rows = tuple(product(*choices))
    else:
        if len(binding) < 3 or binding[0].kind != 'IDENTIFIER' or binding[1].text != '(' or binding[-1].text != ')':
            raise ValueError('table cases require a test name and named parameters')
        names = []
        for group in groups(binding[2:-1]):
            if len(group) != 1 or group[0].kind != 'IDENTIFIER':
                raise ValueError('case parameters must be names')
            names.append(group[0])
        label = label or binding[0].text
        if not tail or tail[0].text != '{' or tail[-1].text != '}':
            raise ValueError('table cases require a brace-delimited row list')
        rows = []
        for group in groups(tail[1:-1]):
            row = expression(group)
            values = row.operands if row.kind == 'tuple' else (row,)
            if len(values) != len(names):
                raise ValueError('case row does not match its parameter count')
            rows.append(values)
    if not names or len({name.text for name in names}) != len(names) or not rows:
        raise ValueError('test cases require unique parameters and at least one row')
    return TestHeader(label=label, cases=tuple(tuple(zip(names, row)) for row in rows))


class Test(DeclarationProvider):
    spelling = 'test'
    scope_kind = 'test'
    development_scope = True

    def parse_header(self, items):
        header = super().parse_header(items)
        if not header:
            return TestHeader()
        cases = case_header(header)
        if cases is not None:
            return cases
        mode = 'unit'
        options = ()
        if header[0].text == 'with':
            if len(header) < 2:
                raise ValueError('test with requires a test capability')
            if header[1].text in ('timeout', 'repeat', 'skip', 'parallel'):
                options, header = runner_options(header[1:])
            else:
                mode, header = header[1].text, header[2:]
                capabilities = [mode]
                while header and header[0].text == ',':
                    if len(header) < 2:
                        raise ValueError('expected test capability after comma')
                    capabilities.append(header[1].text)
                    header = header[2:]
                if 'memory' in capabilities:
                    if len(set(capabilities)) != len(capabilities) or any(
                            item not in ('memory', 'fuzz', 'profile', 'differential') for item in capabilities):
                        raise ValueError('memory combines with fuzz, profile or differential')
                    modes = [item for item in capabilities if item != 'memory']
                    if len(modes) > 1:
                        raise ValueError('choose fuzz, profile or differential with memory')
                    mode = modes[0] if modes else 'unit'
                    options = (('memory', True),)
                elif len(capabilities) > 1:
                    raise ValueError('unsupported test capability combination')
        clauses = ()
        if any(t.text == 'with' for t in header):
            position = next(i for i, t in enumerate(header) if t.text == 'with')
            clauses, header = tuple(header[position + 1:]), header[:position]
        if len(header) > 1 or (header and header[0].kind != 'STRING'):
            raise ValueError('expected test [with capability] ["label"]:')
        return TestHeader(mode, ast.literal_eval(header[0].text) if header else '', clauses, options=options)

    def declare_member(self, node, context, owner):
        self.declare(node, context)

    def expand(self, header, syntax):
        mode, _, _ = header
        from py_compiler.syntax.grammar.expansion import expand
        return expand(syntax, ('test', mode) if mode in syntax.expansions and mode != 'test' else ('test',))

    def declare(self, node, context):
        if node.header.mode in ('test', 'unit', 'fuzz', 'profile', 'differential'):
            from py_compiler.syntax.complex.campaign import expand
            campaign = expand(node, context)
            if campaign is not None:
                self.declare_case(campaign, context)
                context.declare_nodes(campaign.children, (*context.scope, node.identity))
                return
            if node.header.mode == 'differential':
                raise ValueError('differential test requires differential(first, second)')
        if node.header.mode == 'fuzz':
            from py_compiler.syntax.complex.fuzz import cases
            for clone, suffix in cases(node):
                self.declare_case(clone, context, suffix=suffix)
            context.declare_nodes(node.children, (*context.scope, node.identity))
            return
        if node.header.cases:
            from py_compiler.frontend.parser.blocks import Node
            for index, case in enumerate(node.header.cases):
                assignments = []
                for name, value in case:
                    tokens = (name, name.with_text('='), *value.tokens)
                    assignments.append(Node(node.identity + ':case:' + str(index) + ':' + name.text, 'sentence',
                                            node.span, tokens, node.identity, syntax=node.syntax, imports=node.imports, scope=node.scope))
                description = ', '.join(name.text + '=' + ''.join(t.text for t in value.tokens) for name, value in case)
                header = replace(node.header, cases=(), label=(node.header.label or 'case') + ' [' + description + ']')
                clone = replace(node, header=header, children=[*assignments, *node.children])
                self.declare_case(clone, context, suffix='_case_' + str(index))
            context.declare_nodes(node.children, (*context.scope, node.identity))
            return
        self.declare_case(node, context)
        if not dict(node.header.options).get('skip'):
            context.declare_nodes(node.children, (*context.scope, node.identity))

    def declare_case(self, node, context, suffix=''):
        mode, label, clauses = node.header
        if mode not in ('unit', 'profile', 'fuzz', 'differential') and mode not in context.syntax.expansions:
            from importlib import import_module
            import_module('py_compiler.syntax.block.with').apply_pack(
                mode, context, (*context.scope, node.identity), context.scope)
        entry = Callable(node, '.'.join((*context.scope, 'test_' + node.identity + suffix)), (), 'absent', ())
        if clauses:
            from importlib import import_module
            from py_compiler.syntax.function.contracts import add_clause
            guards, obligations, suffix, actions, complexity = [], [], [], [], {}
            for clause in import_module('py_compiler.syntax.block.with').clause_groups(clauses):
                add_clause(clause, guards, obligations, complexity, suffix, actions)
            entry.guards, entry.obligations, entry.suffix = tuple(guards), tuple(obligations), tuple(suffix)
            entry.failure_actions, entry.complexity = tuple(actions), complexity
        entry.scope = context.scope
        entry.imports = node.imports
        entry.test_mode, entry.test_label = mode, label
        entry.test_options = dict(node.header.options)
        if entry.test_options.get('skip'):
            entry.node = replace(node, children=[])
        context.add_callable(entry, self)

    def compile(self, entry, context):
        expected = None
        children = entry.node.children
        while len(children) == 1 and getattr(children[0].provider, 'spelling', None) == 'unsafe':
            children = children[0].children
        if (len(children) == 1 and getattr(children[0].provider, 'spelling', None) == 'throws'
                and len(children[0].header) == 1 and children[0].header[0].kind == 'STRING'):
            expected = ast.literal_eval(children[0].header[0].text)
            if not expected:
                raise ValueError('throws requires a nonempty runtime diagnostic')
            entry.expected_failure_node = children[0].identity
        body = Function().compile(entry, context)
        entry.body = body
        body.test = {'chaos-events': list(getattr(body, 'chaos_replays', ())), 'stdout-contains': getattr(body, 'stdout_expectations', []), 'stdout-excludes': getattr(body, 'stdout_exclusions', []), 'mode': entry.test_mode, 'label': entry.test_label, 'imports': [item.name for item in entry.imports]}
        body.test['output-expectations'] = getattr(body, 'output_expectations', [])
        body.test.update(entry.test_options)
        line, column = context.source.position(entry.node.tokens[0].span.start)
        body.test['location'] = {'file': context.source.path, 'line': line, 'column': column}
        if expected is not None:
            body.test['expected-failure'] = expected
        return body


import unittest


class TestScopeTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_memory_statistics_contract_pending(self):
        result = self.compile('test with memory "ledger" with {suffix memory.leaks == 0}:\n'
                              '    assert(memory.errors == 0)\n', True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_memory_capability_and_composition(self):
        for capability, mode in [('memory', 'unit'), ('memory, fuzz', 'fuzz'),
                                 ('profile, memory', 'profile')]:
            body = ('    n: int {0 <= n and n <= 2} = 0\n    assert(n >= 0)\n'
                    if mode == 'fuzz' else '    assert(true)\n')
            result = self.compile('test with ' + capability + ' "checked":\n' + body, True)
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            body = next(body for body in result.program.bodies if body.test)
            self.assertTrue(body.test['memory'])
            self.assertEqual(body.test['mode'], mode)

    def test_case_rows_and_cartesian_products_generate_distinct_tests(self):
        source = '''test square(value, expected) with cases "squares" { (2, 4), (3, 9) }:
    assert(value * value == expected)
test [a in [1, 2], b in [3, 4]] with cases "product":
    assert(a < b)
'''
        result = self.compile(source, True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        bodies = [body for body in result.program.bodies if body.test]
        self.assertEqual(len(bodies), 6)
        self.assertEqual(len({body.name for body in bodies}), 6)
        self.assertIn('squares [value=3, expected=9]', {body.test['label'] for body in bodies})

    def test_runner_options_and_skipped_body(self):
        result = self.compile('''test with timeout(250ms) and repeat(3) and parallel "options":
    assert(true)
test with skip("backend unavailable") "skip":
    unsupported_backend()
''', True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        tests = {body.test['label']: body.test for body in result.program.bodies if body.test}
        self.assertEqual(tests['options']['timeout'], 0.25)
        self.assertEqual(tests['options']['repeat'], 3)
        self.assertTrue(tests['options']['parallel'])
        self.assertEqual(tests['skip']['skip'], 'backend unavailable')

    def test_invalid_case_rows_and_runner_options_are_diagnostics(self):
        for header in ('test square(a, b) with cases {(1,)}', 'test with cases {}',
                       'test with repeat(0)', 'test with timeout(0s)',
                       'test with parallel and parallel', 'test with skip("")'):
            with self.subTest(header=header):
                self.assertTrue(self.compile(header + ':\n    assert(true)\n', True).diagnostics)

    def test_multiline_label_is_one_header(self):
        result = self.compile('test\n"""first\nsecond""":\n    assert(true)\n', True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertEqual(next(body for body in result.program.bodies if body.test).test['label'], 'first\nsecond')

    def test_expected_failure_body_resolves_its_local_record_type(self):
        result = self.compile('''test "uninitialized record":
    unsafe:
        throws "read of uninitialized array element":
            class Point:
                x: int
            values = allocate[array[Point]](1)
            point = values[0]
            assert(point.x == 0)
''', True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_c_nested_failure_preserves_unsafe_and_local_declarations(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''test "nested failure keeps local record resolution":
    unsafe:
        throws "read of uninitialized array element":
            class Point:
                x: int
            values = allocate[array[Point]](1)
            point = values[0]
            assert(point.x == 0)
test "ordinary expected failure still works":
    throws "assertion failed":
        assert(false)
'''
        with tempfile.TemporaryDirectory(prefix='sev-nested-throws-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'nested-throws', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_location_points_to_indented_test_header(self):
        result = self.compile('class Item:\n    value: int\n    test "location":\n        assert(true)\n', True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.test)
        self.assertEqual(body.test['location'], {'file': 'scopes.sev', 'line': 3, 'column': 5})

    def test_runtime_failure_expectation_is_test_metadata(self):
        result = self.compile('test "bounds":\n    throws "assertion failed":\n        assert(false)\n', True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.test)
        self.assertEqual(body.test['expected-failure'], 'assertion failed')
        self.assertEqual(body.test['location'], {'file': 'scopes.sev', 'line': 1, 'column': 1})

    def compile(self, text, include_tests):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('scopes.sev', text, Syntax(include_tests=include_tests))

    def parse(self, text, syntax=None):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.recognition import Syntax
        source, syntax = SourceFile('scopes.sev', text), syntax or Syntax()
        tokens = lex(source, syntax)
        root, errors = parse_blocks(source, tokens, syntax)
        self.assertFalse(errors, str(errors))
        return root, tokens

    def test_symbol_spans_resolve_to_block_owned_scopes(self):
        root, tokens = self.parse('class Item:\n    x: int\n    def read() -> int:\n        return self.x\ntest "nested":\n    def helper() -> int:\n        return 42\n    value = helper()\n')
        self.assertEqual(root.scope.kind, 'module')
        record, test = root.children
        self.assertEqual(record.scope.kind, 'self')
        self.assertEqual(record.children[1].scope.kind, 'local')
        self.assertIs(record.scope.owner, record.provider)
        literal = next(token for token in tokens if token.text == '42')
        scope = root.scope.at(literal)
        self.assertEqual(scope.kind, 'local')
        self.assertTrue(scope.development)
        self.assertIs(scope.parent, test.scope)
        self.assertTrue(test.scope.allows(record.scope))
        self.assertFalse(record.scope.allows(test.scope))
        self.assertTrue(test.scope.allows(scope))

    def test_production_excludes_test_types_helpers_constants_and_realizations(self):
        source = ('def identity[T](value: T) -> T:\n    return value\n'
                  'def shipped() -> int:\n    return identity(1)\n'
                  'test "development":\n'
                  '    class Fixture:\n        value: int\n'
                  '    def helper() -> string:\n        return identity("dev-only")\n'
                  '    result = helper()\n')
        production, development = self.compile(source, False), self.compile(source, True)
        for result in (production, development):
            self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertFalse(production.program.declarations)
        self.assertTrue(development.program.declarations)
        self.assertFalse(any(body.test for body in production.program.bodies))
        self.assertTrue(any(body.test for body in development.program.bodies))
        self.assertFalse(any(c.value == 'dev-only' for c in production.program.constants))
        self.assertTrue(any(c.value == 'dev-only' for c in development.program.constants))
        self.assertEqual({b.result_type.name for b in production.program.bodies if b.declaration == 'identity'}, {'i64'})
        self.assertEqual({b.result_type.name for b in development.program.bodies if b.declaration == 'identity'}, {'i64', 'string'})
        root, _ = self.parse(source)
        test_scope = root.children[-1].scope
        def development_ids(node):
            return ({node.identity} if node.scope.development else set()) | set().union(*(development_ids(child) for child in node.children))
        excluded = development_ids(root)
        self.assertIn(test_scope.identity, excluded)
        for submodule in production.program.submodules:
            self.assertFalse(excluded.intersection((*submodule.declarations, *submodule.dependencies)))

    def test_nested_class_tests_are_removed_before_member_declaration(self):
        source = 'class Item:\n    x: int\n    test "unfinished":\n        unknown_function()\ndef shipped() -> int:\n    return 1\n'
        self.assertFalse(self.compile(source, False).diagnostics)
        self.assertTrue(self.compile(source, True).diagnostics)

    def test_test_helpers_cannot_be_called_from_production(self):
        source = 'test "helpers":\n    def helper() -> int:\n        return 1\n    value = helper()\ndef shipped() -> int:\n    return helper()\n'
        for include_tests in (False, True):
            self.assertTrue(self.compile(source, include_tests).diagnostics)

    def test_test_callable_can_specialize_a_production_template(self):
        source = ('def apply[T = int, F](value: T) -> T:\n    return F(value)\n'
                  'test "callback":\n    def helper(value: int) -> int:\n        return value + 1\n'
                  '    result = apply[int, helper](1)\n')
        result = self.compile(source, True)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertTrue(any(b.declaration == 'apply' for b in result.program.bodies))
        production = self.compile(source, False)
        self.assertFalse(production.diagnostics, str(production.diagnostics))
        self.assertFalse(any(b.declaration == 'apply' for b in production.program.bodies))

    def test_test_grammar_is_inherited_but_does_not_leak_to_siblings(self):
        for spelling, header in (('throws', 'throws "expected"'), ('when', 'when true')):
            root, _ = self.parse(f'test "owner":\n    def helper():\n        {header}:\n            unimplemented\ndef sibling():\n    return\n')
            nested = root.children[0].children[0].children[0]
            self.assertEqual(nested.provider.spelling, spelling)
            self.assertTrue(nested.scope.contains_kind('test'))
            self.assertNotIn(spelling, root.children[1].syntax.block_providers)
            self.assertTrue(self.compile(f'{header}:\n    unimplemented\n', True).diagnostics)
            result = self.compile(f'test:\n    {header}:\n        unimplemented\n', True)
            self.assertFalse(result.diagnostics, str(result.diagnostics))
        root, _ = self.parse('test "owner":\n    def helper():\n        mock(example() -> 1)\ndef sibling():\n    return\n')
        self.assertEqual(root.children[0].children[0].children[0].provider.spelling, 'mock')
        self.assertNotIn('mock', root.children[1].syntax.block_providers)

    def test_required_scope_is_enforced_even_with_explicit_import(self):
        from py_compiler.syntax.grammar.expansion import expand
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.frontend.src.lib import compile_source
        syntax, _ = expand(Syntax(), ('test',))
        result = compile_source('escaped.sev', 'mock(example() -> 1)\n', syntax)
        self.assertTrue(result.diagnostics)
        self.assertIn('requires test scope', str(result.diagnostics[0]))
