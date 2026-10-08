"""Tests are lexical declarations; imported test capabilities stay in the test body."""
import ast
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.function.contracts import Callable
from py_compiler.syntax.block.function import Function


class Test(DeclarationProvider):
    spelling = 'test'
    scope_kind = 'test'
    development_scope = True

    def parse_header(self, items):
        header = super().parse_header(items)
        if not header:
            return ('unit', '')
        mode = 'unit'
        if header[0].text == 'with':
            if len(header) < 2:
                raise ValueError('test with requires a test capability')
            mode, header = header[1].text, header[2:]
        if len(header) > 1 or (header and header[0].kind != 'STRING'):
            raise ValueError('expected test [with capability] ["label"]:')
        return mode, ast.literal_eval(header[0].text) if header else ''

    def declare_member(self, node, context, owner):
        self.declare(node, context)

    def expand(self, header, syntax):
        mode, _ = header
        from py_compiler.syntax.grammar.expansion import expand
        return expand(syntax, ('test',) if mode == 'unit' else ('test', mode))

    def declare(self, node, context):
        mode, label = node.header
        entry = Callable(node, '.'.join((*context.scope, 'test_' + node.identity)), (), 'absent', ())
        entry.scope = context.scope
        entry.imports = node.imports
        entry.test_mode, entry.test_label = mode, label
        context.add_callable(entry, self)
        context.declare_nodes(node.children, (*context.scope, node.identity))

    def compile(self, entry, context):
        body = Function().compile(entry, context)
        body.test = {'mode': entry.test_mode, 'label': entry.test_label, 'imports': [item.name for item in entry.imports]}
        return body


import unittest


class TestScopeTests(unittest.TestCase):
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
        for spelling in ('mock', 'throws', 'when'):
            root, _ = self.parse(f'test "owner":\n    def helper():\n        {spelling}:\n            unimplemented\ndef sibling():\n    return\n')
            nested = root.children[0].children[0].children[0]
            self.assertEqual(nested.provider.spelling, spelling)
            self.assertTrue(nested.scope.contains_kind('test'))
            self.assertNotIn(spelling, root.children[1].syntax.block_providers)
            self.assertTrue(self.compile(f'{spelling}:\n    unimplemented\n', True).diagnostics)
            result = self.compile(f'test:\n    {spelling}:\n        unimplemented\n', True)
            self.assertTrue(result.diagnostics)
            self.assertIn('no execution provider', str(result.diagnostics[0]))

    def test_required_scope_is_enforced_even_with_explicit_import(self):
        from py_compiler.syntax.grammar.expansion import expand
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.frontend.src.lib import compile_source
        syntax, _ = expand(Syntax(), ('test',))
        result = compile_source('escaped.sev', 'mock:\n    unimplemented\n', syntax)
        self.assertTrue(result.diagnostics)
        self.assertIn('requires test scope', str(result.diagnostics[0]))
