"""Compiler acceptance fixture; available through the compiler capability."""
from py_compiler.syntax.generic.block import BlockProvider


class Accept(BlockProvider):
    spelling = 'accept'
    required_scope = 'test'
    expected_failure = False

    def parse_header(self, items):
        header = super().parse_header(items)
        if not header:
            return ''
        if self.expected_failure and len(header) == 1 and header[0].kind == 'STRING':
            import ast
            return ast.literal_eval(header[0].text)
        raise ValueError('compiler fixture accepts an optional rejection diagnostic string')

    def declare(self, node, context):
        # The compiler capability handles declarations in its isolated fixture.
        return None

    def lower(self, node, cfg, env, nodes, index):
        compile_source = cfg.imported.get('compile_source')
        if compile_source is None:
            raise ValueError('compiler fixture requires its grammar import')
        start = node.children[0].span.start
        text = fixture_source(node, cfg)
        result = compile_source(cfg.source.path + f':fixture:{start}', text, cfg.context.syntax, c_packages=cfg.context.c_packages)
        failed = bool(result.diagnostics)
        if failed and node.header and node.header not in '\n'.join(map(str, result.diagnostics)):
            raise ValueError(f'reject fixture expected {node.header!r}, got: {result.diagnostics}')
        if failed != self.expected_failure:
            details = '\n'.join(map(str, result.diagnostics))
            raise ValueError(f'{self.spelling} fixture {"unexpectedly compiled" if not failed else "failed"}: {details}')
        return 1


def fixture_source(node, cfg):
    """Recreate the lexical setup in an isolated compilation, excluding tests.

    A fixture is a local block: its arrays and views must not accidentally
    become module globals. Earlier setup and production declarations remain
    visible, while sibling fixtures cannot leak declarations into each other.
    """
    import textwrap
    root = cfg.context.root
    source = cfg.context.source

    def spelling(term):
        return getattr(term.provider, 'spelling', '')

    def code(term):
        begin = source.text.rfind('\n', 0, term.span.start) + 1
        return textwrap.dedent(source.text[begin:term.span.end]).rstrip() + '\n'

    def find(parent):
        for child in parent.children:
            if child.identity == node.identity:
                return [(parent, child)]
            nested = find(child)
            if nested:
                return [(parent, child), *nested]
        return []

    path = find(root)
    if not path:
        raise ValueError('compiler fixture has no enclosing source scope')
    production = [code(child) for child in root.children if not child.scope.development]
    fragments = []
    for parent, child in path:
        if parent is root:
            continue
        previous = []
        for sibling in parent.children:
            if sibling is child:
                break
            if spelling(sibling) not in ('accept', 'reject'):
                previous.append(code(sibling))
        fragments.append((''.join(previous), child))
    body = ''.join(code(child) for child in node.children)
    for setup, child in reversed(fragments):
        if child is not node:
            header = code(child).splitlines()[0] + '\n'
            body = header + textwrap.indent(body, '    ')
        body = setup + body
    # Explicit package imports remain module declarations in the isolated unit.
    imports = [code(child) for child in node.children if spelling(child) in ('import', 'from')]
    for imported in imports:
        body = body.replace(imported, '', 1)
    existing_imports = {code(child) for child in root.children if spelling(child) in ('import', 'from')}
    imports = [imported for imported in dict.fromkeys(imports) if imported not in existing_imports]
    return ''.join(production + imports) + '\ntest:\n' + textwrap.indent(body, '    ')


import unittest


class CompilerFixtureTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('fixtures.sev', source, Syntax(include_tests=True))

    def test_fixture_keeps_production_types_and_local_setup(self):
        result = self.compile('class Point:\n    x: int\n'
            'test with compiler:\n    point = Point(42)\n'
            '    accept:\n        assert(point.x == 42)\n'
            '    reject "has no field":\n        value = point.missing\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_fixture_storage_is_local_and_unsafe_scope_is_preserved(self):
        result = self.compile('test with compiler:\n'
            '    accept:\n        values = tensor[int](2)\n        values[0] = 42\n'
            '    unsafe:\n        accept:\n            memory = allocate[int](1)\n            drop(memory)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_valid_fragment_cannot_pass_reject_due_to_a_missing_outer_type(self):
        result = self.compile('class Point:\n    x: int\n'
            'test with compiler:\n    reject:\n        value = Point(42)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('unexpectedly compiled', str(result.diagnostics))

    def test_sibling_fixture_bindings_are_isolated(self):
        result = self.compile('test with compiler:\n'
            '    accept:\n        temporary = 42\n'
            '    reject:\n        result = temporary\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
