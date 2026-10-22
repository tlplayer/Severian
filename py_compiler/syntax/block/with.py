"""With clauses and lexical import packs reuse declared c library contracts."""
from py_compiler.syntax.generic.block import BlockProvider
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBinding:
    name: object
    value: tuple


@dataclass(frozen=True)
class Placement:
    mode: str


def clause_groups(tokens, *, tuple_bindings=False):
    """One inline clause, or a braced list with comma-terminated clauses."""
    if not tokens:
        raise ValueError('with requires a clause')
    braced = tokens[0].text == '{'
    if braced:
        if tokens[-1].text != '}':
            raise ValueError('with clauses require a closing brace')
        tokens = tokens[1:-1]
    groups, current, depth = [], [], 0
    for index, token in enumerate(tokens):
        if token.text == ',' and depth == 0:
            if tuple_bindings and current and all(t.kind == 'IDENTIFIER' or t.text == ',' for t in current):
                tail = tokens[index + 1:]
                end = next((i for i, t in enumerate(tail) if t.kind != 'IDENTIFIER' and t.text != ','), None)
                if end is not None and tail[end].text in ('=', ':='):
                    current.append(token)
                    continue
            if not braced:
                raise ValueError('multiple with clauses require braces')
            if not current:
                raise ValueError('empty with clause')
            groups.append(tuple(current))
            current = []
            continue
        current.append(token)
        depth += (token.text in ('(', '[', '{')) - (token.text in (')', ']', '}'))
    if current:
        if braced:
            raise ValueError("with clauses must end with ','")
        groups.append(tuple(current))
    return tuple(groups)


class With(BlockProvider):
    spelling = "with"

    def parse_header(self, items):
        header = tuple(items[1:-1] if items[-1].text == ':' else items[1:])
        if [t.text for t in header] in (['simd'], ['self', 'and', 'simd'], ['gpu'], ['self', 'and', 'gpu']):
            if items[-1].text != ':':
                raise ValueError('GPU/SIMD placement requires a body')
            return Placement(header[-1].text)
        if len(header) >= 3 and header[-2].text == 'as' and header[-1].kind == 'IDENTIFIER':
            from py_compiler.syntax.grammar.expression import expression
            expression(header[:-2])
            if items[-1].text != ':':
                raise ValueError('resource binding requires an indented body')
            return ResourceBinding(header[-1], header[:-2])
        if not header or header[0].kind != 'IDENTIFIER' or (len(header) > 1 and
                (len(header) != 2 or header[1].text != 'as')):
            raise ValueError('expected with pack: or with pack as:')
        if len(header) == 2 and items[-1].text != ':':
            raise ValueError('pack declaration requires an indented body')
        return header

    def has_body(self, node):
        return node.tokens[-1].text == ':'

    def declare(self, node, context):
        if isinstance(node.header, Placement):
            context.declare_nodes(node.children, (*context.scope, node.identity))
            return
        if isinstance(node.header, ResourceBinding):
            context.declare_nodes(node.children, (*context.scope, node.identity))
            return
        if len(node.header) == 2:
            return
        if not self.has_body(node):
            raise ValueError('bodyless with is only valid inside an import pack')
        scope = (*context.scope, node.identity)
        apply_pack(node.header[0].text, context, scope, context.scope)
        context.declare_nodes(node.children, scope)

    def lower(self, node, cfg, env, nodes, index):
        if isinstance(node.header, Placement):
            previous = getattr(cfg, 'placement', None)
            cfg.placement = node.header.mode
            try:
                cfg.scope(node.children, dict(env))
            finally:
                cfg.placement = previous
            return 1
        if isinstance(node.header, ResourceBinding):
            from dataclasses import replace
            name = node.tokens[0].with_text(node.header.name.text, kind='IDENTIFIER')
            assignment = replace(node, identity=node.identity + ':resource', provider=None, children=[],
                tokens=(name, name.with_text('=', kind='OPERATOR'), *node.header.value))
            previous = cfg.storage_factory
            cfg.storage_factory = None
            try:
                cfg.scope([assignment, *node.children], dict(env))
                cfg.flow.end_scope(cfg.live_binding_ids(env))
            finally:
                cfg.storage_factory = previous
            return 1
        if len(node.header) == 2:
            return 1
        previous = cfg.storage_factory
        cfg.storage_factory = None
        try:
            cfg.scope(node.children, dict(env))
            cfg.flow.end_scope(cfg.live_binding_ids(env))
        finally:
            cfg.storage_factory = previous
        return 1


def register_packs(nodes, context):
    if not hasattr(context, 'import_packs'):
        context.import_packs = {}
    for node in nodes:
        if isinstance(node.provider, With) and not isinstance(node.header, (ResourceBinding, Placement)) and len(node.header) == 2:
            key = (*context.scope, node.header[0].text)
            if key in context.import_packs and context.import_packs[key] is not node:
                raise ValueError('duplicate import pack ' + node.header[0].text)
            context.import_packs[key] = node


def apply_pack(name, context, destination, lookup, active=()):
    from importlib import import_module
    Import = import_module('py_compiler.syntax.block.import').Import
    packs = getattr(context, 'import_packs', {})
    key = next(((*lookup[:length], name) for length in range(len(lookup), -1, -1)
                if (*lookup[:length], name) in packs), None)
    if key is None:
        raise ValueError('unknown import pack ' + name)
    if key in active:
        raise ValueError('cyclic import pack ' + ' -> '.join(k[-1] for k in (*active, key)))
    definition = packs[key]
    previous = context.scope
    context.scope = destination
    try:
        for child in definition.children:
            if isinstance(child.provider, Import):
                child.provider.declare(child, context)
            elif isinstance(child.provider, With) and not isinstance(child.header, ResourceBinding) and len(child.header) == 1 and not child.children:
                apply_pack(child.header[0].text, context, destination, key[:-1], (*active, key))
            else:
                raise ValueError('import packs contain imports and bodyless with pack inclusions')
    finally:
        context.scope = previous


import unittest


class WithClauseTests(unittest.TestCase):
    def test_resource_scope_and_shared_destructor_run_once(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''drops: int = 0
class Resource:
    value: int
    def drop():
        module.drops += 1
def early():
    with Resource(42) as resource:
        return resource.value
test "resource lifetime":
    module.drops = 0
    with Resource(42) as resource:
        assert(resource.value == 42)
        assert(module.drops == 0)
    assert(module.drops == 1)
    assert(early() == 42)
    assert(module.drops == 2)
test "share transfers resources without destroying them":
    module.drops = 0
    owner = Resource(42)
    first = share owner
    assert(module.drops == 0)
    drop(owner)
    assert(first.value == 42)
    assert(module.drops == 0)
    drop(first)
    assert(module.drops == 1)
'''
        with tempfile.TemporaryDirectory(prefix='sev-resource-binding-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'resource-binding', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)
        invalid = self.compile('class Resource:\n    value: int\ndef work():\n    with Resource(42) as resource:\n        value = resource.value\n    return resource.value\n')
        self.assertIn('unknown binding', str(invalid.diagnostics))

    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('with-clauses.sev', source, Syntax())

    def test_inline_and_braced_single_clauses_share_the_same_contract(self):
        from py_compiler.mlir.src.lib import lower, render
        for clause in ('x == 0 -> failed(1, 2)', 'fix x >= 0', 'suffix x == 0',
                       'x == 0 -> Error("expected zero")'):
            for header in (clause, '{' + clause + ',}'):
                with self.subTest(header=header):
                    result = self.compile('def failed(a: int, b: int):\n    return\n'
                                          f'def work(x: int) -> int with {header}:\n    return x\n')
                    self.assertFalse(result.diagnostics, str(result.diagnostics))
                    render(lower(result.program))

    def test_multiple_clauses_require_braces_and_a_final_comma(self):
        for header, diagnostic in (
            ('x >= 0, x <= 3', 'multiple with clauses require braces'),
            ('{x >= 0, x <= 3}', "with clauses must end with ','"),
            ('{x >= 0,,}', 'empty with clause'),
            ('', 'with requires a clause'),
        ):
            result = self.compile(f'def work(x: int) with {header}:\n    return\n')
            self.assertTrue(result.diagnostics)
            self.assertIn(diagnostic, str(result.diagnostics[0]))

    def test_inline_loop_phase_and_control_clauses(self):
        from py_compiler.mlir.src.lib import lower, render
        for clause in ('i = 0', 'prefix i == 0', 'fix i >= 0', 'suffix i += 1',
                       'fix false -> continue', 'fix i == 3 -> break'):
            result = self.compile('def work() -> int:\n    i = 0\n'
                                  f'    while i < 4 with {clause}:\n        i += 1\n    return i\n')
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            render(lower(result.program))


class ImportPackTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        package = {'package': {'name': 'numbers'}, 'objects': ['numbers.o'], 'types': [],
                   'exports': [{'name': 'answer', 'symbol': 'library_answer', 'parameters': [],
                                'result': 'int', 'result-sources': []}]}
        return compile_source('packs.sev', source, Syntax(), c_packages={'numbers': package})

    def test_static_import_and_composed_pack_keep_c_dependency(self):
        from py_compiler.mlir.src.lib import lower, render
        for source in (
            'import answer from numbers\ndef work() -> int:\n    return answer()\n',
            'with base as:\n    import answer from numbers\n'
            'with combined as:\n    with base\n    with base\n'
            'def work() -> int:\n    with combined:\n        return answer()\n',
            'with base as:\n    from numbers import answer\n'
            'test with base "packed":\n    assert(answer() == 42)\n',
            'with base as:\n    import answer from numbers\n'
            'def work() -> int:\n    total = 0\n'
            '    for value in [1,2] with base:\n        total += answer() + value\n'
            '    return total\n',
        ):
            with self.subTest(source=source):
                result = self.compile(source)
                self.assertFalse(result.diagnostics, str(result.diagnostics))
                self.assertIn('func.call @library_answer()', render(lower(result.program)))

    def test_pack_import_does_not_escape_to_sibling(self):
        result = self.compile('with base as:\n    import answer from numbers\n'
            'def work() -> int:\n    with base:\n        value = answer()\n    return answer()\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('no callable matches', str(result.diagnostics))

    def test_cycles_duplicates_unknown_packs_and_runtime_setup_are_rejected(self):
        for source, message in (
            ('with a as:\n    with b\nwith b as:\n    with a\nwith a:\n    value = 1\n', 'cyclic import pack'),
            ('with a as:\n    import answer from numbers\nwith a as:\n    import answer from numbers\n', 'duplicate import pack'),
            ('with missing:\n    value = 1\n', 'unknown import pack'),
            ('with a as:\n    value = 1\nwith a:\n    value = 2\n', 'import packs contain'),
            ('def work():\n    values = tuple[int,1](1)\n    for value in values with simd:\n        value + 1\n', 'unknown import pack simd'),
        ):
            with self.subTest(source=source):
                result = self.compile(source)
                self.assertTrue(result.diagnostics)
                self.assertIn(message, str(result.diagnostics))

    def test_tensor_iteration_requires_an_iterator_provider(self):
        result = self.compile('with base as:\n    import answer from numbers\n'
            'def work():\n    values = tensor[int,(2,)](tuple[int,2](1,2))\n'
            '    for value in values with base:\n        answer()\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('for requires one applicable iterator(self) grammar', str(result.diagnostics))
