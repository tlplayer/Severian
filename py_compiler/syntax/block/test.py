"""Tests are lexical declarations; imported test capabilities stay in the test body."""
import ast
from py_compiler.syntax.generic.block import DeclarationProvider, BlockProvider
from py_compiler.syntax.function.contracts import Callable
from py_compiler.syntax.block.function import Function


class Test(DeclarationProvider):
    spelling = 'test'

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
        if mode == 'unit':
            from py_compiler.syntax.grammar.expansion import visible_imports
            return syntax, visible_imports(syntax)
        from py_compiler.syntax.grammar.expansion import expand
        return expand(syntax, (mode,))

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


class Fixture(BlockProvider):
    spelling = 'accept'
    expected_failure = False

    def parse_header(self, items):
        header = super().parse_header(items)
        if header:
            raise ValueError('compiler fixture takes only a body')
        return header

    def declare(self, node, context):
        # The compiler capability handles declarations in its isolated fixture.
        return None

    def lower(self, node, cfg, env, nodes, index):
        compile_source = cfg.imported.get('compile_source')
        if compile_source is None:
            raise ValueError('compiler fixture requires its grammar import')
        import textwrap
        start = node.children[0].span.start
        line_start = cfg.source.text.rfind('\n', 0, start) + 1
        text = textwrap.dedent(cfg.source.text[line_start:node.children[-1].span.end]) + '\n'
        result = compile_source(cfg.source.path + f':fixture:{start}', text, cfg.context.syntax)
        failed = bool(result.diagnostics)
        if failed != self.expected_failure:
            details = '\n'.join(map(str, result.diagnostics))
            raise ValueError(f'{self.spelling} fixture {"unexpectedly compiled" if not failed else "failed"}: {details}')
        return 1


class Reject(Fixture):
    spelling = 'reject'
    expected_failure = True
