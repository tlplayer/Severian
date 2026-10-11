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
        import textwrap
        start = node.children[0].span.start
        line_start = cfg.source.text.rfind('\n', 0, start) + 1
        text = textwrap.dedent(cfg.source.text[line_start:node.children[-1].span.end]) + '\n'
        result = compile_source(cfg.source.path + f':fixture:{start}', text, cfg.context.syntax, native_packages=cfg.context.native_packages)
        failed = bool(result.diagnostics)
        if failed and node.header and node.header not in '\n'.join(map(str, result.diagnostics)):
            raise ValueError(f'reject fixture expected {node.header!r}, got: {result.diagnostics}')
        if failed != self.expected_failure:
            details = '\n'.join(map(str, result.diagnostics))
            raise ValueError(f'{self.spelling} fixture {"unexpectedly compiled" if not failed else "failed"}: {details}')
        return 1
