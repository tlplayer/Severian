"""Interpolation expands to explicitly imported formatting and concatenation."""
import ast
import json
from py_compiler.syntax.grammar.expression import Expression, expression


def lower(node, cfg, env):
    token = node.token
    try:
        parsed = ast.parse(token.text, mode='eval').body
    except SyntaxError as failure:
        raise ValueError('invalid string interpolation: ' + failure.msg) from failure
    def literal(value):
        return Expression('literal', token.with_text(json.dumps(value, ensure_ascii=False), kind='STRING'))
    def call(function, *args):
        return Expression('call', token, (Expression('name', token.with_text(function, kind='IDENTIFIER')), *args))
    result = literal('')
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.frontend.lexer.lexer import lex
    for part in parsed.values:
        if isinstance(part, ast.Constant):
            value = literal(part.value)
        else:
            if part.conversion != -1 or part.format_spec is not None:
                raise ValueError('interpolation conversion flags and format specifications require a formatting provider')
            source = SourceFile(cfg.source.path + ':interpolation', ast.get_source_segment(token.text, part.value))
            tokens = tuple(t for t in lex(source, cfg.syntax) if t.kind not in ('EOF', 'NEWLINE'))
            value = call('string', expression(tokens))
        member = Expression('member', token.with_text('concat', kind='IDENTIFIER'), (result,))
        result = Expression('call', token, (member, value))
    return cfg.expr(result, env, cfg.context.type('string'))


import unittest


class InterpolationTests(unittest.TestCase):
    def test_interpolation_requires_library_imports(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('missing.sev', 'def show(value: string) -> string:\n    return f"value:{value}"\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('explicit import', str(result.diagnostics))
