"""Interpolation expands to explicitly imported formatting and concatenation."""
import ast
import json
from py_compiler.syntax.grammar.expression import Expression, expression


def lower(node, cfg, env):
    token = node.token
    from py_compiler.syntax.symbol.forms import interpolation_parser_text
    parser_text = interpolation_parser_text(token.text)
    try:
        parsed = ast.parse(parser_text, mode='eval').body
    except SyntaxError as failure:
        raise ValueError('invalid string interpolation: ' + failure.msg) from failure
    def literal(value):
        return Expression('literal', token.with_text(json.dumps(value, ensure_ascii=False), kind='STRING'))
    if all(isinstance(part, ast.Constant) and isinstance(part.value, str) for part in parsed.values):
        return cfg.expr(literal(''.join(part.value for part in parsed.values)), env, cfg.context.type('string'))
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
            source = SourceFile(cfg.source.path + ':interpolation', ast.get_source_segment(parser_text, part.value))
            collector = getattr(cfg.context, 'lint', None)
            if collector is not None:
                lines = token.text.splitlines(keepends=True)
                start = sum(len(line) for line in lines[:part.value.lineno - 1])
                column = part.value.col_offset - int(token.text.startswith('"""') and part.value.lineno == 1)
                start += len(lines[part.value.lineno - 1].encode('utf-8')[:column].decode('utf-8'))
                collector.register_fragment(source, token.lexeme.source, token.span.start + start)
            tokens = tuple(t for t in lex(source, cfg.syntax) if t.kind not in ('EOF', 'NEWLINE', 'INDENT'))
            value = call('string', expression(tokens))
        member = Expression('member', token.with_text('concat', kind='IDENTIFIER'), (result,))
        result = Expression('call', token, (member, value))
    return cfg.expr(result, env, cfg.context.type('string'))


import unittest


class InterpolationTests(unittest.TestCase):
    def test_quote_forms_distinguish_literals_from_formatted_blocks(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.syntax.recognition import Syntax
        import ast
        from py_compiler.syntax.symbol.forms import interpolation_parser_text
        source = SourceFile('quotes.sev', '\"literal {value}\" f\"formatted {value}\" \"\"\"block {value}\"\"\"')
        strings = [token for token in lex(source, Syntax()) if token.kind in ('STRING', 'FSTRING')]
        self.assertEqual([token.kind for token in strings], ['STRING', 'FSTRING', 'FSTRING'])
        self.assertIsInstance(ast.parse(interpolation_parser_text(strings[-1].text), mode='eval').body, ast.JoinedStr)

    def test_implicit_formatted_blocks_rename_interpolated_symbols(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lint.names import fix_names
        from py_compiler.syntax.recognition import Syntax
        for value in ('"""{someValue}"""', '"""\n{someValue}\n"""'):
            result = fix_names(SourceFile('rename.sev', 'def work(someValue: int):\n    text = ' + value + '\n'), Syntax())
            self.assertIn('{some_value}', result)

    def test_interpolation_requires_library_imports(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('missing.sev', 'def show(value: string) -> string:\n    return f"value:{value}"\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('explicit import', str(result.diagnostics))
