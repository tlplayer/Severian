"""Grammar-owned continuation of source sentences across physical lines."""


def indentation(source, token):
    start = max(source.text.rfind('\n', 0, token.span.start), source.text.rfind('\r', 0, token.span.start)) + 1
    prefix = source.text[start:token.span.start]
    return len(prefix) - len(prefix.lstrip(' \t'))


def next_term(tokens, index):
    for following in range(index + 1, len(tokens)):
        token = tokens[following]
        if token.kind not in ('INDENT', 'NEWLINE', 'TRIVIA', 'EOF'):
            return token
    return None


def continues_statement(source, current, following):
    items = [token for token in current if token.kind not in ('INDENT', 'TRIVIA')]
    if not items or following is None:
        return False
    first, last = items[0].text, items[-1].text
    if first in ('from', 'import') and last == '*':
        return False
    if first in ('if', 'elif', 'while', 'for', 'match', 'case', 'test', 'when') and last != ':':
        return True
    if last in ('=', ':=', '+=', '-=', '*=', '/=', '//=', '%=', '|=', '&=', '^=', '->',
                '+', '-', '*', '/', '//', '%', '**', '|', '&', '^', 'and', 'or', 'not', 'with', 'if', 'else',
                '==', '!=', '<', '>', '<=', '>=', 'in', 'is'):
        return True
    if first in ('from', 'import') and (last in (',', 'import') or following.text == 'as'):
        return True
    if first == 'import' and following.text == 'from' and (len(items) > 2 or source.text[following.span.end:].lstrip().startswith(('"', "'"))):
        return True
    deeper = indentation(source, following) > indentation(source, items[0])
    if deeper and last in ('return', 'yield', 'throw'):
        return True
    if following.text == '.' or (following.text == '{' and len(items) > 1 and items[1].text == ':'):
        return True
    operators = {'+', '-', '*', '/', '//', '%', '**', '|', '&', '^', 'and', 'or', 'if', 'else',
                 '==', '!=', '<', '>', '<=', '>=', 'in', 'is'}
    if deeper and following.text in operators and last != ':':
        return True
    return deeper and items[-1].kind == following.kind == 'STRING'


import unittest


class ContinuationTests(unittest.TestCase):
    def test_operators_return_values_and_imports_need_no_wrapper(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.recognition import Syntax
        text = ('from math import\n    pow,\n    log\n'
                'def work(a: int, b: int) -> int:\n    result =\n        a\n        + b\n    return\n        result\n'
                'if true\n    and false:\n    pass\n')
        source, syntax = SourceFile('continued.sev', text), Syntax()
        root, errors = parse_blocks(source, lex(source, syntax), syntax)
        self.assertFalse(errors, str(errors))
        self.assertEqual([node.kind for node in root.children], ['from', 'def', 'if'])
