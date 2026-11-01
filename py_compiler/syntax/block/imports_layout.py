"""Import grammar accepts a parenthesized list with an optional final comma."""


def normalize_import_tokens(items):
    items = tuple(items)
    start = 3 if items and items[0].text == 'from' else 1
    if start >= len(items) or items[start].text != '(':
        return items
    depth, closing = 0, None
    for index in range(start, len(items)):
        depth += (items[index].text == '(') - (items[index].text == ')')
        if depth == 0:
            closing = index
            break
    if closing is None:
        raise ValueError('import list requires a closing parenthesis')
    names = items[start + 1:closing]
    if names and names[-1].text == ',':
        names = names[:-1]
    return (*items[:start], *names, *items[closing + 1:])


import unittest


class ImportLayoutTests(unittest.TestCase):
    def test_both_selective_import_orders_accept_vertical_lists(self):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.parser.blocks import parse_blocks
        from py_compiler.syntax.recognition import Syntax
        for text in ('from math import (\n    pow,\n    log,\n)\n',
                     'import (\n    Item,\n    Other,\n) from "types.sev"\n',
                     'import (\n    "types.sev",\n) as types\n'):
            source = SourceFile('imports.sev', text)
            syntax = Syntax()
            self.assertFalse(parse_blocks(source, lex(source, syntax), syntax)[1])
