"""with owns its block header; execution requires its own lowering provider."""
from py_compiler.syntax.generic.block import BlockProvider


def clause_groups(tokens):
    """One inline clause, or a braced list with comma-terminated clauses."""
    if not tokens:
        raise ValueError('with requires a clause')
    braced = tokens[0].text == '{'
    if braced:
        if tokens[-1].text != '}':
            raise ValueError('with clauses require a closing brace')
        tokens = tokens[1:-1]
    groups, current, depth = [], [], 0
    for token in tokens:
        if token.text == ',' and depth == 0:
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

    def declare(self, node, context):
        raise ValueError("with block has no declared execution/lowering implementation")


import unittest


class WithClauseTests(unittest.TestCase):
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
