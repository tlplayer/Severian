from py_compiler.syntax.sentence.syntax import expression
from py_compiler.syntax.complex.object import Assert


class Call:
    def matches(self, node):
        from py_compiler.syntax.catalog import assignment_spellings
        if len(node.tokens) < 2:
            return False
        assignments = assignment_spellings(node.syntax)
        return not any(token.text in assignments for token in node.tokens)

    def lower(self, node, cfg, env, local_names):
        if node.tokens[0].text == 'assert':
            value = cfg.expr(expression(node.tokens[1:]), env, cfg.syntax.types['bool'])
            cfg.effect('assert', (value,), Assert(f'assertion failed at {cfg.source.path}:{cfg.source.position(node.span.start)[0]}'), node.span)
            return
        term = expression(node.tokens)
        if term.kind != 'call':
            raise ValueError('expression statement requires a call')
        cfg.expr(term, env)


import unittest


class CallTests(unittest.TestCase):
    def sentence(self, text, syntax):
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.frontend.parser.blocks import parse_blocks
        source = SourceFile('call-routing.sev', text + '\n')
        root, errors = parse_blocks(source, lex(source, syntax), syntax)
        self.assertFalse(errors, str(errors))
        return root.children[0]

    def test_registered_assignments_are_not_call_statements(self):
        from py_compiler.syntax.catalog import assignment_spellings
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        for spelling in assignment_spellings(syntax):
            with self.subTest(operator=spelling):
                self.assertFalse(Call().matches(self.sentence(f'value {spelling} 2', syntax)))
                self.assertFalse(Call().matches(self.sentence(f'item.value {spelling} 2', syntax)))
        for text in ('work()', 'item.work()', 'assert(value == 2)'):
            self.assertTrue(Call().matches(self.sentence(text, syntax)), text)

    def test_compound_assignments_reach_owner_atoms(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        cases = {
            'int': ('+=', '-=', '*=', '/=', '//=', '%=', '**=', '<<=', '>>=', '&=', '|=', '^='),
            'float': ('+=', '-=', '*=', '/=', '//=', '%=', '**='),
        }
        for type_, operators in cases.items():
            for operator in operators:
                with self.subTest(type=type_, operator=operator):
                    source = f'def work(value: {type_}) -> {type_}:\n    value {operator} 2\n    return value\n'
                    result = compile_source('assignments.sev', source, Syntax())
                    self.assertFalse(result.diagnostics, str(result.diagnostics))
                    body = next(body for body in result.program.bodies if body.declaration == 'work')
                    operations = [op for block in body.blocks for op in block.operations if op.kind == 'scalar']
                    self.assertEqual(len(operations), 1)
                    self.assertEqual(operations[0].atom.implementation.spelling, operator[:-1])
                    self.assertEqual(operations[0].atom.owner, Syntax().types[type_])

    def test_member_compound_assignment_routes_to_field_storage(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = ('class Counter:\n    value: int = 8\n'
                  'def work() -> int:\n    item = Counter()\n    item.value /= 2\n    return item.value\n')
        result = compile_source('field-assignment.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'work')
        from py_compiler.syntax.generic.storage import FieldPlace
        stores = [op for block in body.blocks for op in block.operations
                  if op.kind == 'write' and isinstance(op.payload, FieldPlace)]
        self.assertEqual(len(stores), 2)
