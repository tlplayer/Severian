from py_compiler.syntax.grammar.expression import expression


class Call:
    def matches(self, node):
        from py_compiler.syntax.catalog import assignment_spellings
        if len(node.tokens) < 2:
            return len(node.tokens) == 1 and node.tokens[0].kind == 'STRING'
        assignments = assignment_spellings(node.syntax)
        depth = 0
        for token in node.tokens:
            if token.text in ('(', '['):
                depth += 1
            elif token.text in (')', ']'):
                depth -= 1
            elif depth == 0 and token.text in assignments:
                return False
        return True

    def lower(self, node, cfg, env, local_names):
        if len(node.tokens) == 1 and node.tokens[0].kind == 'STRING':
            return
        if node.tokens[0].text in ('assert', 'expect', 'snapshot'):
            utility = cfg.imported.get(node.tokens[0].text)
            if utility is None:
                raise ValueError(node.tokens[0].text + ' requires test scope; use throw Error for runtime failures')
            utility.lower(node, cfg, env)
            return
        if node.tokens[0].text == 'await':
            from py_compiler.syntax.block.test import groups
            from py_compiler.syntax.grammar.expression import Expression
            from importlib import import_module
            arguments = groups(node.tokens[1:])
            if len(arguments) > 1:
                from py_compiler.syntax.generic.storage import release_storage
                for argument in arguments:
                    term = Expression('owned', node.tokens[0],
                        (import_module('py_compiler.syntax.operator.await').Await(), expression(argument)))
                    result = cfg.expr(term, env)
                    if result is not None:
                        release_storage(result, cfg, node.span)
                return
        term = expression(node.tokens)
        if term.kind == 'tuple' and node.tokens[0].text == 'await':
            from dataclasses import replace
            for element in term.operands:
                if element.kind != 'owned' or element.token.text != 'await':
                    first = term.operands[0]
                    element = replace(first, operands=(first.operands[0], element))
                cfg.expr(element, env)
            return
        if term.kind == 'owned' and term.token.text in ('async', 'await'):
            value = cfg.expr(term, env)
            if value is not None and value.type.family == 'task':
                from py_compiler.mir.ownership.flow import Binding
                name = '<scoped task ' + str(value.identity) + '>'
                binding = Binding(node.identity + ':task', name, value.type, False, 'own', node.span)
                env[name] = binding
                local_names.add(name)
                cfg.values[binding.identity] = value
                cfg.body.bindings.append(binding)
                cfg.record_lifetime('declare', binding.identity, binding.name)
                cfg.flow.retain(binding.identity, '*', cfg.value_dependencies.get(value.identity, ()))
            elif value is not None:
                from py_compiler.syntax.generic.storage import release_storage
                release_storage(value, cfg, node.span)
            return
        if term.kind != 'call':
            raise ValueError('expression statement requires a call')
        value = cfg.expr(term, env)
        if value and callable(getattr(value.type, 'release', None)) and value.identity not in cfg.call_result_modes:
            if value.identity in cfg.owned_results:
                from py_compiler.syntax.generic.storage import release_storage
                release_storage(value, cfg, node.span)
                cfg.owned_results.discard(value.identity)
            else:
                raise ValueError('memory result has no ownership contract')


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


class DiscardedResultTests(unittest.TestCase):
    def test_c_discard_releases_owned_results_and_preserves_borrows(self):
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.package import encode
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(encode({'package': {'name': 'discard', 'version': '1.0.0'}}))
            (root / 'src/lib.sev').write_text("""def create_values() -> move tensor[int]:
    return tensor[int](4)
def forward(values: move tensor[int]) -> move tensor[int]:
    return values
def alias(values: borrow tensor[int]) -> borrow tensor[int]:
    return borrow values
test "discarded owning calls release storage":
    create_values()
    forward(tensor[int](4))
    values = tensor[int](4)
    alias(values)
    values[0] = 42
    assert(values[0] == 42)
""")
            result = test_package(root, memory=True)
            self.assertTrue(result['success'], result)
