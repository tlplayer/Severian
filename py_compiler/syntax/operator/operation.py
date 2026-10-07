"""Simple prefix operators own an operation word and one object capture."""
from py_compiler.syntax.keywords.keyword import Keyword
from py_compiler.syntax.generic.owned import OwnedGrammar, Capture
from py_compiler.syntax.generic.grammar import Match


class OperatorGrammar(OwnedGrammar):
    def __init__(self, owner):
        super().__init__(owner, 'F.prefix', (owner.name, Capture('object', 'expression')))

    def recognize(self, window):
        if len(window.tokens) < 2 or window.tokens[0].text != self.owner.name:
            return None
        return Match(self, window, window.end, {'operation': window.tokens[0], 'object': window.tokens[1:]})


class PrefixOperator(Keyword):
    def grammars(self):
        return (*super().grammars(), OperatorGrammar(self))

    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        return Expression('owned', token, (self, parse(6)))


import unittest


class OperatorTests(unittest.TestCase):
    def test_prefix_operators_declare_their_object_capture(self):
        from importlib import import_module
        from py_compiler.syntax.operator.drop import Drop
        from py_compiler.syntax.operator.ownership import OPERATORS
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.generic.grammar import SourceWindow
        from py_compiler.syntax.recognition import Syntax
        owners = (*OPERATORS, Drop(), import_module('py_compiler.syntax.operator.async').Async(),
                  import_module('py_compiler.syntax.operator.await').Await())
        syntax = Syntax()
        for owner in owners:
            source = SourceFile('operator.sev', owner.name + ' item')
            tokens = tuple(lex(source, syntax)[:-1])
            grammar = next(rule for rule in owner.grammars() if rule.role == 'F.prefix')
            match = grammar.recognize(SourceWindow(source, 0, len(source.text), tokens))
            self.assertEqual(match.captures['operation'].text, owner.name)
            self.assertEqual(match.captures['object'][0].text, 'item')

    def test_move_and_drop_preserve_ownership_diagnostics(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        valid = 'def work() -> int:\n    item = 3\n    moved := move item\n    drop moved\n    return 0\n'
        result = compile_source('ownership.sev', valid, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        invalid = 'def work() -> int:\n    item = 3\n    drop item\n    return item\n'
        self.assertTrue(compile_source('ownership.sev', invalid, Syntax()).diagnostics)
