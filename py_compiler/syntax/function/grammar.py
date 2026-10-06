"""Expression grammar exposes binding information through the common G contract."""
from py_compiler.syntax.generic.grammar import Grammar, Match


class ExpressionGrammar(Grammar):
    role = 'F'

    def __init__(self, syntax):
        self.syntax = syntax

    def recognize(self, window):
        if not window.tokens:
            return None
        return Match(self, window, window.end, {'terms': window.tokens})

    def construct(self, match):
        from py_compiler.syntax.sentence.syntax import parse_expression
        return parse_expression(match.captures['terms'], self.syntax)


def operators(syntax=None):
    from py_compiler.syntax.catalog import operator_bindings
    from py_compiler.syntax.recognition import Syntax
    return tuple(operator_bindings(syntax or Syntax()).values())
