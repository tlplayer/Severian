"""Binding ownership operations share the simple prefix operator grammar."""
from py_compiler.syntax.operator.operation import PrefixOperator


class Ownership(PrefixOperator):
    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        return Expression('unary', token, (parse(6),))


OPERATORS = tuple(Ownership(name) for name in ('copy', 'view', 'borrow', 'move'))
OWNERSHIP = frozenset(owner.name for owner in OPERATORS)
