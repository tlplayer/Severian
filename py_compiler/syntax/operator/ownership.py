"""Binding ownership operations share the simple prefix operator grammar."""
from py_compiler.syntax.operator.operation import PrefixOperator


class Ownership(PrefixOperator):
    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        if token.text == 'share' and cursor[0] < len(tokens) and tokens[cursor[0]].text == 'atomic':
            atomic = tokens[cursor[0]]
            cursor[0] += 1
            return Expression('unary', token, (Expression('atomic-share', atomic, (parse(6),)),))
        return Expression('unary', token, (parse(6),))


OPERATORS = tuple(Ownership(name) for name in ('copy', 'view', 'borrow', 'move', 'mirror', 'share'))
OWNERSHIP = frozenset(owner.name for owner in OPERATORS)
