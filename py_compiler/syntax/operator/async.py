"""async callable owns prefix capture and its runtime lowering boundary."""
from py_compiler.syntax.operator.operation import PrefixOperator


class Async(PrefixOperator):
    def __init__(self):
        super().__init__("async")

    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        operation = parse(6)
        owner = None
        if cursor[0] < len(tokens) and tokens[cursor[0]].text == 'with':
            cursor[0] += 1
            if cursor[0] == len(tokens):
                raise ValueError('async with requires an owning scope')
            if tokens[cursor[0]].text == 'self':
                owner = Expression('name', tokens[cursor[0]])
                cursor[0] += 1
            else:
                owner = parse(6)
        return Expression('owned', token, (self, operation, owner))

    def lower_expression(self, node, cfg, env, expected):
        raise ValueError("async callable has no declared task/ownership/lowering implementation")
