"""await callable owns prefix capture and its runtime lowering boundary."""
from py_compiler.syntax.keywords.keyword import Keyword


class Await(Keyword):
    def __init__(self):
        super().__init__("await")

    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.sentence.syntax import Expression
        return Expression('owned', token, (self, parse(6)))

    def lower_expression(self, node, cfg, env, expected):
        raise ValueError("await callable has no declared task/ownership/lowering implementation")
