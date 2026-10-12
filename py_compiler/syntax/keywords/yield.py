"""Owner for yield syntax and its future execution contract."""
from py_compiler.syntax.keywords.keyword import Keyword


class Yield(Keyword):
    def __init__(self):
        super().__init__("yield")

    def matches(self, node):
        return bool(node.tokens) and node.tokens[0].text == self.name

    def lower(self, node, cfg, env, local_names):
        if cfg.yield_handler is None:
            raise ValueError('yield requires an iterator grammar')
        from py_compiler.syntax.grammar.expression import expression
        cfg.yield_handler(expression(node.tokens[1:]), env, node.span)
