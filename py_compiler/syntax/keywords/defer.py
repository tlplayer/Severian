"""defer owns its with-clause attachment."""
from py_compiler.syntax.keywords.keyword import Keyword


class Defer(Keyword):
    def __init__(self):
        super().__init__("defer")

    def attach(self, tokens, guards, obligations, suffix):
        from py_compiler.syntax.grammar.expression import expression
        suffix.append(expression(tokens))
