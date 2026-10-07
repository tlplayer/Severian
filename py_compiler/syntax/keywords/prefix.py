"""prefix owns its with-clause attachment."""
from py_compiler.syntax.keywords.keyword import Keyword


class Prefix(Keyword):
    def __init__(self):
        super().__init__("prefix")

    def attach(self, tokens, guards, obligations, suffix):
        from py_compiler.syntax.sentence.syntax import expression
        guards.append(expression(tokens))
