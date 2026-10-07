"""fix owns its with-clause attachment."""
from py_compiler.syntax.keywords.keyword import Keyword


class Fix(Keyword):
    def __init__(self):
        super().__init__("fix")

    def attach(self, tokens, guards, obligations, suffix):
        from py_compiler.syntax.sentence.syntax import expression
        obligations.append(expression(tokens))
