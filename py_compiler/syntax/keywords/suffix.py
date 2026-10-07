"""suffix owns its with-clause attachment."""
from py_compiler.syntax.keywords.keyword import Keyword


class Suffix(Keyword):
    def __init__(self):
        super().__init__("suffix")

    def attach(self, tokens, guards, obligations, suffix):
        from py_compiler.syntax.sentence.syntax import expression
        suffix.append(expression(tokens))
