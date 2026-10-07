"""Owner for atomic syntax and its future execution contract."""
from py_compiler.syntax.keywords.keyword import Keyword


class Atomic(Keyword):
    def __init__(self):
        super().__init__("atomic")

    def matches(self, node):
        return bool(node.tokens) and node.tokens[0].text == self.name

    def lower(self, node, cfg, env, local_names):
        raise ValueError("atomic has no declared execution/lowering implementation")
