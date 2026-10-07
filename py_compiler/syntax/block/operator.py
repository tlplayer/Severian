"""operator owns its block header; execution requires its own lowering provider."""
from py_compiler.syntax.generic.block import BlockProvider


class Operator(BlockProvider):
    spelling = "operator"

    def declare(self, node, context):
        raise ValueError("operator block has no declared execution/lowering implementation")
