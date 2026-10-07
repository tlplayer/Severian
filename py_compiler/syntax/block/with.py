"""with owns its block header; execution requires its own lowering provider."""
from py_compiler.syntax.generic.block import BlockProvider


class With(BlockProvider):
    spelling = "with"

    def declare(self, node, context):
        raise ValueError("with block has no declared execution/lowering implementation")
