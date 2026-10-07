"""match owns its block header; execution requires its own lowering provider."""
from py_compiler.syntax.generic.block import BlockProvider


class Match(BlockProvider):
    spelling = "match"

    def declare(self, node, context):
        raise ValueError("match block has no declared execution/lowering implementation")
