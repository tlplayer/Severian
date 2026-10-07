"""extend owns its block header; execution requires its own lowering provider."""
from py_compiler.syntax.generic.block import BlockProvider


class Extend(BlockProvider):
    spelling = "extend"

    def declare(self, node, context):
        raise ValueError("extend block has no declared execution/lowering implementation")
