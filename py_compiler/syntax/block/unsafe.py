"""unsafe owns its block header; execution requires its own lowering provider."""
from py_compiler.syntax.generic.block import BlockProvider


class Unsafe(BlockProvider):
    spelling = "unsafe"

    def declare(self, node, context):
        raise ValueError("unsafe block has no declared execution/lowering implementation")
