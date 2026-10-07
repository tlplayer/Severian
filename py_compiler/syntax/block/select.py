"""Select owns task-selection block syntax."""
from py_compiler.syntax.generic.block import BlockProvider


class Select(BlockProvider):
    spelling = 'select'

    def declare(self, node, context):
        raise ValueError('select has no declared task-selection/lowering implementation')
