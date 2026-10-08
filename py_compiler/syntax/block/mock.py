"""Test-only mock block; execution semantics are not implemented yet."""
from py_compiler.syntax.generic.block import BlockProvider


class Mock(BlockProvider):
    spelling = 'mock'
    required_scope = 'test'
