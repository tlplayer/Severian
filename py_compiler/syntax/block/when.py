"""Test-only condition block; execution semantics are not implemented yet."""
from py_compiler.syntax.generic.block import BlockProvider


class When(BlockProvider):
    spelling = 'when'
    required_scope = 'test'
