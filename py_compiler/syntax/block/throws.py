"""Test-only expected-error block; execution semantics are not implemented yet."""
from py_compiler.syntax.generic.block import BlockProvider


class Throws(BlockProvider):
    spelling = 'throws'
    required_scope = 'test'
