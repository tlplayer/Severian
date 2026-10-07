"""Compiler rejection fixture; owns the expected-failure contract."""
from py_compiler.syntax.block.accept import Accept


class Reject(Accept):
    spelling = "reject"
    expected_failure = True
