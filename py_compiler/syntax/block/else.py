from importlib import import_module
from py_compiler.syntax.generic.block import BlockProvider


class Else(import_module("py_compiler.syntax.block.if").Conditional):
    spelling = "else"
    continuation = True
    has_condition = False

    def parse_header(self, items):
        header = BlockProvider.parse_header(self, items)
        if header:
            raise ValueError("else does not take a condition")
        return header

    def attach(self, previous, parent):
        super().attach(previous, parent)
        if not previous.provider.has_condition:
            raise ValueError("else cannot follow else")
