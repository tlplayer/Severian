"""operator owns its block header; execution requires its own lowering provider."""
from py_compiler.syntax.block.function import Function


class Operator(Function):
    spelling = "operator"

    def parse_header(self, items):
        header = super().parse_header(items)
        if not header:
            raise ValueError('operator requires an operation and parameters')
        name = '__operator_' + '_'.join(str(ord(c)) for c in header[0].text)
        return (header[0].with_text(name, kind='IDENTIFIER'), *header[1:])

    def register(self, node, context, receiver=None):
        entry = super().register(node, context, receiver)
        entry.receiver_ownership = entry.receiver_ownership or 'view'
        return entry
