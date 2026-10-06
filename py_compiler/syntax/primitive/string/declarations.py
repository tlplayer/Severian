from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.primitive.literal import QuotedLiteral
from py_compiler.syntax.symbol.forms import decode_quoted
from py_compiler.syntax.collection.collection import Collection


class String(Collection, Primitive):
    def grammars(self):
        return (*super().grammars(), QuotedLiteral(self, 'STRING'))

    def render_constant(self, value, symbol):
        import json
        data = list(value.encode('utf-8'))
        storage, name = f'memref<{len(data)}xi8>', symbol + '_bytes'
        return [f'memref.global "private" constant @{name} : {storage} = dense<{json.dumps(data)}>'], [
            f'%storage = memref.get_global @{name} : {storage}',
            f'%value = memref.cast %storage : {storage} to {self.mlir}']

    def decode(self, spelling):
        return decode_quoted(spelling)


STRING = String('string', 'string', 'memref<?xi8>')
