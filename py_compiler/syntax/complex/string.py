from dataclasses import dataclass
from py_compiler.syntax.generic.constructor import Constructor
from py_compiler.syntax.primitive.literal import QuotedLiteral
from py_compiler.syntax.symbol.forms import decode_quoted
from py_compiler.syntax.complex.collection import Collection
from py_compiler.syntax.complex.array import ElementPlace


class StringBytePlace(ElementPlace):
    def write(self, cfg, value, span):
        raise ValueError('string literal storage is read-only')


@dataclass(frozen=True)
class String(Collection):
    name: str = "string"
    family: str = "string"
    mlir: str = "memref<?xi8>"
    no_result: bool = False

    def length(self, value, cfg, span):
        from py_compiler.syntax.function.length import Length
        return cfg.emit('length', cfg.context.type('int'), (value,), Length(), span)

    def element_place(self, receiver, index, cfg, env, span):
        from py_compiler.syntax.generic.owned import select
        index = select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, (index,), env)
        return StringBytePlace(receiver, index, cfg.context.type('u8'))

    def accept_literal(self, owner, value):
        if owner != self:
            raise ValueError(f"{owner.name} literal does not satisfy string")
        return value

    def grammars(self):
        return (Constructor(self), QuotedLiteral(self, 'STRING'))

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


import unittest


class StringTests(unittest.TestCase):
    def test_literal_byte_reads_and_length_are_read_only(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'def read(value: string) -> int:\n    return len(value) + int(value[0])\n'
        result = compile_source('read.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        result = compile_source('write.sev', 'def write(value: string):\n    value[0] = u8(0)\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertRegex(str(result.diagnostics), 'read-only|indexed storage contract')

    def test_string_is_registered_as_a_collection(self):
        from py_compiler.syntax.primitive.catalog import primitives
        from py_compiler.syntax.primitive.contract import Primitive
        from py_compiler.syntax.recognition import Syntax
        self.assertNotIn('string', primitives())
        self.assertIsInstance(Syntax().types['string'], Collection)
        self.assertNotIsInstance(STRING, Primitive)
        self.assertEqual(STRING.binding_ownership, 'view')

    def test_unicode_literal_keeps_utf8_storage(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('string.sev', 'text = "λ😀"\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertIn('memref<6xi8>', render(lower(result.program)))
