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

    def construct_expression(self, argument, arguments, cfg, env, span, expected=None):
        if argument is not None or len(arguments) != 1:
            raise ValueError('string requires one UTF-8 byte array or string')
        value = cfg.expr(arguments[0], env)
        if value.type.family != 'string' and not (value.type.family == 'array' and value.type.element.name == 'u8'):
            raise ValueError('string requires UTF-8 byte storage')
        from py_compiler.syntax.complex.array import MemoryOperation
        result = cfg.emit('memory-copy', self, (value,), MemoryOperation('copy'), span)
        cfg.owned_results.add(result.identity)
        cfg.value_dependencies[result.identity] = set()
        return result

    def copy_value(self, value, cfg, span):
        from py_compiler.syntax.complex.array import MemoryOperation
        result = cfg.emit('memory-copy', self, (value,), MemoryOperation('copy'), span)
        cfg.owned_results.add(result.identity)
        cfg.value_dependencies[result.identity] = set()
        return result

    def release(self, value, cfg, span):
        from py_compiler.syntax.complex.array import MemoryOperation
        cfg.effect('deallocate', (value,), MemoryOperation('release'), span)

    def call_method(self, receiver, method, arguments, cfg, env, span, owner=None):
        if method != 'suffix' or len(arguments) != 1:
            raise ValueError('unknown string method: ' + method)
        suffix = cfg.expr(arguments[0], env, self)
        result = cfg.emit('string-suffix', cfg.context.type('bool'), (receiver, suffix), Suffix(), span)
        if arguments[0].kind == 'literal':
            self.release(suffix, cfg, span)
        return result

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
            f'%source = memref.cast %storage : {storage} to {self.mlir}',
            f'%count = arith.constant {len(data)} : index',
            f'%value = memref.alloc(%count) : {self.mlir}',
            f'memref.copy %source, %value : {self.mlir} to {self.mlir}']

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


@dataclass(frozen=True)
class Suffix:
    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type, ('view', 'view'), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        a, b = map(name, operation.operands)
        r = name(operation.result)
        p = '%suffix_' + str(operation.result.identity)
        return [f'{p}z = arith.constant 0 : index', f'{p}one = arith.constant 1 : index',
                f'{p}true = arith.constant true', f'{p}false = arith.constant false',
                f'{p}a = memref.dim {a}, {p}z : memref<?xi8>',
                f'{p}b = memref.dim {b}, {p}z : memref<?xi8>',
                f'{p}fits = arith.cmpi uge, {p}a, {p}b : index',
                f'{r} = scf.if {p}fits -> (i1) {{',
                f'  {p}start = arith.subi {p}a, {p}b : index',
                f'  {p}match = scf.for {p}i = {p}z to {p}b step {p}one iter_args({p}ok = {p}true) -> (i1) {{',
                f'    {p}at = arith.addi {p}start, {p}i : index',
                f'    {p}left = memref.load {a}[{p}at] : memref<?xi8>',
                f'    {p}right = memref.load {b}[{p}i] : memref<?xi8>',
                f'    {p}equal = arith.cmpi eq, {p}left, {p}right : i8',
                f'    {p}next = arith.andi {p}ok, {p}equal : i1',
                f'    scf.yield {p}next : i1', '  }', f'  scf.yield {p}match : i1',
                '} else {', f'  scf.yield {p}false : i1', '}']
