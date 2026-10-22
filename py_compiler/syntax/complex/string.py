from dataclasses import dataclass
from py_compiler.syntax.generic.constructor import Constructor
from py_compiler.syntax.primitive.literal import QuotedLiteral
from py_compiler.syntax.symbol.forms import decode_quoted
from py_compiler.syntax.complex.collection import Collection
from py_compiler.syntax.complex.array import ElementPlace
from py_compiler.syntax.generic.owned import OwnedGrammar, Capture


class StringMembership(OwnedGrammar):
    spelling = 'in'
    precedence = 3
    associativity = 'left'
    comparison = True

    def __init__(self, owner):
        super().__init__(owner, 'F.operator', (Capture('needle', owner), 'in', Capture('value', owner)))

    def recognize(self, window):
        from py_compiler.syntax.generic.grammar import Match
        if len(window.tokens) == 1 and window.tokens[0].text == 'in':
            return Match(self, window, window.end, {})
        return None

    def lower_expression(self, cfg, node, env):
        needle = cfg.expr(node.operands[0], env, self.owner)
        if needle.type != self.owner:
            raise ValueError('string membership requires a string needle')
        return self.owner.binary_operation(node, needle, cfg, env)


class StringBytePlace(ElementPlace):
    def write(self, cfg, value, span):
        raise ValueError('string literal storage is read-only')


@dataclass(frozen=True)
class String(Collection):
    name: str = "string"
    family: str = "string"
    mlir: str = "memref<?xi8>"
    no_result: bool = False
    pointer_bits: int = 64

    @property
    def global_storage_type(self):
        from py_compiler.syntax.generic.storage import descriptor_type
        return descriptor_type(self.pointer_bits)

    def module_storage(self, symbol):
        storage = self.global_storage_type
        return f'llvm.mlir.global external @{symbol}() : {storage} {{\n    %zero = llvm.mlir.zero : {storage}\n    llvm.return %zero : {storage}\n  }}'

    def construct_expression(self, argument, arguments, cfg, env, span, expected=None):
        if argument is not None or len(arguments) != 1:
            raise ValueError('string requires one UTF-8 byte array or string')
        value = cfg.expr(arguments[0], env)
        if value.type.family == 'type':
            from py_compiler.syntax.function.calls import literal_value
            return literal_value(value.type.declaration.name, self, cfg, span)
        if value.type.family == 'pointer':
            from py_compiler.syntax.primitive.pointer.operations import ArrayAddress
            from py_compiler.syntax.primitive.numeric.conversion import ScalarConversion
            from py_compiler.syntax.grammar.expression import Expression
            raw = cfg.context.type('pointer')
            if value.type.mlir.startswith('memref'):
                value = cfg.emit('pointer-address', raw, (value,), ArrayAddress(raw), span)
            address_type = cfg.context.type('u64')
            address = cfg.emit('pointer-address-bits', address_type, (value,), ScalarConversion(value.type, address_type), span)
            return self.library_call('_pointer', (Expression('capture', arguments[0].token, (StringConversion(address),)),), cfg, env, span)
        if value.type.family != 'string' and not (value.type.family in ('array', 'tensor', 'tuple') and value.type.element.name == 'u8' and getattr(value.type, 'rank', 1) == 1):
            from py_compiler.syntax.grammar.expression import Expression
            return self.library_call('_convert', (Expression('capture', arguments[0].token, (StringConversion(value),)),), cfg, env, span)
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

    def slice_value(self, receiver, bounds, cfg, env, span, owner=None, *, closed=(True, True)):
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.frontend.lexer.lexer import lex
        from py_compiler.frontend.source.source import SourceFile
        start, stop, step = bounds
        if step is not None:
            raise ValueError('string slicing currently requires unit stride')
        token = lex(SourceFile(cfg.source.path + ':slice', 'slice'), cfg.syntax)[0]
        integer = cfg.context.type('int')
        begin = cfg.expr(start, env, integer) if start else literal_value(0, integer, cfg, span)
        end = cfg.expr(stop, env, integer) if stop else self.length(receiver, cfg, span)
        arguments = tuple(Expression('capture', token, (ArgumentValue(v, owner if i == 0 else None),))
                          for i, v in enumerate((receiver, begin, end,
                              literal_value(closed[0] or start is None, cfg.context.type('bool'), cfg, span),
                              literal_value(closed[1] and stop is not None, cfg.context.type('bool'), cfg, span))))
        return self.library_call('_slice', arguments, cfg, env, span)

    def release(self, value, cfg, span):
        from py_compiler.syntax.complex.array import MemoryOperation
        cfg.effect('deallocate', (value,), MemoryOperation('release'), span)

    def library_call(self, method, arguments, cfg, env, span):
        from py_compiler.syntax.function.calls import select, invoke
        entries = cfg.context.lookup_functions(method, cfg.declaration_scope)
        if not entries:
            from py_compiler.syntax.prelude import require_library
            require_library(cfg.context, 'core.string')
        for alias in cfg.context.package_imports:
            package = cfg.context.c_packages[alias]
            if package['package']['name'] == 'core.string':
                entries = cfg.context.lookup_functions(alias + '.' + method, cfg.declaration_scope)
                break
        if not entries:
            raise ValueError('string operation requires an explicit import of core.string')
        existing = set(env)
        entry, values = select(entries, arguments, cfg, env)
        result = invoke(entry, values, cfg, span)
        # These library operations return independent values. Argument literals
        # must not become branch-local bindings carried by short-circuit joins.
        if not getattr(entry.body, 'result_sources', ()):
            for key in set(env) - existing:
                if key.startswith('<owned argument '):
                    binding = env.pop(key)
                    cfg.cleanup_memory((binding,), span)
                    cfg.local_names.discard(key)
        return result

    def call_method(self, receiver, method, arguments, cfg, env, span, owner=None):
        if method in ('concat', 'contains', 'utf', 'upper', 'lower', 'equal', 'slice', 'replace', 'compare'):
            from py_compiler.syntax.grammar.expression import Expression
            from importlib import import_module
            from py_compiler.frontend.lexer.lexer import lex
            from py_compiler.frontend.source.source import SourceFile
            token = lex(SourceFile(cfg.source.path + ':string-method', method), cfg.syntax)[0]
            capture = Expression('capture', token, (import_module('py_compiler.syntax.block.for').FrozenCapture(receiver, owner),))
            return self.library_call('_' + method, (capture, *arguments), cfg, env, span)
        if method != 'suffix' or len(arguments) != 1:
            raise ValueError('unknown string method: ' + method)
        suffix = cfg.expr(arguments[0], env, self)
        result = cfg.emit('string-suffix', cfg.context.type('bool'), (receiver, suffix), Suffix(), span)
        if arguments[0].kind == 'literal':
            self.release(suffix, cfg, span)
        return result

    def binary_operation(self, node, left, cfg, env):
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.generic.storage import root_binding
        from py_compiler.syntax.function.calls import lower_expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        owner = root_binding(node.operands[0], cfg, env)
        frozen = Expression('capture', node.token, (ArgumentValue(left, owner),))
        method = {'+': 'concat', '==': 'equal', '!=': 'equal', 'in': 'contains',
                  '<': 'compare', '<=': 'compare', '>': 'compare', '>=': 'compare'}.get(node.token.text)
        if method is None:
            raise ValueError('unsupported string operation: ' + node.token.text)
        receiver, argument = (node.operands[1], frozen) if node.token.text == 'in' else (frozen, node.operands[1])
        if node.token.text == '+':
            argument = Expression('call', node.token, (Expression('name', node.token.with_text('string', kind='IDENTIFIER')), argument))
        call = Expression('call', node.token, (Expression('member', node.token.with_text(method, kind='IDENTIFIER'), (receiver,)), argument))
        result = lower_expression(call, cfg, env)
        if method == 'compare':
            from py_compiler.syntax.function.calls import literal_value
            from py_compiler.syntax.generic.owned import select
            zero = literal_value(0, result.type, cfg, node.token.span)
            return select(result.type, 'F.operator', node.token.text).expand(cfg, result, zero, node.token.span)
        if node.token.text == '!=':
            from py_compiler.syntax.generic.owned import select
            return select(result.type, 'F.unary', 'not').expand(cfg, result, node.token.span)
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
        from py_compiler.syntax.complex.collection import CollectionCondition
        return (Constructor(self), QuotedLiteral(self, 'STRING'), StringMembership(self), CollectionCondition(self))

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


@dataclass(frozen=True)
class StringConversion:
    value: object

    def type(self, cfg, expected=None):
        return self.value.type

    def binding(self, cfg):
        return None

    def lower(self, cfg, expected=None):
        return self.value
