"""Type-erased owning boxes with a concrete destructor and reference count."""
from dataclasses import dataclass
from hashlib import sha256
from py_compiler.syntax.generic.atom import Atom


@dataclass(frozen=True)
class DynamicType:
    name: str = 'dynamic'
    family: str = 'dynamic'
    mlir: str = '!llvm.struct<(ptr, ptr)>'
    bits: int = 64
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    def require_context(self, cfg):
        pass

    def grammars(self):
        from py_compiler.syntax.generic.owned import WordGrammar
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)

    def from_value(self, value, cfg, span):
        if value.type == self:
            return value
        from py_compiler.syntax.complex.array import ArrayType, MemoryOperation, ElementPlace, RecordElementPlace, record_operation
        from py_compiler.syntax.complex.shared import Shared, SharedOperation
        from py_compiler.syntax.function.allocate import resolve_plan
        from py_compiler.syntax.function.calls import literal_value
        concrete = value.type
        backing = ArrayType(concrete)
        plan = resolve_plan(backing, {'location': 'heap'}, cfg.context)
        count = literal_value(1, cfg.context.type('index'), cfg, span)
        zero = literal_value(0, count.type, cfg, span)
        allocation = record_operation(backing, 'allocate', cfg).with_plan(plan) if backing.aggregate else MemoryOperation('allocate', plan)
        storage = cfg.emit('allocate-array', backing, (count,), allocation, span)
        place = RecordElementPlace(storage, zero, concrete) if backing.aggregate else ElementPlace(storage, zero, concrete)
        place.write(cfg, value, span)
        from py_compiler.syntax.generic.storage import owns_storage
        if (owns_storage(concrete) and value.identity in cfg.owned_results
                and not cfg.value_origins.get(value.identity)):
            concrete.release(value, cfg, span)
        shared = Shared(backing, self.bits)
        handle = cfg.emit('share-create', shared, (storage,), SharedOperation('create', shared), span)
        symbol = '__sev_dynamic_drop_' + sha256((self.name + concrete.name + concrete.mlir + str(self.bits)).encode()).hexdigest()
        helpers = getattr(cfg.context, 'dynamic_helpers', None)
        if helpers is None:
            helpers = cfg.context.dynamic_helpers = {}
        if symbol not in helpers:
            helpers[symbol] = destructor(symbol, shared, self, cfg, span)
        operation = self.box_operation(concrete, symbol, helpers[symbol], cfg, span)
        result = cfg.emit('memory-copy', self, (handle,), operation, span)
        cfg.owned_results.add(result.identity)
        return result

    def box_operation(self, concrete, symbol, helper, cfg, span):
        return DynamicBox(concrete, self, symbol, helper)

    def copy_value(self, value, cfg, span):
        from py_compiler.syntax.complex.shared import Shared, SharedOperation
        from py_compiler.syntax.complex.array import ArrayType
        erased = Shared(ArrayType(cfg.context.type('u8')), self.bits)
        handle = cfg.emit('dynamic-handle', erased, (value,), DynamicOperation('handle'), span)
        cfg.emit('share-retain', erased, (handle,), SharedOperation('retain', erased), span)
        return cfg.emit('memory-copy', self, (value,), DynamicOperation('copy'), span)

    def release(self, value, cfg, span):
        cfg.effect('dynamic-release', (value,), DynamicOperation('release'), span)


def destructor(symbol, shared, type_, cfg, span):
    from py_compiler.hir.thir.src.checking import Checker
    from py_compiler.hir.thir.src.model import Edge, Terminator
    from py_compiler.syntax.complex.shared import SharedOperation
    builder = Checker(cfg.source, cfg.syntax, symbol, symbol, cfg.context.type('absent'))
    builder.context = cfg.context
    source = builder.value(type_)
    builder.current.parameters = (source,)
    handle = builder.emit('dynamic-handle', shared, (source,), DynamicOperation('handle'), span)
    last = builder.emit('share-release', cfg.context.type('bool'), (handle,), SharedOperation('release', shared), span)
    destroy, done = builder.block(), builder.block()
    builder.current.terminator = Terminator('conditional', (Edge(destroy.identity), Edge(done.identity)), last)
    builder.current = destroy
    payload = builder.emit('share-data', shared.backing, (handle,), SharedOperation('data', shared), span)
    shared.backing.release(payload, builder, span)
    builder.effect('counter-free', (handle,), SharedOperation('counter-free', shared), span)
    builder.effect('dynamic-free', (source,), DynamicOperation('free'), span)
    builder.current.terminator = Terminator('jump', (Edge(done.identity),))
    builder.current = done
    body = builder.finish()
    body.source_snapshot = None
    return body.freeze()


@dataclass(frozen=True)
class DynamicBox:
    concrete: object
    owner: object
    symbol: str
    helper: object

    def global_definitions(self):
        return (f'llvm.func @malloc(i{self.owner.bits}) -> !llvm.ptr',
                'llvm.func @free(!llvm.ptr)')

    def auxiliary_bodies(self):
        from py_compiler.mir.lowering import lower_body
        return tuple(lower_body(body) for body in self.generated_bodies())

    def generated_bodies(self):
        return (self.helper,)

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), self.owner, ('move',), ('allocation',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        value, result = operation.operands[0], operation.result
        tag, aggregate = str(result.identity), self.owner.mlir
        return [f'%dynamic_size_{tag} = arith.constant {10 * self.owner.bits // 8} : i{self.owner.bits}',
                f'%dynamic_payload_{tag} = llvm.call @malloc(%dynamic_size_{tag}) : (i{self.owner.bits}) -> !llvm.ptr',
                f'llvm.store {name(value)}, %dynamic_payload_{tag} : {value.type.mlir}, !llvm.ptr',
                f'%dynamic_function_{tag} = func.constant @{self.symbol} : ({aggregate}) -> ()',
                f'%dynamic_descriptor_{tag} = builtin.unrealized_conversion_cast %dynamic_function_{tag} : ({aggregate}) -> () to !llvm.ptr',
                f'%dynamic_empty_{tag} = llvm.mlir.undef : {aggregate}',
                f'%dynamic_tagged_{tag} = llvm.insertvalue %dynamic_descriptor_{tag}, %dynamic_empty_{tag}[0] : {aggregate}',
                f'{name(result)} = llvm.insertvalue %dynamic_payload_{tag}, %dynamic_tagged_{tag}[1] : {aggregate}']


@dataclass(frozen=True)
class DynamicOperation:
    kind: str

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type if operation.result else None,
                    ('move',) if self.kind in ('release', 'free') else ('view',), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        source = operation.operands[0]
        tag = temporary_suffix(operation)
        if self.kind == 'copy':
            return [f'%dynamic_tag_{tag} = llvm.extractvalue {name(source)}[0] : {source.type.mlir}',
                    f'{name(operation.result)} = llvm.insertvalue %dynamic_tag_{tag}, {name(source)}[0] : {source.type.mlir}']
        if self.kind == 'release':
            return [f'%dynamic_drop_{tag} = llvm.extractvalue {name(source)}[0] : {source.type.mlir}',
                    f'llvm.call %dynamic_drop_{tag}({name(source)}) : !llvm.ptr, ({source.type.mlir}) -> ()']
        lines = [f'%dynamic_pointer_{tag} = llvm.extractvalue {name(source)}[1] : {source.type.mlir}']
        if self.kind == 'handle':
            lines.append(f'{name(operation.result)} = llvm.load %dynamic_pointer_{tag} : !llvm.ptr -> {operation.result.type.mlir}')
        elif self.kind == 'free':
            lines.append(f'llvm.call @free(%dynamic_pointer_{tag}) : (!llvm.ptr) -> ()')
        return lines


import unittest


class DynamicTests(unittest.TestCase):
    def test_boxing_and_forwarding_retain_the_descriptor(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        source = 'def identity[T](value: T) -> T:\n    return value\ndef work():\n    original: dynamic = 42\n    result = identity(original)\n'
        result = compile_source('dynamic.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('__sev_dynamic_drop_', text)
        self.assertIn('!llvm.struct<(ptr, ptr)>', text)
        self.assertIn('llvm.call @malloc', text)

    def test_boxed_owner_can_escape(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('dynamic.sev', 'def bad() -> dynamic:\n    value: dynamic = 42\n    return value\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_c_dynamic_owners_arrays_and_union_payloads(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''union Value:
    string
    array[int]
class Buffer:
    values: array[int]

def make() -> dynamic:
    value: dynamic = "escaped"
    return value

def forward(value: dynamic) -> dynamic:
    return value

test "dynamic storage retains all concrete destructors":
    slots = array[dynamic,5]()
    number: dynamic = 42
    slots[0] = number
    text = make()
    slots[1] = text
    buffer = Buffer(array[int,2]())
    record: dynamic = buffer
    slots[2] = record
    payload: Value = "union"
    erased: dynamic = payload
    slots[3] = erased
    nested: dynamic = slots
    saved = copy nested
    drop slots
    drop nested
    if true:
        alias = forward(saved)
        copied = copy alias
        drop copied
    drop saved
'''
        with tempfile.TemporaryDirectory(prefix='sev-dynamic-storage-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'dynamic-storage', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)
