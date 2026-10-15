"""Arrays own bounded storage and element initialization/destruction."""
from dataclasses import dataclass, field
from py_compiler.syntax.generic.atom import Atom
from py_compiler.mlir.src.cfg import name, temporary_suffix


@dataclass(frozen=True)
class Array:
    name: str = 'array'
    family: str = 'array'

    def grammars(self):
        from py_compiler.syntax.generic.owned import WordGrammar
        return (WordGrammar(self, self.name, 'IDENTIFIER'),)

    def specialize(self, element):
        return ArrayType(element)

    def specialize_arguments(self, arguments, context):
        if len(arguments) not in (1, 2):
            raise ValueError('array[T,N] requires an element type and optional fixed count')
        count = int(arguments[1]) if len(arguments) == 2 else None
        if count is not None and count < 0:
            raise ValueError('array count must be nonnegative')
        return ArrayType(context.type(arguments[0]), count=count)

    def construct_expression(self, argument, arguments, cfg, env, span, expected=None):
        from py_compiler.syntax.grammar.expression import type_spelling
        from py_compiler.syntax.generic.owned import select
        from py_compiler.syntax.function.allocate import resolve_plan
        terms = argument.operands if argument and argument.kind == 'template_arguments' else ()
        if len(terms) != 2 or arguments:
            raise ValueError('use array[T, N]() with an element type and element count')
        element = cfg.context.type(type_spelling(terms[0]))
        result = ArrayType(element)
        if isinstance(expected, ArrayType) and expected.element == element:
            result = expected
        count = select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, (terms[1],), env)
        if result.count is not None:
            cfg.effect('array-extent', (count,), ArrayExtent(result.count), span)
        plan = resolve_plan(result, {'location': 'heap'}, cfg.context)
        operation = record_operation(result, 'allocate', cfg).with_plan(plan) if result.aggregate else MemoryOperation('allocate', plan)
        return cfg.emit('allocate-array', result, (count,), operation, span)


@dataclass(frozen=True)
class ArrayExtent:
    count: int

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), None, ('copy',), (), self)

    def render_operation(self, operation):
        tag = temporary_suffix(operation)
        return [f'%extent_{tag} = arith.constant {self.count} : index',
                f'%extent_ok_{tag} = arith.cmpi eq, {name(operation.operands[0])}, %extent_{tag} : index',
                f'cf.assert %extent_ok_{tag}, "array element count does not match annotation"']


@dataclass(frozen=True)
class ArrayType:
    element: object
    count: int | None = None
    location: str = field(default='heap', compare=False)
    family: str = 'array'
    bits: int = 0
    signed: bool = False
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    def grammars(self):
        return ()

    def __post_init__(self):
        if not self.aggregate and not (self.element.mlir.startswith('i') and self.element.mlir[1:].isdigit() or self.element.mlir in ('f16', 'bf16', 'f32', 'f64')):
            raise ValueError('array element requires a supported scalar memory representation')

    @property
    def aggregate(self):
        return self.element.family in ('record', 'string', 'array', 'union', 'dynamic', 'shared')

    @property
    def name(self):
        suffix = '' if self.count is None else f',{self.count}'
        return f'{self.family}[{self.element.name}{suffix}]'

    @property
    def mlir(self):
        return 'memref<?xi8>' if self.aggregate else f'memref<?x{self.element.mlir}>'

    def require_context(self, cfg):
        # Access is bounded by this array's memref descriptor.
        pass

    def element_place(self, receiver, index, cfg, env, span):
        self.require_context(cfg)
        from py_compiler.syntax.generic.owned import select
        index = select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, (index,), env)
        return RecordElementPlace(receiver, index, self.element) if self.aggregate else ElementPlace(receiver, index, self.element)

    def copy_value(self, value, cfg, span):
        from dataclasses import replace
        if self.aggregate:
            return copy_records(self, value, cfg, span)
        return cfg.emit('memory-copy', replace(self, location='heap'), (value,), MemoryOperation('copy'), span)

    def release(self, value, cfg, span):
        if self.aggregate:
            record_loop(value, cfg, span, lambda place: place.destroy(cfg, span))
        if value.identity not in cfg.stack_values and value.type.location != 'stack':
            cfg.effect('deallocate', (value,), MemoryOperation('release'), span)


@dataclass(frozen=True)
class ElementPlace:
    receiver: object
    index: object
    type: object

    def read(self, cfg, span):
        return cfg.emit('memory-load', self.type, (self.receiver, self.index), MemoryOperation('load'), span)

    def write(self, cfg, value, span):
        cfg.effect('memory-store', (self.receiver, self.index, value), MemoryOperation('store'), span)


@dataclass(frozen=True)
class MemoryOperation:
    kind: str
    plan: object = None
    region: str = ''
    offset: int = 0

    def atom(self, operation):
        modes = {'allocate': ('copy',), 'load': ('view', 'copy'),
                 'store': ('borrow', 'copy', 'copy'), 'release': ('move',),
                 'copy': ('view',)}
        effects = {'allocate': ('allocation',), 'load': ('read',), 'store': ('mutation',),
                   'release': ('destruction',), 'copy': ('allocation', 'read')}
        return Atom(self, tuple(v.type for v in operation.operands),
                    operation.result.type if operation.result else None,
                    modes[self.kind], effects[self.kind], self)

    def render_operation(self, operation):
        values = operation.operands
        tag = temporary_suffix(operation)
        if self.kind == 'allocate':
            count = name(values[0])
            maximum = ((1 << (values[0].type.bits - 1)) - 1) // max(1, operation.result.type.element.bits // 8)
            allocator = 'memref.alloca' if self.plan and self.plan.location == 'stack' else 'memref.alloc'
            alignment = f' {{alignment = {self.plan.alignment} : i64}}' if self.plan else ''
            scalar = operation.result.type.element.mlir
            zero = '0.0' if scalar in ('f16', 'bf16', 'f32', 'f64') else '0'
            allocation = [f'{name(operation.result)} = {allocator}({count}){alignment} : {operation.result.type.mlir}']
            if self.region:
                allocation = [f'%shift_{tag} = arith.constant {self.offset} : index',
                              f'{name(operation.result)} = memref.view %{self.region}[%shift_{tag}][{count}] : memref<?xi8> to {operation.result.type.mlir}']
            return [f'%zero_{tag} = arith.constant 0 : index',
                    f'%valid_{tag} = arith.cmpi sge, {count}, %zero_{tag} : index',
                    f'cf.assert %valid_{tag}, "allocation count must be nonnegative"',
                    f'%maximum_{tag} = arith.constant {maximum} : index',
                    f'%fits_{tag} = arith.cmpi ule, {count}, %maximum_{tag} : index',
                    f'cf.assert %fits_{tag}, "allocation byte size overflow"',
                    *allocation,
                    f'%step_{tag} = arith.constant 1 : index',
                    f'%initial_{tag} = arith.constant {zero} : {scalar}',
                    f'scf.for %init_index_{tag} = %zero_{tag} to {count} step %step_{tag} {{',
                    f'  memref.store %initial_{tag}, {name(operation.result)}[%init_index_{tag}] : {operation.result.type.mlir}',
                    '}']
        memory, type_ = name(values[0]), values[0].type.mlir
        if self.kind == 'release':
            return [f'memref.dealloc {memory} : {type_}']
        if self.kind == 'copy':
            return [f'%zero_{tag} = arith.constant 0 : index',
                    f'%size_{tag} = memref.dim {memory}, %zero_{tag} : {type_}',
                    f'{name(operation.result)} = memref.alloc(%size_{tag}) : {type_}',
                    f'memref.copy {memory}, {name(operation.result)} : {type_} to {type_}']
        index = name(values[1])
        lines = [f'%zero_{tag} = arith.constant 0 : index',
                 f'%length_{tag} = memref.dim {memory}, %zero_{tag} : {type_}',
                 f'%valid_{tag} = arith.cmpi ult, {index}, %length_{tag} : index',
                 f'cf.assert %valid_{tag}, "memory index outside allocation"']
        if self.kind == 'load':
            lines.append(f'{name(operation.result)} = memref.load {memory}[{index}] : {type_}')
        else:
            lines.append(f'memref.store {name(values[2])}, {memory}[{index}] : {type_}')
        return lines


def record_storage(type_, context):
    from py_compiler.syntax.function.layout import storage_layout
    from py_compiler.syntax.generic.storage import stored_type
    element = type_.element
    layout = storage_layout(element, context)
    # Every owning slot has a flag, including a slot containing a union.
    return layout.alignment + layout.size, layout.alignment, stored_type(element, context)


def record_operation(type_, kind, cfg):
    stride, offset, layout = record_storage(type_, cfg.context)
    return RecordMemory(kind, stride, offset, layout, cfg.context.type('usize').bits)


def record_loop(value, cfg, span, action):
    from py_compiler.mir.cfg.cfg import Edge, Terminator
    from py_compiler.syntax.function.calls import literal_value
    from py_compiler.syntax.function.length import Length
    integer = cfg.context.type('int')
    zero = literal_value(0, integer, cfg, span)
    one = literal_value(1, integer, cfg, span)
    count = cfg.emit('length', integer, (value,), Length(record_storage(value.type, cfg.context)[0]), span)
    header, body, done = cfg.block((integer,)), cfg.block(), cfg.block()
    cfg.current.terminator = Terminator('jump', (Edge(header.identity, (zero,)),))
    cfg.current = header
    index = header.parameters[0]
    from py_compiler.syntax.generic.owned import select
    condition = select(integer, 'F.operator', '<').expand(cfg, index, count, span)
    cfg.current.terminator = Terminator('conditional', (Edge(body.identity), Edge(done.identity)), condition)
    cfg.current = body
    action(RecordElementPlace(value, index, value.type.element))
    following = select(integer, 'F.operator', '+').expand(cfg, index, one, span)
    cfg.current.terminator = Terminator('jump', (Edge(header.identity, (following,)),))
    cfg.current = done


def copy_records(type_, value, cfg, span):
    from dataclasses import replace
    from py_compiler.syntax.function.length import Length
    from py_compiler.syntax.function.allocate import resolve_plan
    count = cfg.emit('length', cfg.context.type('index'), (value,), Length(record_storage(type_, cfg.context)[0]), span)
    result_type = replace(type_, location='heap')
    plan = resolve_plan(result_type, {'location': 'heap'}, cfg.context)
    result = cfg.emit('allocate-array', result_type, (count,), record_operation(type_, 'allocate', cfg).with_plan(plan), span)
    def copy(place):
        place.when_initialized(cfg, span, lambda: RecordElementPlace(result, place.index, type_.element).write(cfg, place.read(cfg, span), span))
    record_loop(value, cfg, span, copy)
    cfg.value_dependencies[result.identity] = set(cfg.value_dependencies.get(value.identity, ()))
    return result


@dataclass(frozen=True)
class RecordElementPlace:
    receiver: object
    index: object
    type: object

    def operation(self, kind, cfg):
        return record_operation(self.receiver.type, kind, cfg)

    def read(self, cfg, span):
        return cfg.emit('record-element', self.type, (self.receiver, self.index), self.operation('load', cfg), span)

    def when_initialized(self, cfg, span, action):
        from py_compiler.mir.cfg.cfg import Edge, Terminator
        condition = cfg.emit('element-initialized', cfg.context.type('bool'), (self.receiver, self.index), self.operation('initialized', cfg), span)
        body, done = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('conditional', (Edge(body.identity), Edge(done.identity)), condition)
        cfg.current = body
        action()
        cfg.current.terminator = Terminator('jump', (Edge(done.identity),))
        cfg.current = done

    def destroy(self, cfg, span):
        from py_compiler.syntax.generic.storage import owns_storage, release_storage
        if owns_storage(self.type):
            self.when_initialized(cfg, span, lambda: release_storage(self.read(cfg, span), cfg, span))
        cfg.effect('element-clear', (self.receiver, self.index), self.operation('clear', cfg), span)

    def write(self, cfg, value, span):
        # Copy before replacement so a[i] = a[i] remains valid.
        from py_compiler.syntax.generic.storage import copy_storage
        copied = copy_storage(value, cfg, span)
        self.destroy(cfg, span)
        cfg.effect('element-store', (self.receiver, self.index, copied), self.operation('store', cfg), span)


@dataclass(frozen=True)
class RecordMemory:
    kind: str
    stride: int
    offset: int
    layout: str
    bits: int
    plan: object = None
    region: str = ''
    region_offset: int = 0

    def with_plan(self, plan):
        from dataclasses import replace
        return replace(self, plan=plan)

    def atom(self, operation):
        modes = tuple('borrow' if i == 0 and self.kind in ('store', 'clear') else 'view' for i in range(len(operation.operands)))
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type if operation.result else None,
                    modes, ('allocation',) if self.kind == 'allocate' else ('read', 'mutation'), self)

    def render_operation(self, operation):
        tag = temporary_suffix(operation)
        def v(label): return f'%record_{label}_{tag}'
        operands = operation.operands
        width = f'i{self.bits}'
        lines = [f'{v("zero")} = arith.constant 0 : index', f'{v("stride")} = arith.constant {self.stride} : index']
        if self.kind == 'allocate':
            count = name(operands[0])
            maximum = ((1 << (self.bits - 1)) - 1) // self.stride
            allocator = 'memref.alloca' if self.plan.location == 'stack' else 'memref.alloc'
            lines += [f'{v("limit")} = arith.constant {maximum} : index',
                      f'{v("valid")} = arith.cmpi ule, {count}, {v("limit")} : index',
                      f'cf.assert {v("valid")}, "allocation byte size overflow"',
                      f'{v("bytes")} = arith.muli {count}, {v("stride")} : index',
                      f'{name(operation.result)} = {allocator}({v("bytes")}) {{alignment = {self.plan.alignment} : i64}} : memref<?xi8>',
                      f'{v("empty")} = arith.constant 0 : i8',
                      f'scf.for {v("init")} = {v("zero")} to {v("bytes")} step {v("stride")} {{',
                      f'  memref.store {v("empty")}, {name(operation.result)}[{v("init")}] : memref<?xi8>', '}']
            if self.region:
                allocation = next(index for index, line in enumerate(lines)
                                  if line.startswith(name(operation.result) + ' ='))
                lines[allocation:allocation + 1] = [
                    f'{v("region_offset")} = arith.constant {self.region_offset} : index',
                    f'{name(operation.result)} = memref.view %{self.region}[{v("region_offset")}][{v("bytes")}] : memref<?xi8> to memref<?xi8>']
            return lines
        memory = name(operands[0])
        index = name(operands[1])
        if operands[1].type.mlir != 'index':
            lines.append(f'{v("index")} = arith.index_cast {index} : {operands[1].type.mlir} to index')
            index = v('index')
        lines += [f'{v("bytes")} = memref.dim {memory}, {v("zero")} : memref<?xi8>',
                  f'{v("length")} = arith.divui {v("bytes")}, {v("stride")} : index',
                  f'{v("valid")} = arith.cmpi ult, {index}, {v("length")} : index',
                  f'cf.assert {v("valid")}, "memory index outside allocation"',
                  f'{v("slot")} = arith.muli {index}, {v("stride")} : index',
                  f'{v("empty")} = arith.constant 0 : i8']
        if self.kind in ('initialized', 'load'):
            lines += [f'{v("flag")} = memref.load {memory}[{v("slot")}] : memref<?xi8>',
                      f'{name(operation.result) if self.kind == "initialized" else v("ready")} = arith.cmpi ne, {v("flag")}, {v("empty")} : i8']
            if self.kind == 'initialized': return lines
            lines.append(f'cf.assert {v("ready")}, "read of uninitialized array element"')
        if self.kind == 'clear':
            return lines + [f'memref.store {v("empty")}, {memory}[{v("slot")}] : memref<?xi8>']
        lines += [f'{v("offset")} = arith.constant {self.offset} : index',
                  f'{v("position")} = arith.addi {v("slot")}, {v("offset")} : index',
                  f'{v("base")} = memref.extract_aligned_pointer_as_index {memory} : memref<?xi8> -> index',
                  f'{v("address")} = arith.addi {v("base")}, {v("position")} : index',
                  f'{v("bits")} = arith.index_cast {v("address")} : index to {width}',
                  f'{name(operation.result) if self.kind == "load" and operation.result.type.family == "record" else v("pointer")} = llvm.inttoptr {v("bits")} : {width} to !llvm.ptr']
        from py_compiler.syntax.generic.storage import pack_value, unpack_value
        if self.kind == 'load' and operation.result.type.family != 'record':
            lines.append(f'{v("value")} = llvm.load {v("pointer")} : !llvm.ptr -> {self.layout}')
            unpack_value(lines, v('value'), operation.result.type, self.layout, name(operation.result))
        if self.kind == 'store':
            stored = pack_value(lines, name(operands[2]), operands[2].type, self.layout, v('value'))
            lines += [f'llvm.store {stored}, {v("pointer")} : {self.layout}, !llvm.ptr',
                      f'{v("full")} = arith.constant 1 : i8',
                      f'memref.store {v("full")}, {memory}[{v("slot")}] : memref<?xi8>']
        return lines


import unittest


class ArrayTypeTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('integer-memory.sev', source, Syntax())

    def ir(self, source):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return render(lower(result.program))

    def test_integer_array_pointer_and_explicit_release(self):
        text = self.ir('def work() -> int:\n    unsafe:\n        a: array[int] = allocate[int,location=memory.heap](128)\n        a[0] = 42\n        p: pointer[int] = view a\n        value = p[0]\n        deallocate[int](a)\n        return value\n')
        for operation in ('memref.alloc', 'memref.load', 'memref.store', 'memref.dealloc', 'memref<?xi64>'):
            self.assertIn(operation, text)
        self.assertEqual(text.count('memref.dealloc'), 1)

    def test_safe_constructor_owns_runtime_sized_storage(self):
        text = self.ir('def make(n: int) -> array[int]:\n    values = array[int,n]()\n    values[0] = 7\n    return values\ndef work() -> int:\n    values = make(3)\n    return values[0]\n')
        self.assertIn('memref.alloc', text)
        self.assertIn('memref.dealloc', text)
        self.assertIn('allocation count must be nonnegative', text)
        self.assertIn('allocation byte size overflow', text)

    def test_safe_constructor_supports_bare_and_fixed_extent_forms(self):
        text = self.ir('def work() -> int:\n    empty = array[int,0]\n    values: array[int,3] = array[int,3]()\n    return len(empty) + values[0]\n')
        self.assertIn('array element count does not match annotation', text)

    def test_borrow_parameter_mutates_and_default_view_reads(self):
        text = self.ir('def write(a: borrow array[int]):\n    a[0] = 9\ndef read(a: array[int]) -> int:\n    return a[0]\ndef work() -> int:\n    unsafe:\n        a = allocate[int,location=memory.heap](1)\n        write(a)\n        return read(a)\n')
        self.assertIn('memref.store', text)
        self.assertIn('memref.load', text)
        self.assertIn('memory index outside allocation', text)
        self.assertEqual(text.count('memref.dealloc'), 1)

    def test_copy_parameter_has_independent_storage(self):
        text = self.ir('def write(a: copy array[int]):\n    unsafe:\n        a[0] = 9\ndef work():\n    unsafe:\n        a = allocate[int,location=memory.heap](1)\n        a[0] = 4\n        write(a)\n')
        self.assertIn('memref.copy', text)
        self.assertEqual(text.count('memref.dealloc'), 2)

    def test_invalid_access_modes_report_ownership_diagnostics(self):
        cases = (
            ('def work(a: array[int]):\n    unsafe:\n        a[0] = 1\n', 'requires an owner or borrow'),
            ('def work(a: array[int]):\n    unsafe:\n        deallocate(a)\n', 'cannot deallocate a view'),
            ('def work():\n    unsafe:\n        a = allocate[int,location=memory.heap](1)\n        deallocate(a)\n        deallocate(a)\n', 'moved or dropped'),
        )
        for source, message in cases:
            with self.subTest(source=source):
                result = self.compile(source)
                self.assertTrue(result.diagnostics)
                self.assertIn(message, str(result.diagnostics[0]))

    def test_dynamic_index_and_element_count_use_mlir_index(self):
        text = self.ir('def work(n: int, i: int) -> int:\n    unsafe:\n        a = allocate[int,location=memory.heap](n)\n        a[i] = 5\n        return a[i]\n')
        self.assertIn('arith.index_cast', text)
        self.assertIn('memory index outside allocation', text)
        self.assertIn('allocation byte size overflow', text)


    def test_array_catalog_has_a_type_family(self):
        from py_compiler.syntax.prelude import type_definitions
        self.assertEqual(type_definitions(64)['array'].family, 'array')

    def test_safe_allocation_initializes_elements(self):
        text = self.ir('def work() -> int:\n    unsafe:\n        a = allocate[int,location=memory.heap]()\n        return a[0]\n')
        self.assertIn('scf.for', text)
        self.assertIn('memref.store', text)
        self.assertEqual(text.count('memref.dealloc'), 1)

    def test_typed_pointer_cannot_take_array_ownership(self):
        result = self.compile('def work() -> int:\n    unsafe:\n        a = allocate[int,location=memory.heap]()\n        p: pointer[int] = move a\n        p[0] = 9\n        return p[0]\n')
        self.assertIn('cannot transfer array allocation ownership', str(result.diagnostics))
        result = self.compile('def work():\n    unsafe:\n        a = allocate[int,location=memory.heap]()\n        p: pointer[int] = move a\n        deallocate(a)\n')
        self.assertTrue(result.diagnostics)

    def test_native_aggregate_storage_copies_and_destroys_initialized_slots(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.testing import test_package
        source = '''union Value:
    string
    array[int]
    int
class Text:
    value: string

def boxed() -> Value:
    return "escaped"

test "string slots survive replacement and copying":
    values = array[string,3]()
    values[1] = "first"
    saved = copy values
    values[1] = "second"
    drop values
    assert(len(saved[1]) == 5)
    assert(saved[1][0] == u8(102))

test "nested arrays have independent owners":
    values = array[array[int],2]()
    inner = array[int,1]()
    inner[0] = 42
    values[0] = inner
    inner[0] = 7
    saved = copy values
    drop inner
    drop values
    assert(saved[0][0] == 42)

test "class string fields own their storage":
    text = Text("hello")
    texts = array[Text,2]()
    texts[0] = text
    copy_text = copy texts[0]
    drop texts
    assert(len(copy_text.value) == 5)

test "unions own and destroy only their active variant":
    values = array[Value,3]()
    text: Value = "hello"
    numbers = array[int,2]()
    numbers[0] = 42
    wrapped: Value = numbers
    values[0] = text
    values[1] = wrapped
    saved = copy values
    values[0] = boxed()
    drop values
    drop saved
    assert(numbers[0] == 42)
'''
        with tempfile.TemporaryDirectory(prefix='sev-aggregate-storage-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'aggregate-storage', 'version': '0.1.0'}}))
            (root / 'src/lib.sev').write_text(source)
            result = test_package(root)
            self.assertTrue(result['success'], str(result))

    def test_native_borrowed_record_slots_track_source_lifetimes(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.testing import test_package
        source = '''class Handle:
    value: pointer[int]

def handle_for(source: array[int]) -> Handle:
    unsafe:
        pointer = pointer[int](source)
        handle = Handle(pointer)
        return handle

test "scope destruction releases borrowers before later declared sources":
    unsafe:
        slots = array[Handle,1]()
        source = array[int,1]()
        source[0] = 19
        slots[0] = handle_for(source)
        assert(slots[0].value[0] == 19)

test "stored borrowed fields retain sources through independent copies":
    unsafe:
        source = array[int,1]()
        source[0] = 42
        pointer = pointer[int](source)
        handle = Handle(pointer)
        slots = array[Handle,2]()
        slots[0] = handle
        saved = copy slots
        drop slots
        assert(saved[0].value[0] == 42)
        drop saved[0]
        drop handle
        drop source

test "replacing a slot releases its previous source dependency":
    unsafe:
        first = array[int,1]()
        second = array[int,1]()
        second[0] = 7
        slots = array[Handle,1]()
        p1 = pointer[int](first)
        handle1 = Handle(p1)
        slots[0] = handle1
        p2 = pointer[int](second)
        handle2 = Handle(p2)
        slots[0] = handle2
        drop handle1
        drop handle2
        drop first
        assert(slots[0].value[0] == 7)
        drop slots
        drop second

test with compiler "live slot prevents its borrowed source destruction":
    reject "borrow is live":
        class Handle:
            value: pointer[int]
        def invalid():
            unsafe:
                source = array[int,1]()
                slots = array[Handle,1]()
                if true:
                    pointer = pointer[int](source)
                    handle = Handle(pointer)
                    slots[0] = handle
                drop source
                assert(slots[0].value[0] == 0)

test with compiler "copied slots preserve dependencies":
    reject "borrow is live":
        class Handle:
            value: pointer[int]
        def invalid():
            unsafe:
                source = array[int,1]()
                slots = array[Handle,1]()
                if true:
                    pointer = pointer[int](source)
                    handle = Handle(pointer)
                    slots[0] = handle
                saved = copy slots
                drop slots
                drop source
                assert(saved[0].value[0] == 0)

test with compiler "an outer array cannot retain a scoped source":
    reject "borrow":
        class Handle:
            value: pointer[int]
        def invalid():
            unsafe:
                slots = array[Handle,1]()
                if true:
                    source = array[int,1]()
                    pointer = pointer[int](source)
                    handle = Handle(pointer)
                    slots[0] = handle
'''
        with tempfile.TemporaryDirectory(prefix='sev-borrowed-slots-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'borrowed-slots', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_borrowed_slot_effects_cross_calls_and_loop_backedges(self):
        source = '''class Handle:
    value: pointer[int]
def put(slots: borrow array[Handle], source: array[int]):
    unsafe:
        pointer = pointer[int](source)
        handle = Handle(pointer)
        slots[0] = handle
'''
        valid = source + '''def work():
    slots = array[Handle,1]()
    source = array[int,1]()
    put(slots, source)
    drop slots
    drop source
'''
        result = self.compile(valid)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        invalid = valid.replace('    drop slots\n    drop source', '    drop source\n    drop slots')
        self.assertIn('borrow', str(self.compile(invalid).diagnostics))
        loop = source + '''def work():
    slots = array[Handle,1]()
    source = array[int,1]()
    position = 0
    while position < 2:
        drop source
        source = array[int,1]()
        put(slots, source)
        position += 1
'''
        result = self.compile(loop)
        self.assertTrue(result.diagnostics)
        self.assertIn('control-flow path', str(result.diagnostics))

        copied = source + '''def work():
    slots = array[Handle,1]()
    source = array[int,1]()
    position = 0
    while position < 2:
        put(slots, source)
        position += 1
    saved = copy slots
    drop slots
    drop source
'''
        result = self.compile(copied)
        self.assertTrue(result.diagnostics)
        self.assertIn('control-flow path', str(result.diagnostics))

    def test_field_updates_in_distinct_elements_preserve_both_sources(self):
        source = '''class Handle:
    value: pointer[int]
def invalid():
    unsafe:
        first = array[int,1]()
        second = array[int,1]()
        third = array[int,1]()
        fourth = array[int,1]()
        slots = array[Handle,2]()
        slots[0] = Handle(pointer[int](first))
        slots[1] = Handle(pointer[int](second))
        slots[0].value = pointer[int](third)
        slots[1].value = pointer[int](fourth)
        drop third
'''
        result = self.compile(source)
        self.assertTrue(result.diagnostics)
        self.assertIn('borrow', str(result.diagnostics))
