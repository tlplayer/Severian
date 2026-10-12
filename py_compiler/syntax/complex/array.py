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
        if self.element.family != 'record' and not (self.element.mlir.startswith('i') and self.element.mlir[1:].isdigit() or self.element.mlir in ('f16', 'bf16', 'f32', 'f64')):
            raise ValueError('array element requires a supported scalar memory representation')

    @property
    def name(self):
        suffix = '' if self.count is None else f',{self.count}'
        return f'{self.family}[{self.element.name}{suffix}]'

    @property
    def mlir(self):
        return 'memref<?xi8>' if self.element.family == 'record' else f'memref<?x{self.element.mlir}>'

    def require_context(self, cfg):
        # Access is bounded by this array's memref descriptor.
        pass

    def element_place(self, receiver, index, cfg, env, span):
        self.require_context(cfg)
        from py_compiler.syntax.generic.owned import select
        index = select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, (index,), env)
        return RecordElementPlace(receiver, index, self.element) if self.element.family == 'record' else ElementPlace(receiver, index, self.element)

    def copy_value(self, value, cfg, span):
        from dataclasses import replace
        if self.element.family == 'record':
            return copy_records(self, value, cfg, span)
        return cfg.emit('memory-copy', replace(self, location='heap'), (value,), MemoryOperation('copy'), span)

    def release(self, value, cfg, span):
        if self.element.family == 'record':
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
    from py_compiler.syntax.function.layout import layout_of
    element = type_.element
    def check_fields(record):
        for _, name in record.declaration.fields:
            member = context.type(name)
            if member.family in ('pointer', 'trait', 'dynamic'):
                raise ValueError('record array borrowed fields require element lifetime contracts')
            if member.family == 'record':
                check_fields(member)
    check_fields(element)
    layout = layout_of(element, context)
    # One initialized flag, followed by an aligned inline record.
    return layout.alignment + layout.size, layout.alignment, context.provider_by_name[element.name].layout(element.declaration, context)


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
        release = getattr(self.type, 'release', None)
        if release:
            def destroy():
                value = self.read(cfg, span)
                cfg.stack_values.discard(value.identity)
                release(value, cfg, span)
            self.when_initialized(cfg, span, destroy)
        cfg.effect('element-clear', (self.receiver, self.index), self.operation('clear', cfg), span)

    def write(self, cfg, value, span):
        # Copy before replacement so a[i] = a[i] remains valid.
        copied = self.type.copy_value(value, cfg, span)
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
                  f'{name(operation.result) if self.kind == "load" else v("pointer")} = llvm.inttoptr {v("bits")} : {width} to !llvm.ptr']
        if self.kind == 'store':
            lines += [f'{v("value")} = llvm.load {name(operands[2])} : !llvm.ptr -> {self.layout}',
                      f'llvm.store {v("value")}, {v("pointer")} : {self.layout}, !llvm.ptr',
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
        text = self.ir('def work() -> int:\n    a = allocate[int,location=memory.heap]()\n    return a[0]\n')
        self.assertIn('scf.for', text)
        self.assertIn('memref.store', text)
        self.assertEqual(text.count('memref.dealloc'), 1)

    def test_typed_pointer_can_take_array_ownership(self):
        text = self.ir('def work() -> int:\n    a = allocate[int,location=memory.heap]()\n    p: pointer[int] = move a\n    p[0] = 9\n    return p[0]\n')
        self.assertEqual(text.count('memref.dealloc'), 1)
        result = self.compile('def work():\n    a = allocate[int,location=memory.heap]()\n    p: pointer[int] = move a\n    deallocate(a)\n')
        self.assertTrue(result.diagnostics)
