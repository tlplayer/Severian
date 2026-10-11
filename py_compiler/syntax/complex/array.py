"""Integer arrays own allocation, indexing and memref lowering."""
from dataclasses import dataclass, field
from py_compiler.syntax.generic.atom import Atom
from py_compiler.mlir.src.cfg import name


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
        if not (self.element.mlir.startswith('i') and self.element.mlir[1:].isdigit() or self.element.mlir in ('f16', 'bf16', 'f32', 'f64')):
            raise ValueError('array element requires a supported scalar memory representation')

    @property
    def name(self):
        suffix = '' if self.count is None else f',{self.count}'
        return f'{self.family}[{self.element.name}{suffix}]'

    @property
    def mlir(self):
        return f'memref<?x{self.element.mlir}>'

    def require_context(self, cfg):
        # Access is bounded by this array's memref descriptor.
        pass

    def element_place(self, receiver, index, cfg, env, span):
        self.require_context(cfg)
        from py_compiler.syntax.generic.owned import select
        index = select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, (index,), env)
        return ElementPlace(receiver, index, self.element)

    def copy_value(self, value, cfg, span):
        from dataclasses import replace
        return cfg.emit('memory-copy', replace(self, location='heap'), (value,), MemoryOperation('copy'), span)

    def release(self, value, cfg, span):
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
        tag = str(operation.result.identity) if operation.result else 'at_' + str(operation.span.start)
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
