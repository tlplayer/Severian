"""Integer arrays own allocation, indexing and memref lowering."""
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ArrayType:
    element: object
    family: str = 'array'
    bits: int = 0
    signed: bool = False
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    def __post_init__(self):
        if self.element.family != 'integer':
            raise ValueError('array allocation currently requires an integer element type')

    @property
    def name(self):
        return f'{self.family}[{self.element.name}]'

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
        return cfg.emit('memory-copy', self, (value,), MemoryOperation('copy'), span)

    def release(self, value, cfg, span):
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
            return [f'%zero_{tag} = arith.constant 0 : index',
                    f'%valid_{tag} = arith.cmpi sge, {count}, %zero_{tag} : index',
                    f'cf.assert %valid_{tag}, "allocation count must be nonnegative"',
                    f'%maximum_{tag} = arith.constant {maximum} : index',
                    f'%fits_{tag} = arith.cmpi ule, {count}, %maximum_{tag} : index',
                    f'cf.assert %fits_{tag}, "allocation byte size overflow"',
                    f'{name(operation.result)} = memref.alloc({count}) : {operation.result.type.mlir}']
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
        text = self.ir('def work() -> int:\n    unsafe:\n        a: array[int] = allocate[int](128)\n        a[0] = 42\n        p: pointer[int] = view a\n        value = p[0]\n        deallocate[int](a)\n        return value\n')
        for operation in ('memref.alloc', 'llvm.load', 'memref.store', 'memref.dealloc', 'memref<?xi64>', '!llvm.ptr'):
            self.assertIn(operation, text)
        self.assertEqual(text.count('memref.dealloc'), 1)

    def test_borrow_parameter_mutates_and_default_view_reads(self):
        text = self.ir('def write(a: borrow array[int]):\n    a[0] = 9\ndef read(a: array[int]) -> int:\n    return a[0]\ndef work() -> int:\n    unsafe:\n        a = allocate[int](1)\n        write(a)\n        return read(a)\n')
        self.assertIn('memref.store', text)
        self.assertIn('memref.load', text)
        self.assertIn('memory index outside allocation', text)
        self.assertEqual(text.count('memref.dealloc'), 1)

    def test_copy_parameter_has_independent_storage(self):
        text = self.ir('def write(a: copy array[int]):\n    unsafe:\n        a[0] = 9\ndef work():\n    unsafe:\n        a = allocate[int](1)\n        a[0] = 4\n        write(a)\n')
        self.assertIn('memref.copy', text)
        self.assertEqual(text.count('memref.dealloc'), 2)

    def test_invalid_access_modes_report_ownership_diagnostics(self):
        cases = (
            ('def work(a: array[int]):\n    unsafe:\n        a[0] = 1\n', 'requires an owner or borrow'),
            ('def work(a: array[int]):\n    unsafe:\n        deallocate(a)\n', 'cannot deallocate a view'),
            ('def work():\n    unsafe:\n        a = allocate[int](1)\n        deallocate(a)\n        deallocate(a)\n', 'moved or dropped'),
        )
        for source, message in cases:
            with self.subTest(source=source):
                result = self.compile(source)
                self.assertTrue(result.diagnostics)
                self.assertIn(message, str(result.diagnostics[0]))

    def test_dynamic_index_and_element_count_use_mlir_index(self):
        text = self.ir('def work(n: int, i: int) -> int:\n    unsafe:\n        a = allocate[int](n)\n        a[i] = 5\n        return a[i]\n')
        self.assertIn('arith.index_cast', text)
        self.assertIn('memory index outside allocation', text)
        self.assertIn('allocation byte size overflow', text)


    def test_array_catalog_has_a_type_family(self):
        from py_compiler.syntax.prelude import type_definitions
        self.assertEqual(type_definitions(64)['array'].family, 'array')

    def test_allocation_still_requires_unsafe(self):
        result = self.compile('def work():\n    a = allocate[int](1)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('raw allocation requires an unsafe block', str(result.diagnostics[0]))

    def test_raw_pointer_cannot_take_or_release_array_ownership(self):
        for action, message in (
            ('p: pointer[int] = move a', 'cannot transfer array allocation ownership'),
            ('p: pointer[int] = view a\n        deallocate(p)', 'deallocate requires an owning array'),
        ):
            result = self.compile('def work():\n    unsafe:\n        a = allocate[int](1)\n        ' + action + '\n')
            self.assertTrue(result.diagnostics)
            self.assertIn(message, str(result.diagnostics[0]))
