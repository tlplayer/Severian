"""Raw pointer comparisons are available only in an unsafe context."""
from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation


class PointerComparison(ScalarOperation):
    def __init__(self, owner, spelling):
        super().__init__(owner, spelling, 3, comparison=True)

    def expand(self, cfg, left, right, span):
        self.owner.require_context(cfg)
        return super().expand(cfg, left, right, span)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        predicate = {'==': 'eq', '!=': 'ne', '<': 'ult', '<=': 'ule', '>': 'ugt', '>=': 'uge'}[self.spelling]
        left, right = map(name, operation.operands)
        return [f'{name(operation.result)} = llvm.icmp "{predicate}" {left}, {right} : !llvm.ptr']


def grammars(owner):
    return tuple(PointerComparison(owner, spelling) for spelling in ('==', '!=', '<', '<=', '>', '>='))


import unittest


class PointerTests(unittest.TestCase):
    def test_comparison_is_pointer_typed_and_unsafe(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        for spelling, predicate in (('==', 'eq'), ('!=', 'ne'), ('<', 'ult'), ('<=', 'ule'), ('>', 'ugt'), ('>=', 'uge')):
            result = compile_source('pointer.sev', f'def compare(a: pointer, b: pointer) -> bool:\n    unsafe:\n        return a {spelling} b\n', Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            self.assertIn(f'llvm.icmp "{predicate}"', render(lower(result.program)))
        result = compile_source('invalid.sev', 'def compare(a: pointer, b: pointer) -> bool:\n    return a == b\n', Syntax())
        self.assertTrue(result.diagnostics)


from dataclasses import dataclass
from py_compiler.syntax.generic.atom import Atom
from py_compiler.mlir.src.cfg import name


@dataclass(frozen=True)
class ArrayAddress:
    owner: object

    def atom(self, operation):
        return Atom(self.owner, (operation.operands[0].type,), self.owner, ('view',), (), self)

    def render_operation(self, operation):
        source, result = operation.operands[0], operation.result
        address, integer = f'%address_{result.identity}', f'%address_bits_{result.identity}'
        width = f'i{self.owner.bits}'
        return [f'{address} = memref.extract_aligned_pointer_as_index {name(source)} : {source.type.mlir} -> index',
                f'{integer} = arith.index_cast {address} : index to {width}',
                f'{name(result)} = llvm.inttoptr {integer} : {width} to !llvm.ptr']


@dataclass(frozen=True)
class PointerPlace:
    receiver: object
    index: object
    type: object

    def read(self, cfg, span):
        return cfg.emit('pointer-load', self.type, (self.receiver, self.index), PointerAccess(self.type, False), span)

    def write(self, cfg, value, span):
        cfg.effect('pointer-store', (self.receiver, self.index, value), PointerAccess(self.type, True), span)


@dataclass(frozen=True)
class PointerAccess:
    element: object
    write: bool

    def atom(self, operation):
        return Atom(operation.operands[0].type, tuple(v.type for v in operation.operands),
                    None if self.write else self.element,
                    ('borrow', 'copy', 'copy') if self.write else ('view', 'copy'),
                    ('mutation',) if self.write else ('read',), self)

    def render_operation(self, operation):
        pointer, index, *values = operation.operands
        tag = 'store_' + str(operation.span.start) if self.write else str(operation.result.identity)
        address = f'%element_address_{tag}'
        lines = [f'{address} = llvm.getelementptr {name(pointer)}[{name(index)}] : (!llvm.ptr, {index.type.mlir}) -> !llvm.ptr, {self.element.mlir}']
        if self.write:
            lines.append(f'llvm.store {name(values[0])}, {address} : {self.element.mlir}, !llvm.ptr')
        else:
            lines.append(f'{name(operation.result)} = llvm.load {address} : !llvm.ptr -> {self.element.mlir}')
        return lines


class TypedPointerTests(unittest.TestCase):
    def compile(self, source, pointer_bits=64):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('typed-pointer.sev', source, Syntax(pointer_bits))

    def ir(self, source, pointer_bits=64):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile(source, pointer_bits)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return render(lower(result.program))

    def test_noninteger_and_nested_elements_use_their_own_llvm_type(self):
        for element, storage in (('int', 'i64'), ('f64', 'f64'), ('bool', 'i1'), ('pointer[i32]', '!llvm.ptr')):
            with self.subTest(element=element):
                text = self.ir(f'def read(p: pointer[{element}], i: isize) -> {element}:\n    unsafe:\n        return p[i]\n')
                self.assertIn('llvm.getelementptr', text)
                self.assertIn(f'-> !llvm.ptr, {storage}', text)
                self.assertIn(f': !llvm.ptr -> {storage}', text)
                self.assertNotIn('memref.load', text)
                self.assertNotIn('cf.assert', text)
                self.assertNotIn('memref.dealloc', text)

    def test_pointer_store_uses_llvm_and_requires_unsafe(self):
        text = self.ir('def write(p: borrow pointer[f64], i: isize, value: f64):\n    unsafe:\n        p[i] = value\n')
        self.assertIn('llvm.store', text)
        self.assertIn(': f64, !llvm.ptr', text)
        self.assertNotIn('memref.store', text)
        for source in (
            'def read(p: pointer[int]) -> int:\n    return p[0]\n',
            'def write(p: borrow pointer[int]):\n    p[0] = 1\n',
        ):
            result = self.compile(source)
            self.assertTrue(result.diagnostics)
            self.assertIn('pointer indexing requires an unsafe block', str(result.diagnostics[0]))

    def test_array_address_uses_target_pointer_width(self):
        for width in (32, 64):
            text = self.ir('def address(values: array[int]) -> pointer[int]:\n    return pointer[int](values)\n', width)
            self.assertIn('memref.extract_aligned_pointer_as_index', text)
            self.assertIn(f': index to i{width}', text)
            self.assertIn(f': i{width} to !llvm.ptr', text)
            self.assertNotIn('memref.dealloc', text)

    def test_array_address_and_pointer_access_verify_natively(self):
        import shutil
        from py_compiler.mlir.src.lib import verify_native, verifier_path
        if not shutil.which(verifier_path()):
            self.skipTest('install mlir-opt to verify pointer lowering')
        text = self.ir('def work() -> int:\n    unsafe:\n        a = allocate[int](1)\n        a[0] = 7\n        p: pointer[int] = view a\n        return p[0]\n')
        verify_native(text)
