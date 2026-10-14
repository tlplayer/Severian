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
from py_compiler.mlir.src.cfg import name, temporary_suffix


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
        tag = temporary_suffix(operation)
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

    def test_unsafe_pointer_load_and_store_are_bounded(self):
        text = self.ir('def read(p: pointer[f64], i: isize) -> f64:\n    unsafe:\n        return p[i]\ndef write(p: borrow pointer[f64]):\n    unsafe:\n        p[0] = 1.0\n')
        self.assertIn('memref.load', text)
        self.assertIn('memref.store', text)
        self.assertIn('memory index outside allocation', text)

    def test_address_infers_view_and_preserves_owner(self):
        text = self.ir('def work() -> int:\n    unsafe:\n        value = 42\n        p = &value\n        value = 43\n        return p[0] + value\n')
        self.assertIn('memref.alloca', text)
        self.assertNotIn('memref.dealloc', text)
        result = self.compile('def work() -> pointer[int]:\n    unsafe:\n        value = 42\n        return &value\n')
        self.assertTrue(result.diagnostics)

    def test_view_cannot_release_storage(self):
        result = self.compile('def work():\n    unsafe:\n        a = allocate[int]()\n        p: pointer[int] = a\n        deallocate(p)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('cannot deallocate a view', str(result.diagnostics))

    def test_pointer_lowering_verifies_natively(self):
        import shutil
        from py_compiler.mlir.src.lib import verify_native, verifier_path
        if not shutil.which(verifier_path()):
            self.skipTest('install mlir-opt to verify pointer lowering')
        verify_native(self.ir('def work() -> int:\n    unsafe:\n        a = allocate[int]()\n        p: pointer[int] = a\n        return p[0]\n'))


@dataclass(frozen=True)
class BoundedAddress:
    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), (), self)

    def render_operation(self, operation):
        source = operation.operands[0]
        return [f'{name(operation.result)} = memref.cast {name(source)} : {source.type.mlir} to {operation.result.type.mlir}']
