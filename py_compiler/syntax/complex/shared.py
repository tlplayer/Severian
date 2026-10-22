"""Shared references retain a payload descriptor and a separately owned count."""
from dataclasses import dataclass
from py_compiler.syntax.generic.atom import Atom
from py_compiler.mlir.src.cfg import name, temporary_suffix


@dataclass(frozen=True)
class Shared:
    payload: object
    bits: int = 64
    atomic: bool = False
    family: str = 'shared'
    binding_ownership: str = 'view'
    parameter_ownership: str = 'copy'

    @property
    def name(self):
        return ('share atomic ' if self.atomic else 'share ') + self.payload.name

    @property
    def descriptor(self):
        return f'!llvm.struct<(!llvm.ptr, !llvm.ptr, i{self.bits}, !llvm.array<1 x i{self.bits}>, !llvm.array<1 x i{self.bits}>)>'

    @property
    def mlir(self):
        return f'!llvm.struct<({self.descriptor}, {self.descriptor})>'

    def grammars(self):
        return ()

    def require_context(self, cfg):
        pass

    def from_value(self, value, cfg, span):
        if value.type == self:
            return value
        if value.type != self.payload:
            raise ValueError('share requires a matching owning value')
        if self.payload.family == 'record':
            from py_compiler.syntax.complex.array import ArrayType, record_operation, RecordElementPlace
            from py_compiler.syntax.function.allocate import resolve_plan
            from py_compiler.syntax.function.calls import literal_value
            backing = ArrayType(self.payload)
            count = literal_value(1, cfg.context.type('index'), cfg, span)
            plan = resolve_plan(backing, {'location': 'heap'}, cfg.context)
            storage = cfg.emit('allocate-array', backing, (count,), record_operation(backing, 'allocate', cfg).with_plan(plan), span)
            zero = literal_value(0, count.type, cfg, span)
            # Sharing transfers the existing owner into backing storage. A
            # copy followed by release would run resource destructors early.
            cfg.effect('element-store', (storage, zero, value), record_operation(backing, 'store', cfg), span)
            value = storage
        elif not self.payload.mlir.startswith('memref') or not callable(getattr(value.type, 'release', None)):
            raise ValueError('share requires an owning class or memory value')
        elif value.identity in cfg.stack_values:
            raise ValueError('cannot share explicitly stack-owned storage')
        return cfg.emit('memory-copy', self, (value,), SharedOperation('create', self), span)

    @property
    def backing(self):
        from py_compiler.syntax.complex.array import ArrayType
        return ArrayType(self.payload) if self.payload.family == 'record' else self.payload

    def data(self, receiver, cfg, span):
        value = cfg.emit('share-data', self.backing, (receiver,), SharedOperation('data', self), span)
        if self.payload.family == 'record':
            from py_compiler.syntax.complex.array import RecordElementPlace
            from py_compiler.syntax.function.calls import literal_value
            zero = literal_value(0, cfg.context.type('index'), cfg, span)
            return RecordElementPlace(value, zero, self.payload).read(cfg, span)
        return value

    def field_place(self, receiver, field, cfg, span, write=False):
        if write:
            raise ValueError('shared references expose read-only payloads')
        provider = cfg.context.provider_by_name[self.payload.name]
        return SharedPlace(provider.field_place(self.data(receiver, cfg, span), field, cfg, span))

    def call_method(self, receiver, method, arguments, cfg, env, span, binding):
        from py_compiler.syntax.function.calls import select, invoke
        if method.startswith('__'):
            raise ValueError('private method is inaccessible outside its receiver')
        entries = cfg.context.functions.get(self.payload.name + '.' + method, ())
        entry, operands = select(entries, arguments, cfg, env)
        if entry.receiver_ownership != 'view':
            raise ValueError('shared references expose read-only payloads')
        result = invoke(entry, operands, cfg, span, self.data(receiver, cfg, span))
        if result and entry.body and entry.body.result_sources and binding:
            cfg.call_result_owners[result.identity] = binding
            cfg.call_result_modes[result.identity] = 'view'
        return result

    def copy_value(self, value, cfg, span):
        return cfg.emit('share-retain', self, (value,), SharedOperation('retain', self), span)

    def release(self, value, cfg, span):
        from py_compiler.hir.thir.src.model import Edge, Terminator
        last = cfg.emit('share-release', cfg.context.type('bool'), (value,), SharedOperation('release', self), span)
        destroy, done = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('conditional', (Edge(destroy.identity), Edge(done.identity)), last)
        cfg.current = destroy
        payload = cfg.emit('share-data', self.backing, (value,), SharedOperation('data', self), span)
        cfg.stack_values.discard(payload.identity)
        self.backing.release(payload, cfg, span)
        cfg.effect('share-counter-free', (value,), SharedOperation('counter-free', self), span)
        cfg.current.terminator = Terminator('jump', (Edge(done.identity),))
        cfg.current = done
        from py_compiler.mir.memory import EndLocal
        cfg.effect('share-end', (value,), EndLocal(), span)

    @property
    def element(self):
        return self.payload.element

    def element_place(self, receiver, index, cfg, env, span):
        payload = self.data(receiver, cfg, span)
        return SharedPlace(self.payload.element_place(payload, index, cfg, env, span))


@dataclass(frozen=True)
class SharedPlace:
    place: object

    @property
    def type(self):
        return self.place.type

    def read(self, cfg, span):
        return self.place.read(cfg, span)

    def write(self, cfg, value, span):
        raise ValueError('shared references expose read-only payloads')


@dataclass(frozen=True)
class SharedOperation:
    kind: str
    type: Shared

    def atom(self, operation):
        return Atom(self, tuple(v.type for v in operation.operands), operation.result.type if operation.result else None,
                    ('move',) if self.kind == 'create' else ('view',),
                    ('destruction',) if self.kind == 'release' else ('read',), self)

    def render_operation(self, operation):
        source = name(operation.operands[0])
        tag = temporary_suffix(operation)
        desc, shared, data = self.type.descriptor, self.type.mlir, self.type.backing.mlir
        result = name(operation.result) if operation.result else ''
        if self.kind == 'create':
            return [f'%rc_{tag} = memref.alloc() : memref<1xi64>',
                    f'%one_{tag} = arith.constant 1 : i64', f'%zero_{tag} = arith.constant 0 : index',
                    f'memref.store %one_{tag}, %rc_{tag}[%zero_{tag}] : memref<1xi64>',
                    f'%data_{tag} = builtin.unrealized_conversion_cast {source} : {data} to {desc}',
                    f'%counter_{tag} = builtin.unrealized_conversion_cast %rc_{tag} : memref<1xi64> to {desc}',
                    f'%empty_{tag} = llvm.mlir.undef : {shared}',
                    f'%first_{tag} = llvm.insertvalue %data_{tag}, %empty_{tag}[0] : {shared}',
                    f'{result} = llvm.insertvalue %counter_{tag}, %first_{tag}[1] : {shared}']
        if self.kind == 'data':
            return [f'%data_{tag} = llvm.extractvalue {source}[0] : {shared}',
                    f'{result} = builtin.unrealized_conversion_cast %data_{tag} : {desc} to {data}']
        if self.kind == 'counter-free':
            return [f'%counter_{tag} = llvm.extractvalue {source}[1] : {shared}',
                    f'%rc_{tag} = builtin.unrealized_conversion_cast %counter_{tag} : {desc} to memref<1xi64>',
                    f'memref.dealloc %rc_{tag} : memref<1xi64>']
        delta = 1 if self.kind == 'retain' else -1
        lines = [f'%counter_{tag} = llvm.extractvalue {source}[1] : {shared}',
                 f'%rc_{tag} = builtin.unrealized_conversion_cast %counter_{tag} : {desc} to memref<1xi64>',
                 f'%zero_{tag} = arith.constant 0 : index', f'%delta_{tag} = arith.constant {delta} : i64']
        if self.type.atomic:
            lines.append(f'%old_{tag} = memref.atomic_rmw addi %delta_{tag}, %rc_{tag}[%zero_{tag}] : (i64, memref<1xi64>) -> i64')
        else:
            lines += [f'%old_{tag} = memref.load %rc_{tag}[%zero_{tag}] : memref<1xi64>',
                      f'%next_{tag} = arith.addi %old_{tag}, %delta_{tag} : i64',
                      f'memref.store %next_{tag}, %rc_{tag}[%zero_{tag}] : memref<1xi64>']
        if self.kind == 'retain':
            lines += [f'%limit_{tag} = arith.constant 9223372036854775807 : i64',
                      f'%fits_{tag} = arith.cmpi ult, %old_{tag}, %limit_{tag} : i64',
                      f'cf.assert %fits_{tag}, "reference count overflow"',
                      f'{result} = llvm.insertvalue %counter_{tag}, {source}[1] : {shared}']
        else:
            lines += [f'%one_{tag} = arith.constant 1 : i64',
                      f'{result} = arith.cmpi eq, %old_{tag}, %one_{tag} : i64']
        return lines


import unittest


class SharedTests(unittest.TestCase):
    def test_shared_reference_survives_original_drop(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_c
        for atomic in ('', 'atomic '):
            source = f'def work() -> int:\n    unsafe:\n        a = allocate[int]()\n        a[0] = 42\n        b = share {atomic}a\n        drop a\n        return b[0]\n'
            result = compile_source('share.sev', source, Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            text = render(lower(result.program))
            self.assertEqual('memref.atomic_rmw' in text, bool(atomic))
            verify_c(text)

    def test_shared_payload_cannot_mutate(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('share.sev', 'def work():\n    unsafe:\n        a = allocate[int]()\n        b = share atomic a\n        b[0] = 2\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('read-only', str(result.diagnostics))

    def test_c_lifetimes_and_shared_parameter_copy(self):
        import ctypes
        from pathlib import Path
        import shutil
        import subprocess
        import tempfile
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.lir.src.lib import lower, compile_object
        if not shutil.which('cc'):
            self.skipTest('c linker unavailable')
        source = ('def read(value: share atomic array[int]) -> int:\n    return value[0]\ndef work() -> int:\n    unsafe:\n        a = allocate[int]()\n        a[0] = 42\n        b = share atomic a\n        c = share atomic b\n        drop a\n        drop b\n        return read(c) + c[0]\n')
        result = compile_source('shared-c.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        artifact = compile_object(lower(result.program, 'x86_64-unknown-linux-gnu', 64))
        with tempfile.TemporaryDirectory(prefix='sev-shared-test-') as directory:
            path = Path(directory)
            (path / 'shared.o').write_bytes(artifact.object_bytes)
            subprocess.run(['cc', '-shared', str(path / 'shared.o'), '-o', str(path / 'shared.so')], check=True, capture_output=True)
            library = ctypes.CDLL(str(path / 'shared.so'))
            function = library[next(body.name for body in result.program.bodies if body.declaration == 'work')]
            function.restype = ctypes.c_int64
            self.assertEqual(function(), 84)

    def test_c_shared_classes_destroy_owned_fields_at_last_handle(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''class Buffer:
    data: array[int]
    label: string
    def size(self: view Self) -> int:
        return len(self.data)
    def values(self: view Self) -> view array[int]:
        return self.data

test "class handles survive the original owner":
    owner = Buffer(array[int,2](), "shared")
    owner.data[0] = 42
    first = share owner
    last = share first
    drop owner
    drop first
    assert(last.size() == 2)
    assert(last.data[0] == 42)
    assert(len(last.label) == 6)
    values = last.values()
    assert(values[0] == 42)

test "shared aggregate arrays run element destructors":
    values = array[string,2]()
    values[0] = "owned"
    saved = share values
    drop values
    assert(len(saved[0]) == 5)
'''
        with tempfile.TemporaryDirectory(prefix='sev-shared-class-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'shared-class', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)
