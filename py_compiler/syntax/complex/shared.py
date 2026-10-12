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
        if not self.payload.mlir.startswith('memref'):
            raise ValueError('shared class ownership requires an owning aggregate handle')
        if value.type != self.payload or not callable(getattr(value.type, 'release', None)):
            raise ValueError('share requires a matching owning memory value')
        if value.identity in cfg.stack_values:
            raise ValueError('cannot share explicitly stack-owned storage')
        return cfg.emit('share-create', self, (value,), SharedOperation('create', self), span)

    def copy_value(self, value, cfg, span):
        return cfg.emit('share-retain', self, (value,), SharedOperation('retain', self), span)

    def release(self, value, cfg, span):
        cfg.effect('share-release', (value,), SharedOperation('release', self), span)

    def element_place(self, receiver, index, cfg, env, span):
        payload = cfg.emit('share-data', self.payload, (receiver,), SharedOperation('data', self), span)
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
                    ('move',) if self.kind in ('create', 'release') else ('view',),
                    ('destruction',) if self.kind == 'release' else ('read',), self)

    def render_operation(self, operation):
        source = name(operation.operands[0])
        tag = temporary_suffix(operation)
        desc, shared, data = self.type.descriptor, self.type.mlir, self.type.payload.mlir
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
                      f'%last_{tag} = arith.cmpi eq, %old_{tag}, %one_{tag} : i64',
                      f'scf.if %last_{tag} {{', f'  %data_{tag} = llvm.extractvalue {source}[0] : {shared}',
                      f'  %payload_{tag} = builtin.unrealized_conversion_cast %data_{tag} : {desc} to {data}',
                      f'  memref.dealloc %payload_{tag} : {data}', f'  memref.dealloc %rc_{tag} : memref<1xi64>', '}']
        return lines


import unittest


class SharedTests(unittest.TestCase):
    def test_shared_reference_survives_original_drop(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render, verify_native
        for atomic in ('', 'atomic '):
            source = f'def work() -> int:\n    a = allocate[int]()\n    a[0] = 42\n    b = share {atomic}a\n    drop a\n    return b[0]\n'
            result = compile_source('share.sev', source, Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))
            text = render(lower(result.program))
            self.assertEqual('memref.atomic_rmw' in text, bool(atomic))
            verify_native(text)

    def test_shared_payload_cannot_mutate(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('share.sev', 'def work():\n    a = allocate[int]()\n    b = share atomic a\n    b[0] = 2\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('read-only', str(result.diagnostics))

    def test_native_lifetimes_and_shared_parameter_copy(self):
        import ctypes
        from pathlib import Path
        import shutil
        import subprocess
        import tempfile
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.lir.src.lib import lower, compile_object
        if not shutil.which('cc'):
            self.skipTest('native linker unavailable')
        source = ('def read(value: share atomic array[int]) -> int:\n    return value[0]\n'
                  'def work() -> int:\n    a = allocate[int]()\n    a[0] = 42\n'
                  '    b = share atomic a\n    c = share atomic b\n    drop a\n    drop b\n'
                  '    return read(c) + c[0]\n')
        result = compile_source('shared-native.sev', source, Syntax())
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
