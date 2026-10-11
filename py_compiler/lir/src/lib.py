"""Checked MIR -> target LIR -> LLVM IR -> relocatable object."""
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from py_compiler.mlir.src.lib import lower as lower_mlir, render, verifier_path


@dataclass(frozen=True)
class Lir:
    mlir: object
    target: str
    pointer_bits: int
    dependencies: tuple = ()
    atoms: tuple = ()

    def text(self):
        text = render(self.mlir)
        widths = (8, 16, 32, 64, 128)
        layout = f'e-p:{self.pointer_bits}:{self.pointer_bits}-' + '-'.join(f'i{n}:{n}' for n in widths) + '-f16:16-f32:32-f64:64'
        attribute = f'llvm.data_layout = "{layout}"'
        if text.startswith('module attributes {'):
            return text.replace('module attributes {', 'module attributes {' + attribute + ', ', 1)
        return text.replace('module {', 'module attributes {' + attribute + '} {', 1)


@dataclass(frozen=True)
class ObjectArtifact:
    llvm_dialect: str
    llvm_ir: str
    object_bytes: bytes
    dependencies: tuple = ()


def lower(program, target, pointer_bits):
    if program.stage != 'mir' or (program.typed_modules and not program.mir_modules):
        raise ValueError('LIR requires MIR registration after THIR')
    if program.mir_modules:
        registered_bodies = tuple(body for module in program.mir_modules for body in module.bodies)
        if (len({id(body) for body in registered_bodies}) != len(registered_bodies)
                or len({id(body) for body in program.bodies}) != len(program.bodies)
                or {id(body) for body in registered_bodies} != {id(body) for body in program.bodies}):
            raise ValueError('LIR body membership changed after MIR registration')
    for body in program.bodies:
        body.verify()
    from py_compiler.syntax.generic.atom import ExternalSymbol
    dependencies = set()
    for body in program.bodies:
        for block in body.blocks:
            for operation in block.operations:
                if operation.atom is None:
                    raise ValueError(f"unresolved operation at LIR boundary: {operation.kind}")
                implementation = operation.atom.implementation
                if isinstance(implementation, ExternalSymbol):
                    dependencies.update(implementation.dependencies)
    atoms = tuple(atom for module in program.mir_modules for sub in module.submodules for atom in sub.atoms)
    if program.mir_modules:
        current = {(body.name, block.identity, position): operation.atom
                   for body in program.bodies for block in body.blocks
                   for position, operation in enumerate(block.operations)}
        registered = {(atom.callable, atom.block, atom.position): atom.atom for atom in atoms}
        if len(registered) != len(atoms) or current.keys() != registered.keys():
            raise ValueError('LIR operation membership changed after MIR registration')
        if any(current[key] is not registered[key] for key in current):
            raise ValueError('LIR Atom changed after MIR registration')
    return Lir(lower_mlir(program), target, pointer_bits, tuple(sorted(dependencies)), atoms)


def companion(name, optimizer):
    optimizer = Path(shutil.which(optimizer) or optimizer)
    suffix = optimizer.name.removeprefix('mlir-opt')
    candidate = optimizer.with_name(name + suffix)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(name + suffix)
    if not found:
        raise ValueError(f'missing matching LLVM tool: {name + suffix}')
    return found


def compile_object(lir, optimizer=None):
    optimizer = optimizer or verifier_path()
    translator, compiler = companion('mlir-translate', optimizer), companion('llc', optimizer)
    def run(arguments, text):
        result = subprocess.run(arguments, input=text, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise ValueError(f'{Path(arguments[0]).name} failed:\n{result.stderr}')
        return result.stdout
    dialect = run([optimizer, '--expand-strided-metadata', '--arith-expand', '--convert-math-to-llvm', '--convert-scf-to-cf',
                   '--convert-arith-to-llvm', f'--convert-index-to-llvm=index-bitwidth={lir.pointer_bits}',
                   '--convert-cf-to-llvm', f'--finalize-memref-to-llvm=index-bitwidth={lir.pointer_bits}',
                   f'--convert-func-to-llvm=index-bitwidth={lir.pointer_bits}',
                   '--convert-arith-to-llvm', '--reconcile-unrealized-casts',
                   '--ensure-debug-info-scope-on-llvm-func', '--mlir-print-debuginfo'], lir.text())
    llvm_ir = run([translator, '--mlir-to-llvmir'], dialect)
    with tempfile.TemporaryDirectory(prefix='sev-object-') as directory:
        destination = Path(directory) / 'package.o'
        run([compiler, '-O0', '--frame-pointer=all', '-filetype=obj', '-relocation-model=pic', '-mtriple=' + lir.target,
             '-o', str(destination)], llvm_ir)
        object_bytes = destination.read_bytes()
        if not object_bytes:
            raise ValueError('object compiler produced an empty artifact')
    return ObjectArtifact(dialect, llvm_ir, object_bytes, lir.dependencies)


import unittest


class DebugInfoTests(unittest.TestCase):
    def test_source_locations_survive_lowering_into_dwarf(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = 'def leaf(value: int) -> int:\n    return value + 1\ndef caller() -> int:\n    return leaf(41)\n'
        result = compile_source('debug.sev', source, Syntax(), debug_path='/tmp/severian-debug/debug.sev')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        artifact = compile_object(lower(result.program, 'x86_64-unknown-linux-gnu', 64))
        self.assertIn('!DISubprogram', artifact.llvm_ir)
        self.assertIn('!DILocation(line: 4,', artifact.llvm_ir)
        self.assertIn('filename: "debug.sev", directory: "/tmp/severian-debug"', artifact.llvm_ir)
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / 'debug.o'
            obj.write_bytes(artifact.object_bytes)
            tool = companion('llvm-dwarfdump', verifier_path())
            checked = subprocess.run([tool, '--verify', str(obj)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            lines = subprocess.run([tool, '--debug-line', str(obj)], capture_output=True, text=True, check=True)
            self.assertIn('debug.sev', lines.stdout)
