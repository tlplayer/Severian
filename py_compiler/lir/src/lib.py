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

    def text(self):
        return render(self.mlir)


@dataclass(frozen=True)
class ObjectArtifact:
    llvm_dialect: str
    llvm_ir: str
    object_bytes: bytes
    dependencies: tuple = ()


def lower(program, target, pointer_bits):
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
    return Lir(lower_mlir(program), target, pointer_bits, tuple(sorted(dependencies)))


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
    dialect = run([optimizer, '--expand-strided-metadata', '--convert-scf-to-cf',
                   '--convert-arith-to-llvm', f'--convert-index-to-llvm=index-bitwidth={lir.pointer_bits}',
                   '--convert-cf-to-llvm', f'--finalize-memref-to-llvm=index-bitwidth={lir.pointer_bits}',
                   f'--convert-func-to-llvm=index-bitwidth={lir.pointer_bits}',
                   '--convert-arith-to-llvm', '--reconcile-unrealized-casts'], lir.text())
    llvm_ir = run([translator, '--mlir-to-llvmir'], dialect)
    with tempfile.TemporaryDirectory(prefix='sev-object-') as directory:
        destination = Path(directory) / 'package.o'
        run([compiler, '-filetype=obj', '-relocation-model=pic', '-mtriple=' + lir.target,
             '-o', str(destination)], llvm_ir)
        object_bytes = destination.read_bytes()
        if not object_bytes:
            raise ValueError('object compiler produced an empty artifact')
    return ObjectArtifact(dialect, llvm_ir, object_bytes, lir.dependencies)
