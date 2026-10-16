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
        from py_compiler.mir.verification import require_ready
        require_ready(body)
    from py_compiler.syntax.generic.atom import ExternalSymbol
    dependencies = set()
    for body in program.bodies:
        for block in body.blocks:
            for operation in block.operations:
                if operation.atom is None:
                    raise ValueError(f"unresolved operation at LIR boundary: {operation.kind}")
                implementation = operation.atom.implementation
                validate_target = getattr(implementation, 'validate_target', None)
                if validate_target:
                    validate_target(target, pointer_bits)
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


@dataclass(frozen=True)
class Partition:
    module: str
    submodule: str
    source: str
    stem: str
    symbols: tuple
    dependencies: tuple
    lir: Lir
    constant_symbols: tuple = ()


def partitions(program, lir):
    """Emit registered semantic submodules, independent of source snapshots."""
    from dataclasses import replace
    from hashlib import sha256
    import json
    import re
    from py_compiler.hir.hir.src.program import Constant
    from py_compiler.mlir.src.lib import MlirProgram, MlirFunction
    def key(value):
        return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    def readable(value):
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._') or 'initializer'
    output = []
    owned = set()
    for module in program.mir_modules:
        executable = {body.name: body for body in module.bodies}
        typed_module = next(item for item in program.typed_modules if item.identity == module.identity)
        source = typed_module.source.path
        module_name = readable(module.interface + '.' + str(Path(source).with_suffix('')).replace('/', '.'))
        for submodule in module.submodules:
            sub = submodule.typed
            symbols = tuple(sorted(call.body.name for call in sub.callables))
            descriptors = symbols or tuple((c.name, c.type.name, c.span.start) for c in sub.constants)
            stable = key((module.interface, source, descriptors))
            labels = [('test.' + (call.body.test['label'] or 'line_' + str(typed_module.source.position(call.body.source_span.start)[0])))
                      if call.body.test else
                      (call.name + '_' + '_'.join(t.name for t in call.parameters)).replace(typed_module.source.identity + ':', 'scope_') if call.name else 'initializer'
                      for call in sub.callables] or [c.name or 'literal' for c in sub.constants]
            stem = module_name + '/' + readable('+'.join(sorted(labels)))[:100] + '-' + stable[:16]
            constants, bodies, mappings = {}, [], []
            for call in sub.callables:
                body = executable[call.body.name]
                if body.name in owned:
                    raise ValueError('native partition contains a duplicate callable')
                owned.add(body.name)
                blocks = []
                for block in body.blocks:
                    operations = []
                    for operation in block.operations:
                        if isinstance(operation.payload, Constant):
                            old = operation.payload
                            constant = replace(old, identity=stable + ':' + body.name + ':' + str(operation.result.identity))
                            constants[constant.identity] = constant
                            mappings.append((old.identity, '__sev_constant_' + sha256(constant.identity.encode()).hexdigest()))
                            operation = replace(operation, payload=constant, atom=constant.atom(operation))
                        operations.append(operation)
                    blocks.append(replace(block, operations=operations))
                from py_compiler.mir.verification import verify_target_clone
                clone = replace(body, blocks=blocks)
                verify_target_clone(body, clone)
                bodies.append(clone)
            for position, old in enumerate(sub.constants):
                constant = replace(old, identity=stable + ':literal:' + str(position))
                constants[constant.identity] = constant
                mappings.append((old.identity, '__sev_constant_' + sha256(constant.identity.encode()).hexdigest()))
            functions = tuple(MlirFunction('__sev_constant_' + sha256(c.identity.encode()).hexdigest(), c) for c in constants.values())
            globals_ = tuple(place.identity for body in bodies if not body.declaration
                             for place in body.storage if not getattr(place, 'external_symbol', ''))
            mlir = MlirProgram(functions, tuple(bodies), (), globals_)
            native = replace(lir, mlir=mlir, atoms=submodule.atoms)
            output.append(Partition(module.identity, sub.identity, source, stem, (*symbols, *(f.symbol for f in functions)),
                                    sub.dependencies, native, tuple(mappings)))
    if owned != {body.name for body in program.bodies}:
        raise ValueError('native partitions do not cover all registered callables')
    return tuple(output)


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
    llvm_ir = flush_before_abort(llvm_ir)
    with tempfile.TemporaryDirectory(prefix='sev-object-') as directory:
        destination = Path(directory) / 'partition.o'
        run([compiler, '-O0', '--frame-pointer=all', '-filetype=obj', '-relocation-model=pic', '-mtriple=' + lir.target,
             '-o', str(destination)], llvm_ir)
        object_bytes = destination.read_bytes()
        if not object_bytes:
            raise ValueError('object compiler produced an empty artifact')
    return ObjectArtifact(dialect, llvm_ir, object_bytes, lir.dependencies)


def flush_before_abort(llvm_ir):
    """The cf.assert runtime prints to stdout; preserve it when stdout is piped."""
    import re
    pattern = r'(?m)^([ \t]*)(call void @abort\(\).*)$'
    if not re.search(pattern, llvm_ir):
        return llvm_ir
    from itertools import count
    identifiers = count()
    llvm_ir = re.sub(pattern, lambda match:
        f'{match[1]}%sev_flush_before_abort_{next(identifiers)} = call i32 @fflush(ptr null)\n{match[1]}{match[2]}', llvm_ir)
    if not re.search(r'(?m)^(?:declare|define)\s+.*@fflush\(', llvm_ir):
        llvm_ir += '\ndeclare i32 @fflush(ptr)\n'
    return llvm_ir


import unittest


class DebugInfoTests(unittest.TestCase):
    def test_assertion_diagnostic_survives_redirected_stdout(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        import signal
        source = 'def crash() -> int:\n    assert(false)\n    return 0\n'
        result = compile_source('assert-flush.sev', source, Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        artifact = compile_object(lower(result.program, 'x86_64-unknown-linux-gnu', 64))
        symbol = next(body.name for body in result.program.bodies if body.declaration == 'crash')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'crash.o').write_bytes(artifact.object_bytes)
            import json
            (path / 'main.c').write_text('extern long crash(void) __asm__(' + json.dumps('"' + symbol + '"') + ');\nint main(void) { return crash(); }\n')
            subprocess.run(['cc', str(path / 'main.c'), str(path / 'crash.o'), '-o', str(path / 'crash')], check=True, capture_output=True)
            process = subprocess.run([str(path / 'crash')], capture_output=True, text=True)
            self.assertEqual(process.returncode, -signal.SIGABRT)
            self.assertIn('assertion failed', process.stdout)
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
