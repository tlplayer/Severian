"""Package publication: timestamped builds with content identities; index committed last."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
from py_compiler.frontend.src.lib import compile_source
from py_compiler.hir.hir.src.program import Program
from py_compiler.mlir.src.lib import verify_native, verifier_path
from py_compiler.lir.src.lib import lower, compile_object
from py_compiler.syntax.recognition import Syntax


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def digest(value):
    return sha256(value.encode()).hexdigest()


def new_build_id(now=None):
    """Single build-ID format for native and checkpoint outputs, using UTC."""
    instant = datetime.now(timezone.utc) if now is None else now
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S')


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = stream.name
    os.replace(temporary, path)


def compiler_identity():
    root = Path(__file__).parent
    return digest(encode([(path.relative_to(root).as_posix(), path.read_text())
                          for path in sorted(root.rglob('*.py'))]))


def native_dependencies(root, manifest, target, pointer_bits):
    result = {}
    for alias, specification in manifest.get('dependencies', {}).items():
        if not re.fullmatch(r'[A-Za-z_]\w*', alias) or not isinstance(specification, dict) or not isinstance(specification.get('path'), str):
            raise ValueError('native dependencies require an identifier alias and a local path')
        package_root = (root / specification['path']).resolve()
        output = package_root / 'package.pkg'
        def artifact(relative):
            path = (output / relative).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError('dependency artifact escapes package.pkg')
            return path
        index = json.loads(artifact('package.pkgi/index.json').read_text())
        candidates = []
        for item in reversed(index.get('interfaces', [])):
            interface_path = artifact(item['path'])
            interface_text = interface_path.read_text()
            interface = json.loads(interface_text)
            realization = json.loads(artifact(interface['metadata']['path']).read_text())
            if (realization.get('scope') != 'production' or realization.get('state') != 'compiled-object'
                    or realization.get('target') != target or realization.get('pointer-bits') != pointer_bits
                    or realization.get('compiler') != compiler_identity()):
                continue
            if digest(interface_text) != realization['interface']['checksum']:
                raise ValueError('dependency interface checksum mismatch')
            if specification.get('package', interface['package']['name']) != interface['package']['name']:
                raise ValueError('dependency package identity mismatch')
            objects = []
            for entry in realization['native-artifacts']:
                path = artifact(entry['path'])
                if entry['kind'] != 'object' or sha256(path.read_bytes()).hexdigest() != entry['checksum']:
                    raise ValueError('dependency object checksum mismatch')
                objects.append(str(path))
            if not objects:
                raise ValueError('dependency has no native objects')
            if realization.get('dependencies'):
                raise ValueError('transitive native dependencies need a pinned dependency closure')
            candidates.append({'exports': interface.get('exports', []), 'objects': objects,
                               'initializers': realization.get('initializers', []),
                               'package': interface['package'], 'build-id': realization['build-id']})
            break
        if not candidates:
            raise ValueError(f'build compatible production artifacts for dependency {alias!r} first')
        result[alias] = candidates[0]
    return result


def build(root, target="x86_64-unknown-linux-gnu", pointer_bits=64, jobs=4, mlir_opt=None, include_tests=False, stop_after=None):
    root = Path(root).resolve()
    output = root / "package.pkg"
    log = output / "debug/build/log.txt"
    def report(message, failure=False):
        write_atomic(log, message)
        write_atomic(output / "debug/build/error.txt", message if failure else "")
        write_atomic(output / "debug/build/warn.txt", "")
    report("Building with py_compiler\n")
    try:
        return _build(root, output, target, pointer_bits, jobs, mlir_opt, report, include_tests, stop_after)
    except (OSError, ValueError, subprocess.TimeoutExpired) as failure:
        report(str(failure) + "\n", True)
        raise


def _build(root, output, target, pointer_bits, jobs, mlir_opt, report, include_tests=False, stop_after=None):
    if stop_after not in (None, 'thir', 'mir'):
        raise ValueError('stop_after must be thir or mir')
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target):
        raise ValueError("target must be a safe target-triple path component")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    architecture = target.split("-", 1)[0]
    required_width = {"x86_64": 64, "aarch64": 64, "riscv64": 64,
                      "i386": 32, "i686": 32, "arm": 32, "wasm32": 32}.get(architecture)
    if required_width and required_width != pointer_bits:
        raise ValueError(f"{target} requires --pointer-bits {required_width}")
    syntax = Syntax(pointer_bits, include_tests=include_tests)
    syntax.types  # Validate target representation before reading source.
    manifest_text = (root / "package.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    if not isinstance(manifest, dict):
        raise ValueError("package.json must contain an object")
    package = manifest.get("package", {})
    if not isinstance(package, dict) or not package.get("name") or not package.get("version"):
        raise ValueError("package.json requires package.name and package.version")
    if not all(isinstance(package[field], str) for field in ("name", "version")):
        raise ValueError("package.name and package.version must be strings")
    packages = native_dependencies(root, manifest, target, pointer_bits)
    files = sorted((root / "src").rglob("*.sev"))
    if not files:
        raise ValueError("package has no src/**/*.sev sources")
    snapshots = []
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("source symlink escapes the package")
        snapshots.append((path.relative_to(root).as_posix(), path.read_bytes().decode("utf-8")))
    lock = (root / "package.lock").read_text(encoding="utf-8") if (root / "package.lock").exists() else encode({"version": 1, "packages": []})
    # Executor.map preserves source order while independent source units compile concurrently.
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        results = list(executor.map(lambda pair: compile_source(pair[0], pair[1], syntax,
                            stop_after=stop_after or 'mir', interface=package['name'], native_packages=packages), snapshots))
    errors = [diagnostic for result in results for diagnostic in result.diagnostics]
    if errors:
        report("\n\n".join(map(str, errors)) + "\n", True)
        return {"success": False, "diagnostics": len(errors), "sample": [str(e) for e in errors[:5]],
                "log": str(output / "debug/build/log.txt")}
    program = Program(tuple(c for r in results for c in r.program.constants),
                      tuple(s for r in results for s in r.program.submodules),
                      tuple(b for r in results for b in r.program.bodies),
                      tuple(d for r in results for d in r.program.declarations),
                      tuple(m for r in results for m in r.program.typed_modules),
                      tuple(m for r in results for m in r.program.mir_modules), stop_after or 'mir')
    if stop_after:
        return checkpoint(output, program, package, manifest_text, lock, snapshots, target,
                          pointer_bits, include_tests, stop_after, report)
    lir = lower(program, target, pointer_bits)
    target_program = lir.mlir
    ir = lir.text()
    # Publishing an artifact requires the actual dialect verifier, not only our structural checks.
    verify_native(ir, mlir_opt)
    native = compile_object(lir, mlir_opt)
    compiler_id = compiler_identity()
    content_id = digest(encode([manifest_text, lock, snapshots]))
    build_id = new_build_id()
    abi_id = digest(encode([target, pointer_bits, "py-primitive-abi-v1"]))
    identity = {"name": package["name"], "version": package["version"], "content-id": content_id}
    ir_path = f"artifacts/{target}/{build_id}/ir/package.mlir"
    object_path = f"artifacts/{target}/{build_id}/object/package.o"
    llvm_path = f"artifacts/{target}/{build_id}/ir/package.ll"
    lowered_path = f"artifacts/{target}/{build_id}/ir/package.llvm.mlir"
    interface_path = f"package.pkgi/severian/{build_id}/lib.sevi"
    realization_path = f"metadata/realizations/{build_id}.json"
    declarations, symbols = [], []
    for function in target_program.functions:
        c = function.constant
        # File-qualified exports avoid conflating independent module scopes.
        name = f"{c.source}::{c.name}" if c.name else c.identity
        contract = f"{c.name or '<literal>'}: {c.type.name} {c.binding_operator or ''}".strip()
        symbols.append({"id": c.identity, "name": name, "kind": "constant", "type-id": c.type.name,
                        "exported": c.name is not None, "entry": function.symbol,
                        "entry-kind": "constant-materializer", "source": "source/" + c.source})
        if c.name:
            declarations.append({"name": name, "symbol-id": c.identity, "source": "source/" + c.source,
                                 "contract": contract})
    for declaration in program.declarations:
        declarations.append({"name": declaration.name, "symbol-id": declaration.identity,
                             "source": "source/" + declaration.source,
                             "contract": encode({"kind": declaration.kind, "fields": declaration.fields,
                                                 "variants": declaration.variants, "traits": declaration.traits})})
    for body in program.bodies:
        if body.declaration:
            parameters = ", ".join(f"{binding.name}: {binding.type.name}"
                                   for binding in body.bindings[:len(body.blocks[0].parameters)])
            declarations.append({"name": f"{body.source}::{body.declaration}", "symbol-id": body.identity,
                                 "source": "source/" + body.source,
                                 "contract": f"def {body.declaration}({parameters}) -> {body.result_type.name}",
                                 "complexity": body.complexity, "test": body.test})
        symbols.append({"id": body.identity, "name": body.name, "kind": "execution-body", "entry": body.name,
                        "complexity": body.complexity, "test": body.test,
                        "bindings": [{"id": b.identity, "name": b.name, "type": b.type.name,
                                      "constant": b.constant, "ownership": b.ownership} for b in body.bindings]})
    exports = []
    for body in program.bodies:
        if body.declaration and not body.test and '.' not in body.declaration and '[' not in body.declaration:
            parameters = body.bindings[:len(body.blocks[0].parameters)]
            exports.append({'name': body.declaration, 'symbol': body.name,
                            'parameters': [{'name': b.name, 'type': b.type.name, 'ownership': b.ownership} for b in parameters],
                            'result': body.result_type.name, 'result-sources': list(body.result_sources)})
    semantic_id = digest(encode([declarations, exports]))
    interface = {"format": "severian.interface", "version": 1, "package": identity,
                 "semantic-id": semantic_id, "metadata": {"build-id": build_id, "path": realization_path},
                 "declarations": declarations, "exports": exports}
    interface_text = encode(interface)
    realization = {"format": "py_compiler.realization", "version": 1, "package": identity,
                   "scope": "test" if include_tests else "production",
                   "build-id": build_id, "state": "compiled-object", "compiler": compiler_id,
                   "target": target, "abi": abi_id, "pointer-bits": pointer_bits,
                   "interface": {"path": interface_path, "checksum": digest(interface_text)},
                   "artifacts": [{"path": ir_path, "kind": "mlir", "checksum": digest(ir)}],
                   "dependencies": list(native.dependencies), "native-artifacts": [{"path": object_path, "kind": "object", "checksum": sha256(native.object_bytes).hexdigest()}]}
    realization['submodules'] = realization_records(program)
    realization["artifacts"].extend([{"path": llvm_path, "kind": "llvm-ir", "checksum": digest(native.llvm_ir)},
                                     {"path": lowered_path, "kind": "llvm-dialect", "checksum": digest(native.llvm_dialect)},
                                     *realization["native-artifacts"]])
    realization["initializers"] = [body.name for body in program.bodies if not body.declaration]
    realization['dependency-initializers'] = list(dict.fromkeys(
        symbol for dependency in packages.values() for symbol in dependency['initializers']))
    # No native layout is claimed before target data-layout conversion.
    represented = {constant.type.name: constant.type for constant in program.constants}
    for body in program.bodies:
        for block in body.blocks:
            for value in (*block.parameters, *(o.result for o in block.operations if o.result is not None)):
                represented[value.type.name] = value.type
    layouts = {"abi": abi_id, "state": "representation-only", "native-layouts": [],
               "representations": [{"type": t.name, "mlir": t.mlir, "bits": getattr(t, 'bits', 0)}
                                   for t in represented.values()]}
    pending = {ir_path: ir, llvm_path: native.llvm_ir, lowered_path: native.llvm_dialect, interface_path: interface_text, realization_path: encode(realization),
               f"metadata/symbols/{build_id}.json": encode(symbols),
               f"metadata/dependencies/{build_id}.json": encode([{"submodule": sub.identity, "declarations": sub.declarations, "dependencies": sub.dependencies} for sub in program.submodules]),
               f"metadata/layouts/{abi_id}.json": encode(layouts),
               f"build/{build_id}/{target}/artifacts.json": encode(realization["artifacts"]),
               "source/package.json": manifest_text, "source/package.lock": lock}
    pending.update({"source/" + path: text for path, text in snapshots})
    for path, text in pending.items():
        write_atomic(output / path, text)
    object_destination = output / object_path
    object_destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=object_destination.parent, delete=False) as stream:
        stream.write(native.object_bytes)
        temporary = stream.name
    os.replace(temporary, object_destination)
    # Publish the interface index only after every artifact has been written.
    index_path = output / "package.pkgi/index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"interfaces": [], "binding-sets": []}
    if not isinstance(index.get("interfaces"), list):
        raise ValueError("existing interface index uses an incompatible schema")
    entries = [entry for entry in index["interfaces"] if entry.get("build-id") != build_id]
    entries.append({"build-id": build_id, "path": interface_path, "semantic-id": semantic_id})
    index["interfaces"] = entries
    write_atomic(index_path, encode(index))
    report(f"Built {len(snapshots)} source files; {len(program.constants)} primitive values\nMLIR: {ir_path}\n")
    return {"success": True, "build-id": build_id, "ir": str(output / ir_path), "object": str(output / object_path),
            "dependencies": list(native.dependencies), "log": str(output / "debug/build/log.txt")}


def realization_records(program):
    from py_compiler.thir.src.lib import type_identity
    registered = {sub.typed.identity: sub for module in program.mir_modules for sub in module.submodules}
    return [{
        'id': sub.identity, 'source-submodule': sub.source, 'source-members': list(sub.declarations),
        'interface': module.interface, 'module': module.identity, 'source': module.source.path,
        'substitution': sub.substitution, 'dependencies': list(sub.dependencies),
        'constants': [{'id': value.identity, 'name': value.name, 'type': type_identity(value.type)} for value in sub.constants],
        'label': registered[sub.identity].label if sub.identity in registered else None,
        'callables': [{'id': call.identity, 'definition': call.definition, 'name': call.name,
                       'symbol': call.body.name,
                       'parameters': [{'name': value.name, 'id': type_identity(value)} for value in call.parameters],
                       'result': {'name': call.result.name, 'id': type_identity(call.result)},
                       'type-arguments': {name: {'name': value.name, 'id': type_identity(value)}
                                          for name, value in call.type_arguments},
                       'callable-arguments': dict(call.callable_arguments)} for call in sub.callables],
        'atoms': [{'label': atom.label, 'callable': atom.callable, 'block': atom.block, 'position': atom.position}
                  for atom in registered[sub.identity].atoms] if sub.identity in registered else [],
    } for module in program.typed_modules for sub in module.submodules]


def checkpoint(output, program, package, manifest_text, lock, snapshots, target,
               pointer_bits, include_tests, stage, report):
    compiler_root = Path(__file__).parent
    compiler_id = digest(encode([(p.relative_to(compiler_root).as_posix(), p.read_text())
                                for p in sorted(compiler_root.rglob('*.py'))]))
    build_id = new_build_id()
    interface_path = f'package.pkgi/severian/{build_id}/lib.sevi'
    realization_path = f'metadata/realizations/{build_id}.json'
    records = realization_records(program)
    metadata = {'format': 'py_compiler.realization', 'version': 1, 'package': package,
                'build-id': build_id, 'state': stage + '-registered', 'linkable': False,
                'scope': 'test' if include_tests else 'production',
                'compiler': compiler_id, 'target': target, 'pointer-bits': pointer_bits,
                'submodules': records, 'artifacts': [], 'native-artifacts': []}
    interface = {'format': 'severian.interface', 'version': 1, 'package': package,
                 'linkable': False, 'stage': stage,
                 'metadata': {'build-id': build_id, 'path': realization_path},
                 'declarations': [declaration for sub in records for declaration in (*sub['callables'], *sub['constants'])]}
    pending = {interface_path: encode(interface), realization_path: encode(metadata),
               f'metadata/symbols/{build_id}.json': encode(interface['declarations']),
               f'metadata/dependencies/{build_id}.json': encode([
                   {'submodule': sub['id'], 'dependencies': sub['dependencies']} for sub in records]),
               'source/package.json': manifest_text, 'source/package.lock': lock}
    pending.update({'source/' + path: text for path, text in snapshots})
    for path, text in pending.items():
        write_atomic(output / path, text)
    index_path = output / 'package.pkgi/index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'interfaces': [], 'binding-sets': []}
    if not isinstance(index.get('interfaces'), list):
        raise ValueError('existing interface index uses an incompatible schema')
    index['interfaces'] = [entry for entry in index['interfaces'] if entry.get('build-id') != build_id]
    index['interfaces'].append({'build-id': build_id, 'path': interface_path, 'stage': stage, 'linkable': False})
    write_atomic(index_path, encode(index))
    report(f'Stopped after {stage}: {len(records)} typed submodules; no native artifacts\n')
    return {'success': True, 'build-id': build_id, 'stage': stage, 'linkable': False,
            'metadata': str(output / realization_path), 'log': str(output / 'debug/build/log.txt')}


import unittest


class PackageTests(unittest.TestCase):
    def test_build_id_format_is_utc_and_zero_padded(self):
        from datetime import timedelta
        self.assertEqual(new_build_id(datetime(2026, 1, 2, 3, 4, 5, 123456, timezone.utc)),
                         '2026_01_02_03_04_05')
        self.assertEqual(new_build_id(datetime(2026, 1, 1, 22, 4, 5,
                                               tzinfo=timezone(timedelta(hours=-5)))),
                         '2026_01_02_03_04_05')

    def test_hash_checkpoint_layout_has_three_realizations_and_no_native_artifacts(self):
        from unittest.mock import patch
        from py_compiler.thir.src.lib import ThirTests
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src/hash').mkdir(parents=True)
            (root / 'package.json').write_text(encode({'package': {'name': 'geometry', 'version': '0.1.0'}}))
            (root / 'src/hash/mod.sev').write_text(ThirTests.source)
            with patch(__name__ + '.lower') as lir, patch(__name__ + '.compile_object') as native, patch(__name__ + '.verify_native') as verifier:
                result = build(root, include_tests=True, stop_after='mir')
                lir.assert_not_called()
                native.assert_not_called()
                verifier.assert_not_called()
            self.assertTrue(result['success'], str(result))
            output, build_id = root / 'package.pkg', result['build-id']
            expected = {'package.pkgi/index.json', f'package.pkgi/severian/{build_id}/lib.sevi',
                        f'metadata/realizations/{build_id}.json', f'metadata/symbols/{build_id}.json',
                        f'metadata/dependencies/{build_id}.json', 'source/package.json', 'source/package.lock',
                        'source/src/hash/mod.sev', 'debug/build/log.txt', 'debug/build/error.txt', 'debug/build/warn.txt'}
            self.assertEqual({p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}, expected)
            metadata = json.loads(Path(result['metadata']).read_text())
            hashes = [sub for sub in metadata['submodules'] if any(call['name'] == 'hash' for call in sub['callables'])]
            self.assertEqual(len(hashes), 3)
            self.assertTrue(all(sub['interface'] == 'geometry' and sub['label'] for sub in hashes))
            self.assertFalse(metadata['linkable'])
            self.assertEqual(metadata['native-artifacts'], [])

    def test_independent_errors_do_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "package.json").write_text(encode({"package": {"name": "example", "version": "0.1.0"}}))
            (root / "src/a.sev").write_text("x: u8 = 256\n")
            (root / "src/b.sev").write_text("class X:\n")
            result = build(root)
            self.assertFalse(result["success"])
            self.assertEqual(result["diagnostics"], 2)
            self.assertFalse((root / "package.pkg/package.pkgi/index.json").exists())

    def test_verified_package_has_reproducible_identity(self):
        import shutil
        from unittest.mock import patch
        if not shutil.which(verifier_path()):
            self.skipTest("install mlir-opt for package publication test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "package.json").write_text(encode({"package": {"name": "test", "version": "0.1.0"}}))
            (root / "src/constants.sev").write_text("size: i32 = 42\ntext = \"λ\"\n")
            with patch(__name__ + '.new_build_id', side_effect=['2026_01_02_03_04_05', '2026_01_02_03_04_06']) as generate:
                first, second = build(root), build(root)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(first['build-id'], '2026_01_02_03_04_05')
            self.assertEqual(second['build-id'], '2026_01_02_03_04_06')
            index = json.loads((root / "package.pkg/package.pkgi/index.json").read_text())
            self.assertEqual(len(index["interfaces"]), 2)
            self.assertEqual(index['interfaces'][0]['semantic-id'], index['interfaces'][1]['semantic-id'])
            self.assertTrue(Path(first["ir"]).is_file())


class ModularFixtureTests(unittest.TestCase):
    def test_fixture_sources_compile_independently(self):
        root = Path(__file__).parent / 'examples/test/src'
        for path in sorted(root.rglob('*.sev')):
            with self.subTest(source=path.relative_to(root)):
                result = compile_source(str(path), path.read_text(), Syntax())
                self.assertFalse(result.diagnostics, '\n'.join(map(str, result.diagnostics)))
