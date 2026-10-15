"""Reusable development outputs and timestamped publication snapshots."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import tempfile
from py_compiler.frontend.src.lib import compile_source
from py_compiler.hir.hir.src.program import Program
from py_compiler.mlir.src.lib import verify_native, verifier_path
from py_compiler.lir.src.lib import lower, compile_object, partitions, companion
from py_compiler.syntax.recognition import Syntax


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def digest(value):
    return sha256(value.encode()).hexdigest()


def new_build_id(now=None):
    """Publication build-ID format, using UTC."""
    instant = datetime.now(timezone.utc) if now is None else now
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).strftime('%Y_%m_%d_%H_%M_%S')


def output_slot(publish=False, include_tests=False):
    return new_build_id() if publish else ('test' if include_tests else 'dev')


def prepare_output(output, slot, publish=False, preserve_artifacts=False):
    """Replace generated development conflicts; preserve published snapshots."""
    index_path = output / 'package.pkgi/index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'interfaces': [], 'binding-sets': []}
    if not isinstance(index.get('interfaces'), list):
        raise ValueError('existing interface index uses an incompatible schema')
    kept, purge = [], {slot} if not publish else set()
    for entry in index['interfaces']:
        previous = entry.get('output-slot', entry.get('build-id'))
        if entry.get('published'):
            if previous == slot:
                raise ValueError('a publication already exists for this second; retry with a new build ID')
            kept.append(entry)
        elif previous == slot or 'output-slot' not in entry:
            if previous is not None:
                purge.add(previous)
        else:
            kept.append(entry)
    def remove(path):
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError('generated artifact path escapes package.pkg')
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for old in purge:
        if not isinstance(old, str) or not re.fullmatch(r'[A-Za-z0-9_]+', old):
            raise ValueError('invalid generated output slot')
        for target in (output / 'artifacts').glob('*'):
            if not (preserve_artifacts and old == slot):
                remove(target / old)
        for folder in ('package.pkgi/severian', 'build', 'debug/test', 'debug/profile', 'debug/coverage'):
            remove(output / folder / old)
        for folder in ('realizations', 'symbols', 'dependencies'):
            remove(output / 'metadata' / folder / (old + '.json'))
    # Remove invalidated interfaces before writing replacement artifacts.
    index['interfaces'] = kept
    if index_path.exists():
        write_atomic(index_path, encode(index))
    remove(output / 'source')


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


def write_generated(path, data):
    data = data.encode() if isinstance(data, str) else data
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = stream.name
    os.replace(temporary, path)


def compile_partitions(program, lir, output, target, slot, compiler_id, optimizer):
    optimizer = optimizer or verifier_path()
    tools = [shutil.which(optimizer) or optimizer, *(companion(name, optimizer) for name in ('mlir-translate', 'llc'))]
    toolchain = [(str(Path(path).resolve()), Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in tools]
    records, artifacts, payloads, mappings = [], [], {}, {}
    selected = partitions(program, lir)
    stems = {part.submodule: part.stem for part in selected}
    for part in selected:
        text = part.lir.text()
        from py_compiler.mir.verification import artifact_contracts
        cache_key = digest(encode([compiler_id, target, lir.pointer_bits, toolchain, text,
                                  artifact_contracts(part.lir.mlir.bodies, program.bodies)]))
        cache = output / 'cache/native' / cache_key
        names = ('source.mlir', 'lowered.mlir', 'source.ll', 'native.o')
        values = None
        try:
            checksums = json.loads((cache / 'checksums.json').read_text())
            cached = [cache.joinpath(name).read_bytes() for name in names]
            if all(sha256(value).hexdigest() == checksums[name] for name, value in zip(names, cached)):
                values = cached
        except (OSError, ValueError, KeyError, TypeError):
            pass
        reused = values is not None
        if values is None:
            verify_native(text, optimizer)
            native = compile_object(part.lir, optimizer)
            values = [text.encode(), native.llvm_dialect.encode(), native.llvm_ir.encode(), native.object_bytes]
            for name, value in zip(names, values):
                write_generated(cache / name, value)
            write_atomic(cache / 'checksums.json', encode({name: sha256(value).hexdigest() for name, value in zip(names, values)}))
        base = f'artifacts/{target}/{slot}'
        paths = [f'{base}/ir/{part.stem}.mlir', f'{base}/ir/{part.stem}.llvm.mlir',
                 f'{base}/ir/{part.stem}.ll', f'{base}/object/{part.stem}.o']
        entries = [{'path': path, 'kind': kind, 'checksum': sha256(value).hexdigest(),
                    'module': part.module, 'submodule': part.submodule}
                   for path, kind, value in zip(paths, ('mlir', 'llvm-dialect', 'llvm-ir', 'object'), values)]
        entries[-1]['llvm-ir'] = paths[2]
        artifacts.extend(entries)
        payloads.update(zip(paths, values))
        mappings.update(part.constant_symbols)
        records.append({'module': part.module, 'submodule': part.submodule, 'source': part.source,
                        'name': part.stem, 'symbols': part.symbols, 'dependencies': list(part.dependencies),
                        'dependency-partitions': [stems[item] for item in part.dependencies],
                        'cache-key': cache_key, 'reused': reused, 'artifacts': paths,
                        'object': paths[-1], 'llvm-ir': paths[2]})
    return records, artifacts, payloads, mappings


def dependency_closure(packages):
    ordered, names = {}, {}
    for package in packages.values():
        for entry in package['closure']:
            root, name = entry['root'], entry['package']['name']
            if name in names and names[name] != root:
                raise ValueError(f'conflicting local package identity {name!r}: {names[name]} and {root}')
            if root in ordered and ordered[root] != entry:
                raise ValueError(f'inconsistent dependency artifacts for {root}')
            names[name], ordered[root] = root, entry
    return list(ordered.values())


def native_dependencies(root, manifest, target, pointer_bits, *, _active=(), _cache=None):
    root = Path(root).resolve()
    if root in _active:
        raise ValueError('cyclic package dependency: ' + ' -> '.join(map(str, (*_active, root))))
    cache = {} if _cache is None else _cache
    result = {}
    for alias, specification in sorted(manifest.get('dependencies', {}).items()):
        if not re.fullmatch(r'[A-Za-z_]\w*', alias) or not isinstance(specification, dict) or not isinstance(specification.get('path'), str):
            raise ValueError('native dependencies require an identifier alias and a local path')
        package_root = (root / specification['path']).resolve()
        if package_root in (*_active, root):
            raise ValueError('cyclic package dependency: ' + ' -> '.join(map(str, (*_active, root, package_root))))
        if package_root not in cache:
            dependency_manifest = json.loads((package_root / 'package.json').read_text())
            children = native_dependencies(package_root, dependency_manifest, target, pointer_bits,
                                           _active=(*_active, root), _cache=cache)
            closure = dependency_closure(children)
            output = package_root / 'package.pkg'
            def artifact(relative):
                path = (output / relative).resolve()
                if not path.is_relative_to(output.resolve()):
                    raise ValueError('dependency artifact escapes package.pkg')
                return path
            index = json.loads(artifact('package.pkgi/index.json').read_text())
            for item in reversed(index.get('interfaces', [])):
                interface_text = artifact(item['path']).read_text()
                interface = json.loads(interface_text)
                realization = json.loads(artifact(interface['metadata']['path']).read_text())
                if (realization.get('scope') != 'production' or realization.get('state') != 'compiled-object'
                        or realization.get('target') != target or realization.get('pointer-bits') != pointer_bits
                        or realization.get('compiler') != compiler_identity()):
                    continue
                if digest(interface_text) != realization['interface']['checksum']:
                    raise ValueError('dependency interface checksum mismatch')
                if realization.get('dependency-closure', []) != closure:
                    raise ValueError(f'dependency closure changed; rebuild {package_root}')
                objects = []
                for entry in realization['native-artifacts']:
                    path = artifact(entry['path'])
                    if entry['kind'] != 'object' or sha256(path.read_bytes()).hexdigest() != entry['checksum']:
                        raise ValueError('dependency object checksum mismatch')
                    objects.append(str(path))
                if not objects:
                    raise ValueError('dependency has no native objects')
                for entry in realization.get('link-artifacts', []):
                    path = Path(entry['path'])
                    if not path.is_absolute() or path.suffix != '.o' or sha256(path.read_bytes()).hexdigest() != entry['checksum']:
                        raise ValueError('dependency link artifact checksum mismatch')
                    objects.append(str(path))
                if set(realization.get('dependencies', [])) != {entry['path'] for entry in realization.get('link-artifacts', [])}:
                    raise ValueError('dependency is missing checked link artifacts; rebuild it')
                own = {'root': str(package_root), 'package': interface['package'],
                       'interface-checksum': digest(interface_text), 'initializers': realization.get('initializers', [])}
                all_entries = [*closure, own]
                cache[package_root] = {'exports': interface.get('exports', []), 'types': interface.get('types', []),
                    'templates': interface.get('templates', []), 'globals': interface.get('globals', []), 'objects': list(dict.fromkeys(objects)),
                    'initializers': list(dict.fromkeys(symbol for entry in all_entries for symbol in entry['initializers'])),
                    'closure': all_entries, 'package': interface['package'], 'build-id': realization['build-id']}
                break
            else:
                raise ValueError(f'build compatible production artifacts for dependency {alias!r} first')
        dependency = cache[package_root]
        if specification.get('package', dependency['package']['name']) != dependency['package']['name']:
            raise ValueError('dependency package identity mismatch')
        result[alias] = dependency
    dependency_closure(result)
    return result


def build(root, target="x86_64-unknown-linux-gnu", pointer_bits=64, jobs=4, mlir_opt=None, include_tests=False, stop_after=None, publish=False):
    root = Path(root).resolve()
    output = root / "package.pkg"
    log = output / "debug/build/log.txt"
    def report(message, failure=False):
        write_atomic(log, message)
        write_atomic(output / "debug/build/error.txt", message if failure else "")
        write_atomic(output / "debug/build/warn.txt", "")
    report("Building with py_compiler\n")
    try:
        return _build(root, output, target, pointer_bits, jobs, mlir_opt, report, include_tests, stop_after, publish)
    except (OSError, ValueError, subprocess.TimeoutExpired) as failure:
        report(str(failure) + "\n", True)
        raise


def _build(root, output, target, pointer_bits, jobs, mlir_opt, report, include_tests=False, stop_after=None, publish=False):
    if publish and (include_tests or stop_after):
        raise ValueError('publication requires a complete production build')
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
    closure = dependency_closure(packages)
    if any(entry['package']['name'] == package['name'] for entry in closure):
        raise ValueError('root package name conflicts with a dependency')
    files = sorted((root / "src").rglob("*.sev"))
    if not files:
        raise ValueError("package has no src/**/*.sev sources")
    snapshots = []
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("source symlink escapes the package")
        snapshots.append((path.relative_to(root).as_posix(), path.read_bytes().decode("utf-8")))
    lock = (root / "package.lock").read_text(encoding="utf-8") if (root / "package.lock").exists() else encode({"version": 1, "packages": []})
    def compile_snapshot(pair):
        return compile_source(pair[0], pair[1], syntax, stop_after=stop_after or 'mir',
                              interface=package['name'], native_packages=packages,
                              debug_path=str(root / pair[0]))
    # Keep single-source compilation on the interruptible main thread.
    if len(snapshots) == 1 or jobs == 1:
        results = [compile_snapshot(pair) for pair in snapshots]
    else:
        # Executor.map preserves source order for independent source units.
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(compile_snapshot, snapshots))
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
    compiler_id = compiler_identity()
    content_id = digest(encode([manifest_text, lock, snapshots]))
    build_id = output_slot(publish, include_tests)
    abi_id = digest(encode([target, pointer_bits, "py-primitive-abi-v2-record-results"]))
    identity = {"name": package["name"], "version": package["version"], "content-id": content_id}
    partition_records, artifact_records, artifact_payloads, constant_symbols = compile_partitions(
        program, lir, output, target, build_id, compiler_id, mlir_opt)
    from dataclasses import replace
    target_program = replace(target_program, functions=tuple(
        replace(function, symbol=constant_symbols.get(function.constant.identity, function.symbol))
        for function in target_program.functions))
    dependencies = list(dict.fromkeys([*(obj for dependency in packages.values() for obj in dependency['objects']), *lir.dependencies]))
    link_artifacts = [{'path': obj, 'checksum': sha256(Path(obj).read_bytes()).hexdigest()} for obj in dependencies]
    native_records = [entry for entry in artifact_records if entry['kind'] == 'object']
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
    types = [{'name': d.name, 'fields': list(d.fields), 'defaults': list(d.defaults), 'traits': list(d.traits),
              'memory-defaults': list(d.memory_defaults)}
             for d in program.declarations if d.kind == 'class' and '.' not in d.name]
    types += [{'name': d.name, 'kind': 'union', 'variants': list(d.variants)}
              for d in program.declarations if d.kind == 'union' and '.' not in d.name]
    type_names = {t['name'] for t in types}
    exports = []
    for body in program.bodies:
        if body.declaration and not body.declaration.startswith('__iterator_') and not body.test and (('.' not in body.declaration and '[' not in body.declaration) or body.declaration.rsplit('.', 1)[0] in type_names):
            parameters = body.bindings[:len(body.blocks[0].parameters)]
            receiver = parameters[0] if parameters and parameters[0].name == 'self' else None
            if receiver:
                parameters = parameters[1:]
            exports.append({'name': body.declaration, 'symbol': body.name,
                            'receiver': receiver.type.name if receiver else None,
                            'receiver-ownership': receiver.ownership if receiver else None,
                            'parameters': [{'name': b.name, 'type': b.type.name, 'ownership': b.ownership} for b in parameters],
                            'result': body.result_type.name, 'result-sources': list(body.result_sources),
                            'retained-result-sources': list(body.retained_result_sources),
                            'retained-parameter-sources': list(body.retained_parameter_sources),
                            'defaults': body.parameter_defaults,
                            'result-ownership': body.result_ownership, 'acceptance': list(body.dispatch_guards)})
    globals_ = []
    for body in program.bodies:
        if body.declaration:
            continue
        for place in body.storage:
            if getattr(place, 'exported_name', ''):
                if any(item['name'] == place.exported_name for item in globals_):
                    raise ValueError(f'ambiguous exported global {place.exported_name!r}')
                globals_.append({'name': place.exported_name, 'symbol': place.symbol,
                                 'type': place.type.name, 'constant': place.constant})
    templates = [template for result in results for template in result.templates]
    semantic_id = digest(encode([declarations, exports, types, globals_, templates]))
    source_root = f'build/{build_id}/{target}/source' if publish else 'source'
    if publish:
        for entry in (*declarations, *symbols):
            if 'source' in entry:
                entry['source'] = source_root + '/' + entry['source'].removeprefix('source/')
    interface = {"format": "severian.interface", "version": 1, "package": identity,
                 "semantic-id": semantic_id, "metadata": {"build-id": build_id, "path": realization_path},
                 "declarations": declarations, "exports": exports, "types": types, "globals": globals_, "templates": templates}
    interface_text = encode(interface)
    realization = {"format": "py_compiler.realization", "version": 1, "package": identity,
                   "scope": "test" if include_tests else "production",
                   "build-id": build_id, "state": "compiled-object", "compiler": compiler_id,
                   "target": target, "abi": abi_id, "pointer-bits": pointer_bits,
                   "interface": {"path": interface_path, "checksum": digest(interface_text)},
                   "artifacts": artifact_records, "partitions": partition_records,
                   "dependencies": dependencies, "dependency-closure": closure,
                   "link-artifacts": link_artifacts, "native-artifacts": native_records}
    realization['submodules'] = realization_records(program)
    realization['modules'] = [{'id': module.identity, 'interface': module.interface,
                              'partitions': [part['name'] for part in partition_records if part['module'] == module.identity]}
                             for module in program.mir_modules]
    by_submodule = {entry['submodule']: entry for entry in partition_records}
    for submodule in realization['submodules']:
        submodule['artifacts'] = by_submodule[submodule['id']]['artifacts']
    for symbol in symbols:
        owners = [part for part in partition_records if symbol['entry'] in part['symbols']]
        if owners:
            symbol['object'] = owners[0]['object']
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
    interface['metadata'].update({'build-id': build_id if publish else None, 'output-slot': build_id})
    interface_text = encode(interface)
    realization.update({'build-id': build_id if publish else None, 'output-slot': build_id, 'published': publish})
    realization['interface']['checksum'] = digest(interface_text)
    previous_path = output / realization_path
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    prepare_output(output, build_id, publish, preserve_artifacts=True)
    pending = {interface_path: interface_text, realization_path: encode(realization),
               f"metadata/symbols/{build_id}.json": encode(symbols),
               f"metadata/dependencies/{build_id}.json": encode([{"submodule": sub.identity, "declarations": sub.declarations, "dependencies": sub.dependencies} for sub in program.submodules]),
               f"metadata/layouts/{abi_id}.json": encode(layouts),
               f"build/{build_id}/{target}/artifacts.json": encode(realization["artifacts"]),
               source_root + '/package.json': manifest_text, source_root + '/package.lock': lock}
    pending.update({source_root + '/' + path: text for path, text in snapshots})
    for path, text in pending.items():
        write_atomic(output / path, text)
    for path, data in artifact_payloads.items():
        write_generated(output / path, data)
    for entry in previous.get('artifacts', []):
        path = output / entry['path']
        if entry['path'] not in artifact_payloads and path.resolve().is_relative_to(output.resolve()):
            if path.is_file():
                path.unlink()
    # Publish the interface index only after every artifact has been written.
    index_path = output / "package.pkgi/index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"interfaces": [], "binding-sets": []}
    if not isinstance(index.get("interfaces"), list):
        raise ValueError("existing interface index uses an incompatible schema")
    entries = index['interfaces']
    entries.append({'build-id': build_id if publish else None, 'output-slot': build_id, 'published': publish,
                    'path': interface_path, 'semantic-id': semantic_id})
    index["interfaces"] = entries
    write_atomic(index_path, encode(index))
    reused = sum(part['reused'] for part in partition_records)
    report(f"Built {len(snapshots)} source files; {len(partition_records)} native partitions; {reused} reused\n")
    objects = [str(output / entry['path']) for entry in native_records]
    irs = [str(output / entry['path']) for entry in artifact_records if entry['kind'] == 'mlir']
    return {"success": True, "build-id": build_id if publish else None, 'output-slot': build_id,
            "irs": irs, "objects": objects, "ir": irs[0] if irs else None,
            "object": objects[0] if objects else None, "partitions": partition_records,
            "dependencies": dependencies, "log": str(output / "debug/build/log.txt")}


def realization_records(program):
    from py_compiler.hir.thir.src.lib import type_identity
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
    build_id = output_slot(include_tests=include_tests)
    interface_path = f'package.pkgi/severian/{build_id}/lib.sevi'
    realization_path = f'metadata/realizations/{build_id}.json'
    records = realization_records(program)
    metadata = {'format': 'py_compiler.realization', 'version': 1, 'package': package,
                'build-id': None, 'output-slot': build_id, 'published': False, 'state': stage + '-registered', 'linkable': False,
                'scope': 'test' if include_tests else 'production',
                'compiler': compiler_id, 'target': target, 'pointer-bits': pointer_bits,
                'submodules': records, 'artifacts': [], 'native-artifacts': []}
    interface = {'format': 'severian.interface', 'version': 1, 'package': package,
                 'linkable': False, 'stage': stage,
                 'metadata': {'build-id': None, 'output-slot': build_id, 'path': realization_path},
                 'declarations': [declaration for sub in records for declaration in (*sub['callables'], *sub['constants'])]}
    pending = {interface_path: encode(interface), realization_path: encode(metadata),
               f'metadata/symbols/{build_id}.json': encode(interface['declarations']),
               f'metadata/dependencies/{build_id}.json': encode([
                   {'submodule': sub['id'], 'dependencies': sub['dependencies']} for sub in records]),
               'source/package.json': manifest_text, 'source/package.lock': lock}
    pending.update({'source/' + path: text for path, text in snapshots})
    prepare_output(output, build_id)
    for path, text in pending.items():
        write_atomic(output / path, text)
    index_path = output / 'package.pkgi/index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'interfaces': [], 'binding-sets': []}
    if not isinstance(index.get('interfaces'), list):
        raise ValueError('existing interface index uses an incompatible schema')
    index['interfaces'].append({'build-id': None, 'output-slot': build_id, 'published': False,
                                'path': interface_path, 'stage': stage, 'linkable': False})
    write_atomic(index_path, encode(index))
    report(f'Stopped after {stage}: {len(records)} typed submodules; no native artifacts\n')
    return {'success': True, 'build-id': None, 'output-slot': build_id, 'stage': stage, 'linkable': False,
            'metadata': str(output / realization_path), 'log': str(output / 'debug/build/log.txt')}


import unittest


class PackageTests(unittest.TestCase):
    def test_native_generic_defaults_and_class_grammars(self):
        from py_compiler.testing import test_package
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('library', 'consumer'):
                path = root / name
                (path / 'src').mkdir(parents=True)
                manifest = {'package': {'name': name, 'version': '0.1.0'}}
                if name == 'consumer':
                    manifest['dependencies'] = {'library': {'path': '../library'}}
                (path / 'package.json').write_text(encode(manifest))
            (root / 'library/src/lib.sev').write_text(
                'def identity[T](value: T) -> T:\n    return value\n'
                'def answer(value: int = 42) -> int:\n    return value\n'
                'class Cell:\n    value: int\n'
                '    grammar read[self, "[", index: int, "]"]() -> int:\n        return self.value\n'
                '    grammar write[self, "[", index: int, "]", "=", value: int]():\n        self.value = value\n'
                '    grammar iterate["iterator", "(", self, ")"]() -> iterator[int]:\n        yield self.value\n'
                'class GenericCell[T = int]:\n    value: T\n'
                '    grammar read[self, "[", index: int, "]"]() -> T:\n        return self.value\n'
                '    grammar write[self, "[", index: int, "]", "=", value: T]():\n        self.value = value\n')
            (root / 'consumer/src/lib.sev').write_text(
                'import library\ntest "native interface preserves callable and grammar contracts":\n'
                '    assert(library.identity(i32(42)) == i32(42))\n'
                '    assert(library.identity(f64(3.5)) == f64(3.5))\n'
                '    assert(library.answer() == 42)\n    assert(library.answer(7) == 7)\n'
                '    cell = library.Cell(1)\n    cell[0] = 42\n    assert(cell[0] == 42)\n'
                '    cursor = iterator(cell)\n    assert(next(cursor).value == 42)\n    assert(next(cursor) == None)\n'
                '    generic = library.GenericCell[i32](i32(5))\n    generic[0] = i32(9)\n    assert(generic[0] == i32(9))\n')
            result = test_package(root / 'consumer')
            self.assertTrue(result['success'], result)

    def test_single_source_compiles_on_the_calling_thread(self):
        from unittest.mock import patch
        import threading
        caller = threading.get_ident()
        def interrupt(*args, **kwargs):
            self.assertEqual(threading.get_ident(), caller)
            raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text('value = 1\n')
            (root / 'package.json').write_text(encode({'package': {'name': 'interrupt', 'version': '0.1.0'}}))
            with patch(__name__ + '.compile_source', side_effect=interrupt), patch(__name__ + '.ThreadPoolExecutor') as executor:
                with self.assertRaises(KeyboardInterrupt):
                    build(root, stop_after='mir')
                executor.assert_not_called()

    def test_native_diamond_globals_and_initializers(self):
        from py_compiler.testing import test_package, build_package
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def package(name, source, dependencies=()):
                path = root / name
                (path / 'src').mkdir(parents=True)
                (path / 'package.json').write_text(encode({
                    'package': {'name': name, 'version': '0.1.0'},
                    'dependencies': {dep: {'package': dep, 'path': '../' + dep} for dep in dependencies}}))
                (path / 'src/lib.sev').write_text(source)
                return path
            package('base', 'value = 0\ndef bump() -> int:\n    module.value += 1\n    return module.value\n')
            branch = 'import base\nvalue = base.bump()\ndef read() -> int:\n    return module.value\n'
            package('left', branch, ('base',))
            package('right', branch, ('base',))
            package('unused', 'import base\nvalue = base.bump()\n', ('base',))
            package('constants', 'answer := 42\n')
            app = package('app',
                'import base\nimport left\nimport right\nimport constants\n'
                'test "diamond initializes once and isolates identical source globals":\n'
                '    assert(left.read() == 1)\n    assert(right.read() == 2)\n'
                '    assert(base.value == 3)\n    assert(left.value == 1)\n'
                '    left.value = 40\n    left.value += 2\n'
                '    assert(left.read() == 42)\n    assert(right.read() == 2)\n'
                '    assert(constants.answer == 42)\n'
                'test with compiler "imported constants reject mutation":\n'
                '    reject "imported constant":\n        import constants\n'
                '        def invalid():\n            constants.answer = 0\n',
                ('right', 'base', 'unused', 'left', 'constants'))
            with contextlib.redirect_stdout(io.StringIO()):
                result = test_package(app)
            self.assertTrue(result['success'], result)
            metadata = json.loads((app / 'package.pkg/metadata/realizations/test.json').read_text())
            self.assertGreaterEqual(len(metadata['dependencies']), 5)
            self.assertEqual(len(metadata['dependency-initializers']), 5)
            self.assertEqual([entry['package']['name'] for entry in metadata['dependency-closure']],
                             ['base', 'constants', 'left', 'right', 'unused'])
            def interface(name):
                return json.loads((root / name / 'package.pkg/package.pkgi/severian/dev/lib.sevi').read_text())
            self.assertNotEqual(interface('left')['globals'][0]['symbol'], interface('right')['globals'][0]['symbol'])
            # A rebuilt leaf invalidates its consumers even when its interface shape is unchanged.
            (root / 'base/src/lib.sev').write_text('value = 1\ndef bump() -> int:\n    module.value += 1\n    return module.value\n')
            self.assertTrue(build_package(root / 'base')['success'])
            with self.assertRaisesRegex(ValueError, 'dependency closure changed'):
                native_dependencies(app, json.loads((app / 'package.json').read_text()), 'x86_64-unknown-linux-gnu', 64)
            base_metadata = json.loads((root / 'base/package.pkg/metadata/realizations/dev.json').read_text())
            obj = root / 'base/package.pkg' / base_metadata['native-artifacts'][0]['path']
            obj.write_bytes(obj.read_bytes() + b'corrupt')
            left = root / 'left'
            with self.assertRaisesRegex(ValueError, 'object checksum mismatch'):
                native_dependencies(left, json.loads((left / 'package.json').read_text()), 'x86_64-unknown-linux-gnu', 64)

    def test_dependency_closure_rejects_conflicting_package_roots(self):
        first = {'root': '/first', 'package': {'name': 'same'}, 'initializers': []}
        second = {**first, 'root': '/second'}
        with self.assertRaisesRegex(ValueError, 'conflicting local package identity'):
            dependency_closure({'a': {'closure': [first]}, 'b': {'closure': [second]}})
        self.assertEqual(dependency_closure({'a': {'closure': [first]}, 'b': {'closure': [first]}}), [first])

    def test_only_publication_allocates_a_build_id(self):
        from unittest.mock import patch
        with patch(__name__ + '.new_build_id', return_value='2026_10_11_12_34_56') as generate:
            self.assertEqual(output_slot(), 'dev')
            self.assertEqual(output_slot(include_tests=True), 'test')
            generate.assert_not_called()
            self.assertEqual(output_slot(publish=True), '2026_10_11_12_34_56')
            generate.assert_called_once_with()

    def test_conflicting_dev_artifacts_are_purged_and_publications_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'package.pkg'
            old, published = '2026_10_10_01_02_03', '2026_10_11_01_02_03'
            entries = [{'build-id': old, 'path': f'package.pkgi/severian/{old}/lib.sevi'},
                       {'build-id': None, 'output-slot': 'dev'},
                       {'build-id': None, 'output-slot': 'test'},
                       {'build-id': published, 'output-slot': published, 'published': True}]
            write_atomic(output / 'package.pkgi/index.json', encode({'interfaces': entries}))
            for slot in (old, published, 'dev', 'test'):
                write_atomic(output / f'artifacts/target/{slot}/object/package.o', 'object')
            write_atomic(output / 'source/src/removed.sev', 'old source')
            prepare_output(output, 'dev')
            self.assertFalse((output / f'artifacts/target/{old}').exists())
            self.assertFalse((output / 'artifacts/target/dev').exists())
            self.assertFalse((output / 'source').exists())
            self.assertTrue((output / 'artifacts/target/test/object/package.o').is_file())
            self.assertTrue((output / f'artifacts/target/{published}/object/package.o').is_file())
            self.assertEqual(json.loads((output / 'package.pkgi/index.json').read_text())['interfaces'], entries[2:])
            with self.assertRaisesRegex(ValueError, 'publication already exists'):
                prepare_output(output, published, publish=True)

    def test_repeated_development_builds_reuse_the_slot(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(encode({'package': {'name': 'stable', 'version': '0.1.0'}}))
            (root / 'src/lib.sev').write_text('def answer() -> int:\n    return 42\n')
            with patch(__name__ + '.new_build_id') as generate:
                first = build(root, stop_after='mir')
                (root / 'src/lib.sev').write_text('def answer() -> int:\n    return 43\n')
                second = build(root, stop_after='mir')
                generate.assert_not_called()
            self.assertTrue(first['success'] and second['success'])
            self.assertIsNone(second['build-id'])
            self.assertEqual(first['metadata'], second['metadata'])
            index = json.loads((root / 'package.pkg/package.pkgi/index.json').read_text())
            self.assertEqual(len(index['interfaces']), 1)
            self.assertEqual(index['interfaces'][0]['output-slot'], 'dev')

    def test_build_id_format_is_utc_and_zero_padded(self):
        from datetime import timedelta
        self.assertEqual(new_build_id(datetime(2026, 1, 2, 3, 4, 5, 123456, timezone.utc)),
                         '2026_01_02_03_04_05')
        self.assertEqual(new_build_id(datetime(2026, 1, 1, 22, 4, 5,
                                               tzinfo=timezone(timedelta(hours=-5)))),
                         '2026_01_02_03_04_05')

    def test_hash_checkpoint_layout_has_three_realizations_and_no_native_artifacts(self):
        from unittest.mock import patch
        from py_compiler.hir.thir.src.lib import ThirTests
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
            output, build_id = root / 'package.pkg', result['output-slot']
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
                first, second = build(root, publish=True), build(root, publish=True)
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


class PartitionArtifactTests(unittest.TestCase):
    def test_imported_lifetime_change_invalidates_native_cache_without_mlir_change(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.lir.src.lib import lower
        package = {'package': {'name': 'lifetimes'}, 'objects': ['lifetimes.o'], 'types': [],
                   'exports': [{'name': 'choose', 'symbol': 'choose', 'receiver': None,
                     'parameters': [{'name': name, 'type': 'array[int]', 'ownership': 'view'}
                                    for name in ('a', 'b')],
                     'result': 'array[int]', 'result-ownership': 'view', 'result-sources': [0]}]}
        source = ('import lifetimes\ndef work(a: array[int], b: array[int]) -> int:\n'
                  '    return len(lifetimes.choose(a, b))\n'
                  'def unrelated() -> int:\n    return 42\n')
        target = 'x86_64-unknown-linux-gnu'
        with tempfile.TemporaryDirectory() as directory:
            def compile_consumer():
                result = compile_source('consumer.sev', source, Syntax(),
                                        native_packages={'lifetimes': json.loads(json.dumps(package))})
                self.assertFalse(result.diagnostics, str(result.diagnostics))
                lir = lower(result.program, target, 64)
                records, _, _, _ = compile_partitions(result.program, lir, Path(directory), target,
                                                       'dev', 'contract-regression', None)
                return lir.text(), {record['symbols']: record for record in records}
            before_text, before = compile_consumer()
            _, repeated = compile_consumer()
            self.assertTrue(all(record['reused'] for record in repeated.values()))
            package['exports'][0]['result-sources'] = [1]
            after_text, after = compile_consumer()
            self.assertEqual(before_text, after_text)
            changed = [symbols for symbols in before if before[symbols]['cache-key'] != after[symbols]['cache-key']]
            self.assertEqual(len(changed), 1)
            self.assertTrue(any('__sev_fn_work' in symbol for symbol in changed[0]))
            self.assertFalse(after[changed[0]]['reused'])
            self.assertTrue(all(record['reused'] for symbols, record in after.items() if symbols not in changed))

    def fixture(self, directory, source):
        root = Path(directory)
        (root / 'src').mkdir()
        (root / 'package.json').write_text(encode({'package': {'name': 'partitioned', 'version': '0.1.0'}}))
        (root / 'src/lib.sev').write_text(source)
        return root

    def test_native_submodules_reuse_unchanged_objects_within_one_source(self):
        from unittest.mock import patch
        source = ('def first() -> int:\n    return 41\n'
                  'def second() -> int:\n    return first() + 1\n')
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, source)
            first = build(root)
            self.assertTrue(first['success'], first)
            self.assertEqual(len(first['objects']), 3)
            self.assertTrue(any('/first-' in p or '/first_' in p for p in first['objects']))
            self.assertTrue(all(Path(p).name != 'package.o' for p in first['objects']))
            times = {p: Path(p).stat().st_mtime_ns for p in first['objects']}
            with patch(__name__ + '.compile_object', wraps=compile_object) as compile_native:
                second = build(root)
                compile_native.assert_not_called()
            self.assertTrue(all(p['reused'] for p in second['partitions']))
            self.assertEqual(times, {p: Path(p).stat().st_mtime_ns for p in second['objects']})
            (root / 'src/lib.sev').write_text(source.replace('return 41', 'return 42'))
            with patch(__name__ + '.compile_object', wraps=compile_object) as compile_native:
                changed = build(root)
                self.assertEqual(compile_native.call_count, 1)
            self.assertEqual(sum(not p['reused'] for p in changed['partitions']), 1)
            metadata = json.loads((root / 'package.pkg/metadata/realizations/dev.json').read_text())
            self.assertEqual(len(metadata['modules']), 1)
            self.assertEqual(len(metadata['native-artifacts']), 3)
            for part in metadata['partitions']:
                self.assertTrue(part['symbols'])
                self.assertTrue(all((root / 'package.pkg' / path).is_file() for path in part['artifacts']))
            # Corrupt cache entries are rebuilt rather than trusted.
            part = changed['partitions'][0]
            (root / 'package.pkg/cache/native' / part['cache-key'] / 'native.o').write_bytes(b'broken')
            repaired = build(root)
            self.assertEqual(sum(not p['reused'] for p in repaired['partitions']), 1)
            (root / 'src/lib.sev').write_text('def first() -> int:\n    return 42\n')
            removed = build(root)
            self.assertEqual(len(removed['objects']), 2)
            self.assertTrue(all(not Path(path).exists() for path in set(repaired['objects']) - set(removed['objects'])))

    def test_partitioned_tests_link_only_reachable_local_objects(self):
        from py_compiler.testing import test_package
        source = ('counter = 0\ndef used() -> int:\n    module.counter += 1\n    return module.counter\n'
                  'def unused() -> int:\n    return 99\n'
                  'test "partition linkage":\n    assert(used() == 1)\n')
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, source)
            result = test_package(root)
            self.assertTrue(result['success'], result)
            report = json.loads(Path(result['report']).read_text())
            objects = report['tests'][0]['objects']
            self.assertFalse(any('/unused' in path for path in objects))
            self.assertTrue(any('/used' in path for path in objects))
            self.assertTrue(any('/initializer' in path for path in objects))

    def test_recursive_scc_and_generic_realizations_have_distinct_artifacts(self):
        source = ('def even(n: int) -> bool:\n    if n == 0:\n        return true\n    return odd(n - 1)\n'
                  'def odd(n: int) -> bool:\n    if n == 0:\n        return false\n    return even(n - 1)\n'
                  'def identity[T](value: T) -> T:\n    return value\n'
                  'test "realizations":\n    assert(even(4))\n    assert(identity[i32](3) == 3)\n'
                  '    assert(identity[i64](4) == 4)\n')
        from py_compiler.testing import test_package
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, source)
            result = test_package(root)
            self.assertTrue(result['success'], result)
            metadata = json.loads((root / 'package.pkg/metadata/realizations/test.json').read_text())
            recursion = [p for p in metadata['partitions'] if any('__sev_fn_even' in symbol for symbol in p['symbols'])]
            self.assertEqual(len(recursion), 1)
            self.assertTrue(any('__sev_fn_odd' in symbol for symbol in recursion[0]['symbols']))
            generics = [p for p in metadata['partitions'] if any('__sev_fn_identity[' in symbol for symbol in p['symbols'])]
            self.assertEqual(len(generics), 2)
            self.assertNotEqual(generics[0]['object'], generics[1]['object'])
