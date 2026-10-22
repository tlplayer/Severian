"""Compile a source file into a validated development executable."""
import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
import tempfile

from py_compiler.frontend.src.lib import compile_source
from py_compiler.lir.src.lib import lower, compile_object, companion
from py_compiler.mlir.src.lib import symbol_ref, verifier_path
from py_compiler.packaging.package import compiler_identity, digest, encode, c_dependencies, write_atomic, write_generated
from py_compiler.packaging.testing import build_package, TestEntry
from py_compiler.syntax.recognition import Syntax


TARGET = 'x86_64-unknown-linux-gnu'


class DemandPackages(dict):
    """Context-local imported names with shared, demand-loaded root dependencies."""
    def __init__(self, root, manifest, *, resolved=None, built=None, **options):
        super().__init__()
        self.root, self.manifest, self.options = Path(root), manifest, options
        self.manifest.setdefault('dependencies', {})
        self.resolved = {} if resolved is None else resolved
        self.built = {} if built is None else built

    def copy(self):
        result = DemandPackages(self.root, self.manifest, resolved=self.resolved,
                                built=self.built, **self.options)
        result.update(self)
        return result

    def get(self, alias, default=None):
        if alias not in self and alias in self.manifest.get('dependencies', {}):
            if alias not in self.resolved:
                selected = {'dependencies': {alias: self.manifest['dependencies'][alias]}}
                self.resolved.update(prepare_dependencies(self.root, selected, built=self.built, **self.options))
            self[alias] = self.resolved[alias]
        return super().get(alias, default)

    def resolve_prelude(self, name):
        if name == self.manifest.get('package', {}).get('name'):
            return None
        for alias, spec in self.manifest.get('dependencies', {}).items():
            identity = spec.get('package')
            if identity is None and 'path' in spec:
                path = self.root / spec['path'] / 'package.json'
                if path.is_file():
                    identity = json.loads(path.read_text()).get('package', {}).get('name')
            if identity == name:
                return alias, self.get(alias)
        alias = {'core.string': 'string', 'system.io': 'io'}.get(name, name.replace('.', '_'))
        while alias in self.manifest['dependencies']:
            alias = '__prelude_' + alias
        self.manifest['dependencies'][alias] = {'package': name, 'version': '*'}
        return alias, self.get(alias)


def source_files(root, manifest):
    selected = manifest.get('sources')
    return [root / name for name in selected] if selected is not None else sorted((root / 'src').rglob('*.sev'))


def prepare_dependencies(root, manifest, *, target=TARGET, pointer_bits=64, jobs=4,
                         mlir_opt=None, built=None, progress=None, registry_cache=None):
    """Reuse checked artifacts only while their manifest and sources still match."""
    built, active = {} if built is None else built, set()
    compiler_id = compiler_identity()
    from py_compiler.packaging.registry import dependency_root
    registry_cache = registry_cache or root / "package.pkg/cache/registry"

    def visit(path):
        path = path.resolve()
        if path in active:
            raise ValueError('cyclic package dependency: ' + str(path))
        if path in built:
            if built[path] is None:
                raise ValueError('cyclic package dependency: ' + str(path))
            return
        active.add(path)
        if progress:
            progress('Checking dependency: ' + str(path))
        current = json.loads((path / 'package.json').read_text())
        metadata_path = path / 'package.pkg/metadata/realizations/dev.json'
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text())
            for specification in previous.get('dependency-specifications', current.get('dependencies', {})).values():
                visit(dependency_root(path, specification, registry_cache))
        valid = False
        manifest_text = (path / 'package.json').read_text()
        snapshots = [(source.relative_to(path).as_posix(), source.read_bytes().decode('utf-8'))
                     for source in source_files(path, current)]
        lock_path = path / 'package.lock'
        lock = lock_path.read_text() if lock_path.exists() else encode({'version': 1, 'packages': []})
        content_id = digest(encode([manifest_text, lock, snapshots]))
        foreign = [(name, sha256((path / name).read_bytes()).hexdigest())
                   for provider in current.get('xxi', {}).values() for name in provider.get('sources', ())]
        marker = path / 'package.pkg/build/dev/.source-command-dependencies.json'
        freshness = {'compiler': compiler_id, 'content-id': content_id, 'foreign': foreign}
        try:
            checked = c_dependencies(root, {'dependencies': {'check': {'path': str(path)}}},
                                          target, pointer_bits, _compiler_id=compiler_id, _registry_cache=registry_cache)['check']
            valid = checked['package'].get('content-id') == content_id
            if foreign:
                valid = valid and marker.read_text() == encode(freshness)
        except (OSError, ValueError, KeyError):
            pass
        if not valid:
            if progress:
                progress('Building dependency: ' + str(path))
            built[path] = None
            try:
                result = build_package(path, _built=built, _registry_cache=registry_cache, target=target, pointer_bits=pointer_bits,
                                       jobs=jobs, mlir_opt=mlir_opt)
            finally:
                if built.get(path) is None:
                    built.pop(path, None)
            if not result['success']:
                raise ValueError('\n'.join(result.get('sample', ())) or 'dependency build failed: ' + str(path))
        elif progress:
            progress('Reusing dependency: ' + str(path))
        if foreign:
            write_atomic(marker, encode(freshness))
        built[path] = {'success': True}
        active.remove(path)

    for specification in manifest.get('dependencies', {}).values():
        visit(dependency_root(root, specification, registry_cache))
    return c_dependencies(root, manifest, target, pointer_bits, _compiler_id=compiler_id, _registry_cache=registry_cache)


def source_package(source):
    root = next((parent for parent in source.parents if (parent / 'package.json').is_file()), source.parent)
    manifest_path = root / 'package.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest = {**manifest, 'dependencies': dict(manifest.get('dependencies', {}))}
    return root, manifest


def entry_source(initializers, main):
    calls = list(dict.fromkeys(initializers))
    declarations = [f'func.func private {symbol_ref(symbol)}()' for symbol in calls]
    instructions = [f'func.call {symbol_ref(symbol)}() : () -> ()' for symbol in calls]
    if main is not None:
        if main.blocks[0].parameters:
            raise ValueError('main must take no parameters; read process arguments through the CLI library')
        result = main.result_type
        if result.mlir and result.family != 'integer':
            raise ValueError('main must return an integer exit status or no value')
        suffix = ' -> ' + result.mlir if result.mlir else ''
        declarations.append(f'func.func private {symbol_ref(main.name)}(){suffix}')
        instructions.append(('%status = ' if result.mlir else '') +
                            f'func.call {symbol_ref(main.name)}() : () -> {result.mlir or "()"}')
        if result.mlir:
            if result.bits == 32:
                instructions.append('func.return %status : i32')
            else:
                opcode = 'trunci' if result.bits > 32 else ('extsi' if result.signed else 'extui')
                instructions += [f'%exit = arith.{opcode} %status : {result.mlir} to i32',
                                 'func.return %exit : i32']
    if main is None or not main.result_type.mlir:
        instructions += ['%exit = arith.constant 0 : i32', 'func.return %exit : i32']
    return ('module {\n' + '\n'.join(declarations) + '\nfunc.func @main() -> i32 {\n' +
            '\n'.join(instructions) + '\n}\n}\n')


def link_package(program, partitions, output, manifest, slot, target, pointer_bits,
                 dependencies, initializers, *, destination=None, optimizer=None):
    """Link declared binaries (or an inferred main) from the completed partitions."""
    import re
    from py_compiler.syntax.complex.profile import linker_options
    mains = [body for body in program.bodies if body.declaration == 'main' and not body.test]
    declarations = manifest.get('bin')
    if declarations is None:
        if len(mains) > 1:
            raise ValueError('multiple main functions require explicit bin entries in package.json')
        declarations = [{'name': manifest['package']['name'], 'path': mains[0].source}] if mains else []
    if not isinstance(declarations, list):
        raise ValueError('bin must be a list of executable declarations')
    if destination is not None and len(declarations) != 1:
        raise ValueError('-o requires exactly one binary target')
    plans, names = [], set()
    for declaration in declarations:
        name, source = declaration.get('name'), declaration.get('path')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name) or name in names:
            raise ValueError('binary targets require unique, safe names')
        names.add(name)
        if not isinstance(source, str) or Path(source).is_absolute() or '..' in Path(source).parts:
            raise ValueError('binary source must be package-relative')
        selected = [body for body in mains if body.source == Path(source).as_posix()]
        if len(selected) != 1:
            raise ValueError(f'binary {name!r} requires exactly one main function in {source}')
        path = Path(destination).resolve() if destination is not None else output / 'build' / slot / (name if name.endswith('.exe') else name + '.exe')
        if path.suffix == '.sev' or path.name in ('package.json', 'package.lock'):
            raise ValueError('executable output must not overwrite package source or configuration')
        plans.append((name, selected[0], path))
    by_module = {part['submodule']: part for part in partitions}
    owners = {symbol: key for key, part in by_module.items() for symbol in part['symbols']}
    binary_sources = {main.source for _, main, _ in plans}
    optimizer = optimizer or (verifier_path() if plans else None)
    binaries = []
    for name, main, path in plans:
        local_initializers = [body.name for body in program.bodies if not body.declaration
                              and (body.source not in binary_sources or body.source == main.source)]
        entry = entry_source([*initializers, *local_initializers], main)
        pending = [owners[symbol] for symbol in [main.name, *local_initializers] if symbol in owners]
        selected = set()
        while pending:
            owner = pending.pop()
            if owner not in selected:
                selected.add(owner)
                pending.extend(by_module[owner]['dependencies'])
        objects = list(dict.fromkeys([*(str(output / part['object']) for key, part in by_module.items() if key in selected),
                                      *dependencies]))
        linker = companion('clang', optimizer)
        identity = digest(encode([compiler_identity(), target, pointer_bits, entry,
                                  [(obj, sha256(Path(obj).read_bytes()).hexdigest()) for obj in objects],
                                  str(Path(linker).resolve()), Path(linker).stat().st_mtime_ns]))
        cache = output / 'cache/executable' / identity
        binary = cache / 'program.exe'
        reused = False
        if binary.is_file() and (cache / 'checksum').is_file():
            reused = sha256(binary.read_bytes()).hexdigest() == (cache / 'checksum').read_text()
        if not reused:
            wrapper = compile_object(TestEntry(entry, target, pointer_bits), optimizer)
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.link-', dir=cache) as temporary:
                temporary = Path(temporary)
                driver, linked = temporary / 'entry.o', temporary / 'program.exe'
                driver.write_bytes(wrapper.object_bytes)
                result = subprocess.run([linker, '--target=' + target, str(driver), *objects,
                                         *linker_options(objects), '-pthread', '-lm', '-o', str(linked)],
                                        capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise ValueError(f'link failed for {name}:\n' + result.stderr)
                os.replace(linked, binary)
            write_atomic(cache / 'checksum', sha256(binary.read_bytes()).hexdigest())
        relative = f'artifacts/{target}/{slot}/bin/{name}.exe'
        artifact = output / relative
        data = binary.read_bytes()
        for location in dict.fromkeys((artifact, path)):
            write_generated(location, data)
            location.chmod(0o755)
        binaries.append({'name': name, 'path': relative, 'executable': str(path), 'reused': reused,
                         'checksum': sha256(data).hexdigest(), 'entry': main.name})
    return binaries


def build_source(source, output=None):
    source = Path(source).resolve(strict=True)
    if source.suffix != '.sev' or not source.is_file():
        raise ValueError('source must be a .sev file')
    root, manifest = source_package(source)
    destination = Path(output).resolve() if output is not None else root / 'package.pkg/build/dev' / (source.stem + '.exe')
    if destination == source or destination.suffix == '.sev':
        raise ValueError('executable output must not overwrite Severian source')
    packages = DemandPackages(root, manifest)
    # Local source imports and foreign providers participate in executable reuse.
    snapshots = [(str(path.relative_to(root)), sha256(path.read_bytes()).hexdigest())
                 for path in sorted(root.rglob('*.sev'))
                 if not any(part.endswith('.pkg') for part in path.relative_to(root).parts)]
    for provider in manifest.get('xxi', {}).values():
        for name in provider.get('sources', ()):
            path = root / name
            snapshots.append((name, sha256(path.read_bytes()).hexdigest()))
    source_identity = digest(encode([compiler_identity(), str(source), manifest, snapshots, TARGET]))
    metadata = root / 'package.pkg/build/dev/.metadata' / (digest(str(destination)) + '.json')
    if metadata.is_file() and destination.is_file() and os.access(destination, os.X_OK):
        try:
            saved = json.loads(metadata.read_text())
        except (OSError, ValueError):
            saved = {}
        if saved.get('source-identity') == source_identity:
            for alias, spec in saved.get('dependency-specifications', {}).items():
                manifest['dependencies'].setdefault(alias, spec)
            for alias in saved.get('dependencies', ()):
                packages.get(alias)
            identity = digest(encode([source_identity, packages.resolved]))
            if saved.get('identity') == identity and saved.get('checksum') == sha256(destination.read_bytes()).hexdigest():
                return {'executable': str(destination), 'reused': True}
    result = compile_source(str(source.relative_to(root)), source.read_text(),
                            Syntax(64, include_tests=False, target=TARGET),
                            c_packages=packages, debug_path=str(source))
    if result.diagnostics:
        raise ValueError('\n\n'.join(map(str, result.diagnostics)))
    from py_compiler.packaging.registry import record_lock
    record_lock(root, manifest['dependencies'])
    packages = packages.resolved
    identity = digest(encode([source_identity, packages]))
    mains = [body for body in result.program.bodies if body.declaration == 'main']
    if len(mains) > 1:
        raise ValueError('source has more than one main function')
    initializers = [symbol for package in packages.values() for symbol in package['initializers']]
    initializers.extend(body.name for body in result.program.bodies if not body.declaration)
    entry = entry_source(initializers, mains[0] if mains else None)
    artifact = compile_object(lower(result.program, TARGET, 64))
    wrapper = compile_object(TestEntry(entry, TARGET, 64))
    dependencies = list(dict.fromkeys([*(obj for package in packages.values() for obj in package['objects']),
                                       *artifact.dependencies]))
    from py_compiler.syntax.complex.profile import linker_options
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sev-link-', dir=destination.parent) as directory:
        directory = Path(directory)
        obj, driver, binary = directory / 'source.o', directory / 'entry.o', directory / 'program.exe'
        obj.write_bytes(artifact.object_bytes)
        driver.write_bytes(wrapper.object_bytes)
        linked = subprocess.run([companion('clang', verifier_path()), str(driver), str(obj), *dependencies,
                                 *linker_options(artifact.dependencies), '-lm', '-o', str(binary)],
                                capture_output=True, text=True, timeout=30)
        if linked.returncode:
            raise ValueError('link failed:\n' + linked.stderr)
        os.replace(binary, destination)
    write_atomic(metadata, encode({'identity': identity, 'source-identity': source_identity,
                                   'dependencies': list(packages),
                                   'dependency-specifications': {alias: manifest['dependencies'][alias] for alias in packages}, 'checksum': sha256(destination.read_bytes()).hexdigest(),
                                   'source': str(source), 'executable': str(destination)}))
    return {'executable': str(destination), 'reused': False}


def run_source(source, output=None, arguments=(), *, run=True):
    result = build_source(source, output)
    if not run:
        return 0
    status = subprocess.call([result['executable'], *arguments])
    return status if status >= 0 else 128 - status


import unittest


class DependencyPreparationTests(unittest.TestCase):
    def test_selected_example_does_not_prepare_unrelated_packages(self):
        from unittest.mock import patch
        root = Path(__file__).resolve().parents[2]
        for name in ('01-inferred-field-mutation.sev', '02-read-write-fields.sev'):
            source = root / 'docs/examples/01-types/02-classes' / name
            package_root, manifest = source_package(source)
            packages = DemandPackages(package_root, manifest)
            with patch(__name__ + '.prepare_dependencies', side_effect=AssertionError('unused dependency')) as prepare:
                result = compile_source(name, source.read_text(), Syntax(), c_packages=packages)
            self.assertFalse(result.diagnostics, result.diagnostics)
            self.assertEqual(packages.resolved, {})
            prepare.assert_not_called()

    def test_manifest_providers_are_available_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'package.json').write_text(encode({'dependencies': {
                alias: {'path': alias} for alias in ('one', 'two', 'three', 'unused')}}))
            source = root / 'selected.sev'
            source.write_text('import one\nfrom two import value\nimport value from three\n'
                              '# import unused\ntest:\n    assert(true)\n')
            _, manifest = source_package(source)
            packages = DemandPackages(root, manifest)
            self.assertEqual(set(manifest['dependencies']), {'one', 'two', 'three', 'unused'})
            self.assertEqual(packages.resolved, {})

    def test_explicit_renaming_and_primitive_string_storage_need_no_libraries(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'local.sev'
            source.write_text('print() as output()\ndef print(value: int) -> int:\n    return value + 1\n'
                              'test:\n    assert(print(2) == 3)\n    text = "hello"\n    assert(len(text) == 5)\n')
            root, manifest = source_package(source)
            packages = DemandPackages(root, manifest)
            with patch(__name__ + '.prepare_dependencies', side_effect=AssertionError('unused dependency')):
                result = compile_source(source.name, source.read_text(), Syntax(), c_packages=packages)
            self.assertFalse(result.diagnostics, result.diagnostics)
            self.assertEqual(packages.resolved, {})

    def test_operations_resolve_only_their_providers(self):
        from unittest.mock import patch
        from py_compiler.syntax.prelude import PreludeTests
        io = PreludeTests().library_packages()['console']
        strings = io['template-dependencies']['text']
        for text, alias, package in (
            ('test:\n    print("hello")\n', 'io', io),
            ('test:\n    assert("a" == "a")\n', 'string', strings),
        ):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'use.sev'
                source.write_text(text)
                root, manifest = source_package(source)
                packages = DemandPackages(root, manifest)
                with patch(__name__ + '.prepare_dependencies', return_value={alias: package}) as prepare:
                    result = compile_source(source.name, text, Syntax(), c_packages=packages)
                self.assertFalse(result.diagnostics, result.diagnostics)
                self.assertEqual(set(packages.resolved), {alias})
                self.assertEqual(prepare.call_count, 1)
                self.assertEqual(set(prepare.call_args.args[1]['dependencies']), {alias})

    def test_excluded_test_scopes_do_not_load_prelude(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'test-only.sev'
            source.write_text('test:\n    print("test only")\ndef value() -> int:\n    return 1\n')
            root, manifest = source_package(source)
            with patch(__name__ + '.prepare_dependencies', side_effect=AssertionError('excluded dependency')):
                result = compile_source(source.name, source.read_text(), Syntax(include_tests=False),
                                        c_packages=DemandPackages(root, manifest))
            self.assertFalse(result.diagnostics, result.diagnostics)

    def test_checked_dependencies_reuse_and_source_edits_rebuild(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dep = root / 'dep'
            dep.mkdir()
            text = encode({'package': {'name': 'dep', 'version': '1'}, 'sources': ['lib.sev']})
            (dep / 'package.json').write_text(text)
            source = dep / 'lib.sev'
            source.write_text('def value() -> int:\n    return 1\n')
            content_id = digest(encode([text, encode({'version': 1, 'packages': []}),
                                        [('lib.sev', source.read_text())]]))
            manifest = {'dependencies': {'dep': {'path': 'dep'}}}
            checked = {'package': {'content-id': content_id}}
            def artifacts(root, manifest, target, pointer_bits, **kwargs):
                self.assertEqual((target, pointer_bits), ('aarch64-unknown-linux-gnu', 64))
                return {alias: checked for alias in manifest['dependencies']}
            with patch(__name__ + '.c_dependencies', side_effect=artifacts):
                with patch(__name__ + '.build_package', return_value={'success': True}) as builder:
                    built, messages = {}, []
                    prepare_dependencies(root, manifest, target='aarch64-unknown-linux-gnu',
                                         built=built, progress=messages.append)
                    builder.assert_not_called()
                    self.assertIn(dep, built)
                    self.assertTrue(any(message.startswith('Reusing') for message in messages))
                    source.write_text('def value() -> int:\n    return 2\n')
                    prepare_dependencies(root, manifest, target='aarch64-unknown-linux-gnu')
                    builder.assert_called_once()
                    self.assertEqual(builder.call_args.kwargs['target'], 'aarch64-unknown-linux-gnu')


class ExecutableTests(unittest.TestCase):
    def test_standalone_prelude_default_output_and_verified_reuse(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix='sev source ') as directory:
            root = Path(directory)
            source = root / 'hello.sev'
            source.write_text('def main():\n    print("hello " + string(42))\n')
            result = build_source(source)
            binary = root / 'package.pkg/build/dev/hello.exe'
            self.assertEqual(Path(result['executable']), binary)
            executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertEqual(executed.stdout, 'hello 42\n')
            with patch(__name__ + '.compile_source', side_effect=AssertionError('unexpected recompilation')):
                self.assertTrue(build_source(source)['reused'])
            source.write_text('def main():\n    print("changed")\n')
            self.assertFalse(build_source(source)['reused'])
            self.assertEqual(subprocess.check_output([str(binary)], text=True, timeout=5), 'changed\n')

    def test_explicit_output_exit_status_and_failed_rebuild(self):
        with tempfile.TemporaryDirectory(prefix='sev executable ') as directory:
            root = Path(directory)
            source, output = root / 'status.sev', root / 'out space/custom.exe'
            source.write_text('def main() -> int:\n    return 7\n')
            self.assertEqual(run_source(source, output), 7)
            source.write_text('def main():\n    unresolved_name()\n')
            with self.assertRaises(ValueError):
                run_source(source, output)
            self.assertTrue(output.is_file())

    def test_script_initializers_run_without_main(self):
        with tempfile.TemporaryDirectory(prefix='sev script ') as directory:
            source = Path(directory) / 'script.sev'
            source.write_text('print("top level")\n')
            result = build_source(source)
            self.assertEqual(subprocess.check_output([result['executable']], text=True, timeout=5), 'top level\n')


class PackageExecutableTests(unittest.TestCase):
    def package(self, root, name, source, **manifest):
        (root / 'src').mkdir(parents=True)
        (root / 'package.json').write_text(encode({'package': {'name': name, 'version': '1.0.0'}, **manifest}))
        (root / 'src/main.sev').write_text(source)

    def test_build_links_declared_main_with_initializers_and_reuses_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.package(root / 'library', 'library', 'value = 40\ndef answer() -> int:\n    return module.value\n')
            app = root / 'app'
            self.package(app, 'application', 'import library\nvalue = 2\ndef main() -> int:\n    return library.answer() + module.value\n',
                         dependencies={'library': {'path': '../library'}}, bin=[{'name': 'demo', 'path': 'src/main.sev'}])
            result = build_package(app)
            self.assertTrue(result['success'], result)
            executable = app / 'package.pkg/build/dev/demo.exe'
            self.assertEqual(result['executables'], [str(executable)])
            self.assertEqual(subprocess.run([str(executable)], timeout=5).returncode, 42)
            self.assertTrue(os.access(executable, os.X_OK))
            build_package(app)
            metadata = json.loads((app / 'package.pkg/metadata/realizations/dev.json').read_text())
            self.assertTrue(metadata['executables'][0]['reused'])
            custom = root / 'sort.exe'
            result = build_package(app, executable_output=custom)
            self.assertEqual(result['executables'], [str(custom)])
            self.assertEqual(subprocess.run([str(custom)], timeout=5).returncode, 42)

    def test_inferred_main_and_library_only_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.package(root / 'app', 'inferred', 'def main() -> int:\n    return 7\n')
            result = build_package(root / 'app')
            self.assertEqual(subprocess.run(result['executables'], timeout=5).returncode, 7)
            self.package(root / 'lib', 'library', 'def answer() -> int:\n    return 42\n')
            self.assertEqual(build_package(root / 'lib')['executables'], [])
            with self.assertRaisesRegex(ValueError, 'exactly one binary'):
                build_package(root / 'lib', executable_output=root / 'invalid.exe')

    def test_multiple_bins_select_their_main_and_reject_ambiguous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.package(root, 'multiple', 'def main() -> int:\n    return 3\n',
                         bin=[{'name': 'first', 'path': 'src/main.sev'}, {'name': 'second', 'path': 'src/second.sev'}])
            (root / 'src/second.sev').write_text('def main() -> int:\n    return 9\n')
            result = build_package(root)
            self.assertEqual([subprocess.run([path], timeout=5).returncode for path in result['executables']], [3, 9])
            with self.assertRaisesRegex(ValueError, 'exactly one binary'):
                build_package(root, executable_output=root / 'ambiguous.exe')

    def test_output_cannot_overwrite_source_and_bin_requires_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.package(root, 'app', 'def main():\n    return\n')
            source = root / 'src/main.sev'
            original = source.read_text()
            with self.assertRaisesRegex(ValueError, 'must not overwrite'):
                build_package(root, executable_output=source)
            self.assertEqual(source.read_text(), original)
            self.package(root / 'missing', 'missing', 'def other():\n    return\n',
                         bin=[{'name': 'missing', 'path': 'src/main.sev'}])
            with self.assertRaisesRegex(ValueError, 'requires exactly one main'):
                build_package(root / 'missing')
