"""Local SIP-0003 publications and relocatable source/c realizations."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from hashlib import sha256

from py_compiler.packaging.package import encode, digest, write_atomic, compiler_identity


def registry_root():
    return Path(os.environ.get('SEVERIAN_REGISTRY') or
                Path(os.environ.get('SEVERIAN_HOME') or Path.home() / '.severian') / 'packages/registry').resolve()


def safe_component(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('invalid registry package name or version: ' + repr(value))
    return value


def inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('publication path must be package-relative: ' + repr(relative))
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('publication path escapes its root: ' + relative)
    return path


def release(specification):
    name = safe_component(specification.get('package'))
    selector = specification.get('version', '*')
    directory = registry_root() / name
    versions = [path.name for path in directory.iterdir() if path.is_dir()
                and re.fullmatch(r'\d+\.\d+\.\d+', path.name)
                and (selector in ('*', 'latest') or path.name == selector or path.name.startswith(selector + '.'))] if directory.is_dir() else []
    if not versions:
        raise ValueError(f'no registry release for {name}@{selector}; publish this provider with sev publish --local')
    version = max(versions, key=lambda value: tuple(map(int, value.split('.'))))
    version_root = directory / version
    selected = specification.get('publication')
    if selected is None:
        selected = json.loads((version_root / 'index.json').read_text())['package']
    result = inside(version_root, selected)
    metadata = json.loads((result / 'metadata/publication.json').read_text())
    if metadata.get('format') != 'py_compiler.publication':
        raise ValueError('publication requires a compatible Python compiler source distribution: ' + str(result))
    if metadata['package']['name'] != name or metadata['package']['version'] != version:
        raise ValueError('registry package identity mismatch')
    if specification.get('source', metadata['source']) != metadata['source']:
        raise ValueError('registry source identity does not match package.lock')
    for relative, checksum in metadata['files'].items():
        if sha256(inside(result, relative).read_bytes()).hexdigest() != checksum:
            raise ValueError('publication checksum mismatch: ' + relative)
    return result, metadata


def dependency_root(root, specification, cache=None):
    if not isinstance(specification, dict):
        raise ValueError('dependency must specify a package/version or local path')
    if 'path' in specification:
        return (root / specification['path']).resolve()
    requested = specification
    lock = root / 'package.lock'
    if 'publication' not in specification and lock.is_file():
        for entry in json.loads(lock.read_text()).get('packages', []):
            if entry.get('package') == specification.get('package'):
                selector = specification.get('version', '*')
                if selector in ('*', 'latest') or entry['version'] == selector or entry['version'].startswith(selector + '.'):
                    pin = dict(entry)
                    if 'publication' in pin and 'source' not in pin:
                        _, previous = release(pin)
                        pin.pop('publication')
                        pin['source'] = previous['source']
                    specification = {**specification, **pin}
                    break
    published, metadata = release(specification)
    requested.update({'package': metadata['package']['name'], 'version': metadata['package']['version'],
                      'source': metadata['source']})
    cache = Path(cache) if cache is not None else root / 'package.pkg/cache/registry'
    key = digest(str(published))
    destination = cache / key
    if not destination.is_dir():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.materialize-', dir=cache) as temporary:
            stage = Path(temporary) / 'package'
            shutil.copytree(published / 'source', stage)
            for directory in ('package.pkgi', 'metadata', 'artifacts'):
                if (published / directory).is_dir():
                    shutil.copytree(published / directory, stage / 'package.pkg' / directory)
            try:
                stage.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    return destination


def record_lock(root, specifications):
    path = root / 'package.lock'
    lock = json.loads(path.read_text()) if path.is_file() else {'version': 1, 'packages': []}
    pins = {entry['package']: entry for entry in lock.get('packages', []) if isinstance(entry.get('package'), str)}
    for spec in specifications.values():
        if 'source' in spec and 'path' not in spec:
            pins[spec['package']] = {key: spec[key] for key in ('package', 'version', 'source')}
    if pins:
        retained = [entry for entry in lock.get('packages', []) if not isinstance(entry.get('package'), str)]
        lock['packages'] = [*retained, *[pins[name] for name in sorted(pins)]]
        text = encode(lock)
        if not path.is_file() or path.read_text() != text:
            write_atomic(path, text)
    return path.read_text() if path.is_file() else encode(lock)


def source_files(root):
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in ('.git', '.codex', '.agents', '__pycache__')
                         and not name.endswith('.pkg'))
        for name in sorted(files):
            path = Path(directory) / name
            if path.name in ('package.lock', 'package.json'):
                continue
            relative = path.relative_to(root).as_posix()
            yield relative, inside(root, relative).read_bytes()


def publish_package(root, reference=None, *, target='x86_64-unknown-linux-gnu', pointer_bits=64, jobs=4,
                    mlir_opt=None, _active=()):
    """Freeze local edges, build the shipped source, then atomically install it."""
    from py_compiler.packaging.testing import build_package
    root = Path(root).resolve()
    if root in _active:
        raise ValueError('cyclic publication dependency: ' + str(root))
    manifest = json.loads((root / 'package.json').read_text())
    identity = manifest['package']
    name, version = (safe_component(identity[key]) for key in ('name', 'version'))
    if identity.get('publish') is False:
        raise ValueError('package is marked publish = false')
    if reference and reference not in (name, name + ':' + version, name + '@' + version):
        raise ValueError('publication reference must match package.json: ' + name + ':' + version)
    dependencies = {}
    for alias, spec in manifest.get('dependencies', {}).items():
        if 'path' in spec:
            dependency = (root / spec['path']).resolve()
            published = publish_package(dependency, target=target, pointer_bits=pointer_bits, jobs=jobs,
                                        mlir_opt=mlir_opt, _active=(*_active, root))
            spec = published['dependency']
        selected, record = release(spec)
        dependencies[alias] = {'package': record['package']['name'], 'version': record['package']['version'],
                               'source': record['source']}
    manifest['dependencies'] = dependencies
    manifest.pop('dev-dependencies', None)
    sources = dict(source_files(root))
    sources['package.json'] = encode(manifest).encode()
    sources['package.lock'] = encode({'version': 1, 'packages': list(dependencies.values())}).encode()
    identity_manifest = json.loads(encode(manifest))
    for spec in identity_manifest['dependencies'].values():
        _, dependency = release(spec)
        spec.pop('publication', None)
        spec['source'] = dependency['source']
    identity_sources = {**sources, 'package.json': encode(identity_manifest).encode()}
    source_id = digest(encode([(key, sha256(value).hexdigest()) for key, value in sorted(identity_sources.items())
                              if key != 'package.lock']))
    registry = registry_root()
    version_root = registry / name / version
    registry.mkdir(parents=True, exist_ok=True)
    # A different source cannot silently replace an existing version.
    index = version_root / 'index.json'
    if index.is_file():
        previous, record = release({'package': name, 'version': version})
        if record['source'] != source_id:
            raise ValueError('PackageVersionConflict: different source already owns ' + name + ':' + version)
        if record.get('compiler') == compiler_identity() and record.get('target') == target and record.get('pointer-bits') == pointer_bits:
            return {'success': True, 'path': str(previous), 'dependency': {'package': name, 'version': version,
                    'source': record['source']}}
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=registry) as temporary:
        stage = Path(temporary)
        project = stage / 'project'
        for relative, data in sources.items():
            path = inside(project, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        result = build_package(project, target=target, pointer_bits=pointer_bits, jobs=jobs, mlir_opt=mlir_opt)
        if not result['success']:
            raise ValueError('\n'.join(result['sample']))
        output = project / 'package.pkg'
        distribution = stage / 'package.pkg'
        distribution.mkdir()
        shutil.copytree(project, distribution / 'source', ignore=shutil.ignore_patterns('package.pkg'))
        for directory in ('package.pkgi', 'metadata', 'artifacts'):
            shutil.copytree(output / directory, distribution / directory)
        realization = distribution / 'metadata/realizations/dev.json'
        metadata = json.loads(realization.read_text())
        metadata['published'] = True
        # C link inputs travel with the publication; checkout paths never do.
        dependency_outputs = [Path(entry['root']) / 'package.pkg' for entry in metadata['dependency-closure']]
        metadata['link-artifacts'] = [entry for entry in metadata['link-artifacts']
                                      if not any(Path(entry['path']).is_relative_to(path) for path in dependency_outputs)]
        for entry in metadata['link-artifacts']:
            source = Path(entry['path'])
            relative = 'artifacts/' + target + '/dev/link/' + entry['checksum'] + '.o'
            destination = inside(distribution, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            entry['path'] = relative
        metadata['dependencies'] = [entry['path'] for entry in metadata['link-artifacts']]
        for entry in metadata['dependency-closure']:
            entry.pop('root', None)
        write_atomic(realization, encode(metadata))
        files = {path.relative_to(distribution).as_posix(): sha256(path.read_bytes()).hexdigest()
                 for path in sorted(distribution.rglob('*')) if path.is_file()}
        publication_id = digest(encode(files))
        record = {'format': 'py_compiler.publication', 'version': 1, 'package': {'name': name, 'version': version},
                  'source': source_id, 'compiler': metadata['compiler'], 'target': target,
                  'pointer-bits': pointer_bits, 'files': files}
        write_atomic(distribution / 'metadata/publication.json', encode(record))
        relative = f'sha256-{source_id}/realizations/sha256-{publication_id}/package.pkg'
        destination = inside(version_root, relative)
        with (registry / '.sev-publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if index.is_file():
                _, current = release({'package': name, 'version': version})
                if current['source'] != source_id:
                    raise ValueError('PackageVersionConflict: another revision already owns ' + name + ':' + version)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                distribution.rename(destination)
            write_atomic(index, encode({'format': 1, 'package': relative}))
    return {'success': True, 'path': str(destination),
            'dependency': {'package': name, 'version': version, 'source': source_id}}


import unittest


class RegistryTests(unittest.TestCase):
    def test_paths_cannot_escape_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../outside', '/tmp/outside'):
                with self.assertRaisesRegex(ValueError, 'package-relative'):
                    inside(root, name)
            (root / 'escape').symlink_to('/tmp')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                inside(root, 'escape/file')

    def test_publish_relocate_reuse_and_reject_modified_source(self):
        from unittest.mock import patch
        from py_compiler.packaging.executable import prepare_dependencies
        from py_compiler.packaging.testing import build_package
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider, child = root / 'provider', root / 'child'
            for path, name, text, dependencies in (
                (child, 'test.child', 'def answer() -> int:\n    return 41\n', {}),
                (provider, 'test.provider', 'import child\ndef answer() -> int:\n    return child.answer() + 1\n',
                 {'child': {'path': '../child'}}),
            ):
                (path / 'src').mkdir(parents=True)
                (path / 'src/lib.sev').write_text(text)
                (path / 'package.json').write_text(encode({'package': {'name': name, 'version': '1.0.0'},
                                                          'dependencies': dependencies}))
            with patch.dict(os.environ, {'SEVERIAN_REGISTRY': str(root / 'registry')}):
                published = publish_package(provider)
                self.assertEqual(publish_package(provider)['path'], published['path'])
                distribution = Path(published['path'])
                self.assertFalse(any((distribution / name).exists() for name in ('cache', 'debug', 'build')))
                frozen = json.loads((distribution / 'source/package.json').read_text())
                self.assertNotIn('path', frozen['dependencies']['child'])
                shutil.move(root / 'registry', root / 'relocated')
                # Both original source checkouts become unavailable.
                provider.rename(root / 'provider-hidden')
                child.rename(root / 'child-hidden')
                consumer = root / 'consumer'
                (consumer / 'src').mkdir(parents=True)
                manifest = {'package': {'name': 'consumer', 'version': '1.0.0'},
                            'dependencies': {'library': {'package': 'test.provider', 'version': '1.0.0'}}}
                (consumer / 'package.json').write_text(encode(manifest))
                (consumer / 'src/lib.sev').write_text('import library\ndef answer() -> int:\n    return library.answer()\n')
                with patch.dict(os.environ, {'SEVERIAN_REGISTRY': str(root / 'relocated')}):
                    with patch('py_compiler.packaging.executable.build_package', side_effect=AssertionError('rebuilt compatible publication')):
                        packages = prepare_dependencies(consumer, manifest)
                    self.assertEqual(packages['library']['package']['name'], 'test.provider')
                    self.assertTrue(build_package(consumer)['success'])
                    # A local bridge must also reuse this registry closure from a
                    # second consumer's cache without linking the first copy twice.
                    other = root / 'other'
                    (other / 'src').mkdir(parents=True)
                    (other / 'package.json').write_text(encode({'package': {'name': 'other', 'version': '1.0.0'},
                                                               'dependencies': {'bridge': {'path': '../consumer'}}}))
                    (other / 'src/lib.sev').write_text('import bridge\ntest:\n    assert(bridge.answer() == 42)\n')
                    from py_compiler.packaging.testing import test_package
                    self.assertTrue(test_package(other)['success'])
                    selected, _ = release({'package': 'test.provider', 'version': '1.0.0'})
                    (selected / 'source/src/lib.sev').write_text('tampered\n')
                    with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                        release({'package': 'test.provider', 'version': '1.0.0'})
                (root / 'provider-hidden/src/lib.sev').write_text('def changed():\n    return\n')
                # Conflict is checked on the leaf before any replacement is installed.
                with patch.dict(os.environ, {'SEVERIAN_REGISTRY': str(root / 'relocated')}):
                    (root / 'child-hidden/src/lib.sev').write_text('def answer() -> int:\n    return 0\n')
                    with self.assertRaisesRegex(ValueError, 'PackageVersionConflict'):
                        publish_package(root / 'child-hidden')

    def test_prelude_uses_identity_and_has_no_checkout_path(self):
        from unittest.mock import patch
        from py_compiler.packaging.executable import DemandPackages
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'package': {'name': 'app', 'version': '1.0.0'}, 'dependencies': {}}
            packages = DemandPackages(root, manifest)
            with patch('py_compiler.packaging.executable.prepare_dependencies', return_value={'io': {'package': {'name': 'system.io'}}}) as prepare:
                alias, package = packages.resolve_prelude('system.io')
            self.assertEqual(alias, 'io')
            self.assertEqual(prepare.call_args.args[1]['dependencies']['io'], {'package': 'system.io', 'version': '*'})
            self.assertNotIn('path', manifest['dependencies']['io'])

    def test_package_build_and_tests_resolve_and_lock_implicit_prelude(self):
        from unittest.mock import patch
        from py_compiler.packaging.testing import build_package, test_package
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider, app = root / 'provider', root / 'app'
            for path in (provider, app):
                (path / 'src').mkdir(parents=True)
            (provider / 'package.json').write_text(encode({'package': {'name': 'system.io', 'version': '1.0.0'}}))
            (provider / 'src/lib.sev').write_text('def __print(value: int) -> int:\n    return value + 1\n')
            manifest = {'package': {'name': 'app', 'version': '1.0.0'},
                        'dependencies': {'unused': {'path': 'does-not-exist'}}}
            (app / 'package.json').write_text(encode(manifest))
            (app / 'src/lib.sev').write_text('def answer() -> int:\n    return print(41)\n'
                                             'test "implicit provider":\n    assert(answer() == 42)\n')
            with patch.dict(os.environ, {'SEVERIAN_REGISTRY': str(root / 'registry')}):
                original = publish_package(provider)
                self.assertTrue(build_package(app)['success'])
                self.assertEqual(json.loads((app / 'package.json').read_text()), manifest)
                pins = json.loads((app / 'package.lock').read_text())['packages']
                self.assertEqual(pins, [original['dependency']])
                (provider / 'package.json').write_text(encode({'package': {'name': 'system.io', 'version': '2.0.0'}}))
                (provider / 'src/lib.sev').write_text('def __print(value: int) -> int:\n    return 0\n')
                publish_package(provider)
                result = test_package(app)
                self.assertTrue(result['success'], result)
                self.assertEqual(json.loads((app / 'package.lock').read_text())['packages'], pins)
