#!/usr/bin/env python3
"""Build, inspect, publish, relocate and consume the package format contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read_json(path):
    return json.loads(path.read_text())


def snapshot(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def envelope(path, magic):
    data = path.read_bytes()
    assert data[:8] == magic
    version, count = struct.unpack_from('>II', data, 8)
    assert version == 1 and count == 2
    end = 16 + 20 * count
    sections = []
    for i in range(count):
        tag, offset, size = struct.unpack_from('>IQQ', data, 16 + 20 * i)
        assert tag == i + 1 and offset == end and offset + size <= len(data)
        sections.append(data[offset:offset + size])
        end = offset + size
    assert end == len(data)
    return sections


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--compiler', type=Path, default=ROOT / 'bin/sev_compiler')
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    work = args.work.resolve() if args.work else Path(tempfile.mkdtemp(prefix='sev-package-formats-'))
    work.mkdir(parents=True, exist_ok=True)
    compiler = args.compiler.resolve()
    env = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT), 'SEVERIAN_HOME': str(work / 'home'),
           'SEVERIAN_REGISTRY': str(work / 'registry'), 'CARGO_TARGET_DIR': str(work / 'cargo-target')}
    print(f'Evidence and generated packages: {work}', flush=True)
    records = []

    def run(*command, cwd=work, succeeds=True, extra_env=None, timeout=300):
        started = time.monotonic()
        if str(command[0]) == str(compiler):
            command = (*command, '--sysroot', ROOT) if len(command) > 1 and not str(command[1]).startswith('__') else command
        result = subprocess.run(list(map(str, command)), cwd=cwd, env={**env, **(extra_env or {})},
                                capture_output=True, text=True, timeout=timeout)
        records.append({'command': list(map(str, command)), 'status': result.returncode,
                        'seconds': time.monotonic() - started, 'stdout': result.stdout, 'stderr': result.stderr})
        (work / 'results.json').write_text(json.dumps(records, indent=2))
        assert (result.returncode == 0) == succeeds, f'{command}\n{result.stdout}\n{result.stderr}'
        return result

    producer = work / 'geometry'
    consumer = work / 'consumer'
    shutil.copytree(HERE / 'geometry', producer, ignore=shutil.ignore_patterns('package.pkg', '.sev-*'))
    shutil.copytree(HERE / 'consumer', consumer, ignore=shutil.ignore_patterns('package.pkg', '.sev-*'))
    run(compiler, 'build', producer)
    package = work / 'package.pkg'
    assert (package / 'bin/geometry-tool').is_file()
    assert not list(producer.rglob('package.pkg'))
    manifests = [read_json(p) for p in (package / 'metadata/realizations').glob('*.json')
                 if not p.name.endswith('.inputs.json')]
    library = next(m for m in manifests if m['kind'] == 'library')
    native = next(m for m in manifests if m['kind'] == 'native')
    build_id, triple = library['build-id'], library['target']
    artifact_root = package / 'artifacts' / triple / 'dev' / build_id
    assert run(package / native['output']).stdout.strip() == '85'
    for location in ['package.json', 'package.lock', 'source-index.json', 'artifacts.json']:
        read_json(package / 'metadata' / location)
    read_json(producer / 'package.lock')
    assert (package / 'source/src/lib.sev').read_bytes() == (producer / 'src/lib.sev').read_bytes()
    assert (package / 'source/native/scale.c').is_file()
    for directory in ['dep-info', 'incremental', 'staging']:
        assert (package / 'build' / build_id / directory).is_dir()
    read_json(package / 'build' / build_id / 'inputs.json')
    assert list((package / 'build' / build_id / 'dep-info').glob('*.d'))
    for name, expected in [('object/geometry.o', b'\x7fELF'), ('archive/libgeometry.a', b'!<arch>\n'),
                           ('dynamic/libgeometry.so', b'\x7fELF'), ('ir/geometry.bc', b'BC\xc0\xde'),
                           ('ir/geometry.mlirbc', b'ML\xefR')]:
        assert (artifact_root / name).read_bytes().startswith(expected), name
    run('llvm-dis-21', artifact_root / 'ir/geometry.bc', '-o', work / 'decoded.ll')
    run('mlir-opt-21', artifact_root / 'ir/geometry.mlirbc', '-o', work / 'decoded.mlir')
    assert 'define' in (artifact_root / 'ir/geometry.ll').read_text()
    envelope(package / library['interface'], b'SEVPKGI\0')
    envelope(artifact_root / 'ir/geometry.mir', b'SEVMIR\0\0')
    run(compiler, '__verify-mir', artifact_root / 'ir/geometry.mir')
    damaged_mir = work / 'damaged.mir'
    damaged_mir.write_bytes((artifact_root / 'ir/geometry.mir').read_bytes()[:-1])
    run(compiler, '__verify-mir', damaged_mir, succeeds=False)
    for relative, checksum in library['files'].items():
        assert 'sha256:' + hashlib.sha256((package / relative).read_bytes()).hexdigest() == checksum
    windows = package / 'artifacts/x86_64-pc-windows-msvc/dev' / build_id
    assert (windows / 'dynamic/geometry-native.dll').read_bytes()[:2] == b'MZ'
    assert 'COFF-x86-64' in run('llvm-readobj-21', '--file-headers', windows / 'object/scale.obj').stdout
    for suffix in ['.lib', '.import.lib']:
        assert (windows / ('archive/geometry-native' + suffix)).read_bytes().startswith(b'!<arch>\n')
    kinds = {a['path']: a['kind'] for a in library['artifact']}
    assert kinds[str((windows / 'archive/geometry-native.import.lib').relative_to(package))] == 'import-library'
    assert list((package / 'debug/symbols' / build_id).rglob('*.pdb'))
    darwin = package / 'artifacts/aarch64-apple-macos11/dev' / build_id
    assert 'Mach-O' in run('llvm-readobj-21', '--file-headers', darwin / 'dynamic/geometry-native.dylib').stdout
    assert list((package / 'debug/symbols' / build_id).rglob('*.dSYM'))
    assert (artifact_root / 'archive/libprovider.rlib').read_bytes().startswith(b'!<arch>\n')
    assert (artifact_root / 'archive/libprovider.rmeta').stat().st_size > 0
    rust_probe = work / 'provider_probe.rs'
    rust_probe.write_text('fn main() { assert_eq!(provider::perimeter(6, 7), 26); }\n')
    run('rustc', rust_probe, '--extern', f'provider={artifact_root / "archive/libprovider.rlib"}', '-o', work / 'provider-probe')
    run(work / 'provider-probe')
    # Inspect the OCI graph and verify every content-addressed blob.
    oci = package / 'container' / native['target'] / native['build-id']
    assert json.loads((oci / 'oci-layout').read_text())['imageLayoutVersion'] == '1.0.0'
    def blob(descriptor):
        path = oci / 'blobs/sha256' / descriptor['digest'].split(':')[1]
        assert path.stat().st_size == descriptor['size']
        assert 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['digest']
        return path
    image = json.loads(blob(json.loads((oci / 'index.json').read_text())['manifests'][0]).read_text())
    config = json.loads(blob(image['config']).read_text())
    assert config['config']['Entrypoint'] == ['/app']
    with tarfile.open(blob(image['layers'][0])) as layer:
        assert './app' in layer.getnames()
        assert hashlib.sha256(layer.extractfile('./app').read()).digest() == hashlib.sha256((package / native['output']).read_bytes()).digest()

    # Actual native instrumentation, scoped to this native provider's C lines.
    reports = package / 'debug'
    profile = reports / 'profile' / build_id
    coverage = reports / 'coverage' / build_id
    probe = work / 'profile_probe.c'
    probe.write_text('#include <stdint.h>\nint64_t geometry_scale(int64_t);\nint main(void) { return geometry_scale(42) != 84; }\n')
    run('clang-21', '-g', '-gsplit-dwarf', '-fprofile-instr-generate', '-fcoverage-mapping', '-c', producer / 'native/scale.c', '-o', work / 'scale.o')
    shutil.copy2(work / 'scale.dwo', reports / 'symbols' / build_id / 'scale.dwo')
    run('clang-21', '-fprofile-instr-generate', '-fcoverage-mapping', probe, work / 'scale.o', '-o', work / 'profile-probe')
    run(work / 'profile-probe', extra_env={'LLVM_PROFILE_FILE': str(profile / 'geometry.profraw')})
    run('llvm-profdata-21', 'merge', '-sparse', profile / 'geometry.profraw', '-o', profile / 'geometry.profdata')
    for fmt, name in [('text', 'geometry.json'), ('lcov', 'geometry.lcov')]:
        result = run('llvm-cov-21', 'export', work / 'profile-probe', f'-instr-profile={profile / "geometry.profdata"}', f'-format={fmt}')
        (coverage / name).write_text(result.stdout)
    (reports / 'test' / build_id / 'results.json').write_text(json.dumps({'passed': True, 'scope': 'native provider', 'commands': records}, indent=2))
    # A warm build cannot re-enter Severian code generation.
    warm = run(compiler, 'build', producer)
    assert 'compiling ' not in warm.stderr, warm.stderr
    publication = run(compiler, 'publish', producer, '--local', '--build-profile', 'dev')
    releases = list((work / 'registry').rglob('metadata/publication.json'))
    assert len(releases) == 1, publication.stdout
    release = releases[0].parent.parent
    assert not any((release / name).exists() for name in ['source', 'build', 'cache', 'debug'])
    read_json(release / 'metadata/publication.json')
    frozen = snapshot(release)
    shutil.move(producer, work / 'producer-unavailable')
    # Each new consumer compiles itself while loading the dependency's interface.
    result = run(compiler, 'run', consumer)
    assert result.stdout.strip() == '85', result.stdout
    assert 'compiling ' in result.stderr and '/metadata/src/' not in result.stderr
    published_lib = next(read_json(p) for p in (release / 'metadata/realizations').glob('*.json')
                         if read_json(p).get('kind') == 'library')
    binding = next(i for i in read_json(release / 'package.pkgi/index.json')['interface']
                   if i['build-id'] == published_lib['build-id'])
    header = release / 'package.pkgi' / binding['c']
    archive = release / published_lib['output']
    shared = next(release / p for p in published_lib['files'] if p.endswith('/dynamic/libgeometry.so'))
    c_probe = work / 'consumer.c'
    c_probe.write_text('#include "geometry.h"\nint main(void) { return geometry_area(6, 7) != 85 || geometry_answer() != 42; }\n')
    run('clang-21', c_probe, '-I', header.parent, archive, '-lm', '-o', work / 'c-static')
    run(work / 'c-static')
    run('clang-21', c_probe, '-I', header.parent, shared, '-Wl,-rpath,' + str(shared.parent), '-o', work / 'c-shared')
    run(work / 'c-shared')
    rust_consumer = work / 'rust-consumer'
    (rust_consumer / 'src').mkdir(parents=True)
    crate = (release / 'package.pkgi' / binding['rust']).parent
    (rust_consumer / 'Cargo.toml').write_text('[package]\nname="consumer"\nversion="1.0.0"\nedition="2024"\n[dependencies]\ngeometry={path=' + json.dumps(str(crate)) + '}\n[workspace]\n')
    (rust_consumer / 'src/main.rs').write_text('fn main() { unsafe { assert_eq!(geometry::area(6, 7), 85); assert_eq!(geometry::answer(), 42); } }\n')
    run('cargo', 'run', '--offline', '--manifest-path', rust_consumer / 'Cargo.toml')
    assert snapshot(release) == frozen, 'consumption changed immutable publication'
    # Inventory corruption is a hard failure, never permission to compile source.
    interface = release / published_lib['interface']
    interface.write_bytes(interface.read_bytes()[:-1])
    failure = run(compiler, 'build', consumer, succeeds=False)
    assert 'PackageIntegrityError' in failure.stderr
    print(f'PASS: native formats, binary metadata/MIR, bindings, OCI, reports and source-free reuse ({len(records)} commands)', flush=True)


if __name__ == '__main__':
    main()
