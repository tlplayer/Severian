#!/usr/bin/env python3
"""Immutable native-provider publication through the Severian source compiler."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
COMPILER = Path(os.environ.get('SEVERIAN_SOURCE_COMPILER', ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler'))

def snapshot(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}

def main():
    with tempfile.TemporaryDirectory(prefix='sev-publication-contract-') as temporary:
        root = Path(temporary)
        registry = root / 'registry'
        env = {**os.environ, 'SEVERIAN_REGISTRY': str(registry), 'SEVERIAN_SYSROOT': str(ROOT)}
        def sev(*args, cwd=root, succeeds=True):
            result = subprocess.run([str(COMPILER), *map(str, args)], cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=180)
            assert (result.returncode == 0) == succeeds, (args, result.stdout, result.stderr)
            return result
        producer = root / 'native_value'
        (producer / 'src').mkdir(parents=True)
        (producer / 'extern').mkdir()
        (producer / 'package.toml').write_text('''[package]
name = "native_value"
version = "0.1.0"
[lib]
path = "src/lib.sev"
[xxi.c]
sources = ["extern/value.c"]
''')
        (producer / 'src/lib.sev').write_text('@c(symbol="library_answer")\ndef answer() -> int\n')
        (producer / 'extern/value.c').write_text('#include <stdint.h>\nint64_t library_answer(void) { return 42; }\n')
        sev('publish', cwd=producer)
        release = registry / 'packages/native_value/0.1.0'
        assert (release / 'source/extern/value.c').is_file()
        frozen = snapshot(release)
        sev('publish', cwd=producer)
        assert snapshot(release) == frozen, 'identical publication mutated the release'
        consumer = root / 'consumer'
        sev('new', 'consumer')
        sev('add', 'native_value@0.1.0', cwd=consumer)
        (consumer / 'src/main.sev').write_text('import native_value\ndef main():\n    print(native_value.answer())\n')
        assert sev('run', cwd=consumer).stdout.strip() == '42'
        executable = consumer / 'package.pkg/host/dev/bin/consumer'
        before_build = executable.stat().st_mtime_ns
        sev('build', cwd=consumer)
        assert executable.stat().st_mtime_ns == before_build, 'unchanged consumer was linked again'
        assert list((consumer / 'package.pkg/build/units').glob('*/inputs'))
        for cached in (consumer / 'package.pkg/build/units').glob('*/inputs'):
            cached.write_text('invalid cache content')
        sev('build', cwd=consumer)
        assert executable.stat().st_mtime_ns != before_build, 'damaged cache was reused'

        sev('clean', cwd=producer)
        assert snapshot(release) == frozen, 'build/consume/clean mutated the release'
        changed = producer / 'extern/value.c'
        changed.write_text(changed.read_text().replace('42', '99'))
        rejected = sev('publish', cwd=producer, succeeds=False)
        assert 'PackageVersionConflict' in rejected.stderr
        assert snapshot(release) == frozen
        (producer / 'package.toml').write_text((producer / 'package.toml').read_text().replace('0.1.0', '0.1.1'))
        (producer / 'src/lib.sev').write_text('def broken(:\n')
        sev('publish', cwd=producer, succeeds=False)
        assert not (registry / 'packages/native_value/0.1.1').exists()
        assert not [p for p in registry.glob('.sev-*') if p.is_dir()]
        shutil.rmtree(producer)
        assert sev('run', cwd=consumer).stdout.strip() == '42'
        for operation in ['build', 'clean']:
            rejected = sev(operation, cwd=release / 'source', succeeds=False)
            assert 'immutable' in rejected.stderr
        assert snapshot(release) == frozen
        # Consumers detect a modified provider, even if its version is unchanged.
        provider = release / 'source/extern/value.c'
        provider.write_text(provider.read_text().replace('42', '77'))
        rejected = sev('build', cwd=consumer, succeeds=False)
        assert 'PackageIntegrityError' in rejected.stderr
        assert not [p for p in registry.glob('.sev-*') if p.is_dir()]
    print('immutable native-provider publication: passed')

if __name__ == '__main__':
    main()
