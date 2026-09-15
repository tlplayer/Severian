#!/usr/bin/env python3
"""Check generic specialization, invocation-owned output, and incremental reuse."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    work = args.work.resolve() if args.work else Path(tempfile.mkdtemp(prefix='sev-generic-consumer-'))
    work.mkdir(parents=True, exist_ok=True)
    for name in ('producer', 'consumer'):
        shutil.copytree(HERE / name, work / name, ignore=shutil.ignore_patterns('package.pkg', '.sev-*'))
    env = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT), 'SEVERIAN_HOME': str(work / 'home')}
    records = []

    def build():
        result = subprocess.run([str(args.compiler.resolve()), 'build', 'consumer', '--sysroot', str(ROOT)], cwd=work,
                                env=env, text=True, capture_output=True, timeout=300)
        records.append({'stdout': result.stdout, 'stderr': result.stderr, 'status': result.returncode})
        assert result.returncode == 0, result.stdout + result.stderr
        (work / 'results.json').write_text(json.dumps(records, indent=2))
        return result

    build()
    package = work / 'package.pkg'
    binary = package / 'bin/generic-consumer'
    expected = (HERE / 'expected.stdout').read_text()
    assert subprocess.check_output([binary], text=True) == expected
    assert not list((work / 'consumer').rglob('package.pkg'))
    assert not list((work / 'producer').rglob('package.pkg'))
    report = json.loads((package / 'debug/quality/build-inputs.json').read_text())
    files = {row['file']: row for row in report['files']}
    assert files['src/main.sev']['contributed']
    assert not files['src/unused.sev']['contributed']
    assert 'compiling ' not in build().stderr
    source = work / 'consumer/src/main.sev'
    source.write_text(source.read_text().replace('return 41', 'return 41 + 1'))
    build()
    assert subprocess.check_output([binary], text=True) == expected[:-3] + '42\n'
    profiles = sorted((package / 'debug/profile').glob('backend-*/stages.json'), key=lambda p: p.stat().st_mtime_ns)
    stages = json.loads(profiles[-1].read_text())['stages']
    assert any(row['stage'] == 'object' and not row['reused'] for row in stages)
    assert any(row['stage'] == 'c-object' and row['reused'] for row in stages)
    # A same-line comment changes source identity but leaves emitted IR unchanged.
    source.write_text(source.read_text().replace('return 41 + 1', 'return 41 + 1  # no semantic change'))
    build()
    profiles = sorted((package / 'debug/profile').glob('backend-*/stages.json'), key=lambda p: p.stat().st_mtime_ns)
    stages = json.loads(profiles[-1].read_text())['stages']
    assert all(row['reused'] for row in stages if row['stage'] in ('ownership', 'llvm-dialect', 'llvm-ir', 'object'))
    assert subprocess.check_output([binary], text=True) == expected[:-3] + '42\n'
    # The same cross-package generic remains callable through a grouped alias.
    source.write_text(source.read_text().replace('import printer', 'from printer import\n{\n    foo as emit,\n}').replace('printer.foo', 'emit'))
    build()
    assert subprocess.check_output([binary], text=True) == expected[:-3] + '42\n'
    print(f'PASS: 15 generic calls, +1 invalidation, IR reuse, and file contribution reports: {work}')


if __name__ == '__main__':
    main()
