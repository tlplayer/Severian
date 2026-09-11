#!/usr/bin/env python3
"""Build, execute, and test every numerics example with the source compiler.

Failures remain failures; each example and each independent mode still runs.
The Rust seed is never used by this audit. Reports and native artifacts remain
under sev_compiler/package.pkg/numerics/run-*.
"""
import hashlib
import json
from pathlib import Path
import tempfile

from migration import COMPILER, ROOT
from resource_guard import run


def main():
    parent = ROOT / 'sev_compiler/package.pkg/numerics'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    digest = hashlib.sha256(COMPILER.read_bytes()).hexdigest()
    expected_output = {
        '01-numeric-types': '8\n42\n1000000\n0.5\n0.125\n',
        '02-promotion': '4.5\n',
    }
    examples = sorted((ROOT / 'docs/examples/08-numerics').glob('*.sev'))
    if not examples:
        raise RuntimeError('numerics examples are missing')
    records = []
    for subject in examples:
        record = {'source': str(subject.relative_to(ROOT)), 'stages': {}}
        for mode in ('build', 'test'):
            destination = output / (subject.stem + ('.tests' if mode == 'test' else ''))
            command = [COMPILER, mode, subject, '--sysroot', ROOT, '-o', destination]
            result = run(command, cwd=ROOT, timeout=180)
            (output / f'{subject.stem}.{mode}.stdout').write_text(result.stdout)
            (output / f'{subject.stem}.{mode}.stderr').write_text(result.stderr)
            record['stages'][mode] = {
                'command': list(map(str, command)), 'exit': result.returncode,
                'resources': result.resources,
            }
            if mode == 'build' and result.returncode == 0:
                execution = run([destination], cwd=ROOT, timeout=30)
                (output / f'{subject.stem}.run.stdout').write_text(execution.stdout)
                (output / f'{subject.stem}.run.stderr').write_text(execution.stderr)
                expected = expected_output.get(subject.stem, '')
                output_fixture = subject.with_suffix('.stdout')
                if output_fixture.exists():
                    expected = output_fixture.read_text()
                error_fixture = subject.with_suffix('.stderr')
                record['stages']['run'] = {
                    'command': [str(destination)], 'exit': execution.returncode,
                    'stdout_matches': execution.stdout == expected,
                    'stderr_matches': execution.stderr == (error_fixture.read_text() if error_fixture.exists() else ''),
                    'resources': execution.resources,
                }
        record['passed'] = all(
            stage['exit'] == 0 and stage.get('stdout_matches', True)
            and stage.get('stderr_matches', True)
            for stage in record['stages'].values()
        ) and 'run' in record['stages']
        records.append(record)
        print(('PASS ' if record['passed'] else 'FAIL ') + subject.name, flush=True)
    unchanged = hashlib.sha256(COMPILER.read_bytes()).hexdigest() == digest
    report = {'compiler': str(COMPILER), 'compiler_sha256': digest,
              'compiler_unchanged': unchanged, 'examples': records}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    passed = sum(record['passed'] for record in records)
    print(f'{passed}/{len(records)} examples passed; report: {output / "report.json"}')
    return 0 if unchanged and passed == len(records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
