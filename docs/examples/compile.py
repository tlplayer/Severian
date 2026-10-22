"""Compile every example independently; retain objects and every diagnostic.

Run from the repository: python3 docs/examples/compile.py
Use --test to execute test blocks in isolated c processes. A failed example does not
prevent other sources from reaching MLIR verification and object emission.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import os
import subprocess

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))

from py_compiler.frontend.src.lib import compile_source
from py_compiler.lir.src.lib import lower, compile_object, companion
from py_compiler.mlir.src.lib import verifier_path
from py_compiler.packaging.package import c_dependencies, compiler_identity
from py_compiler.syntax.recognition import Syntax
from py_compiler.packaging.testing import (build_package, TestEntry, entry_source,
                                 execute_test_case, check_snapshots)


def review_flags(source, diagnostics):
    """Keep executable success separate from unfinished example contracts."""
    flags = []
    for line in source.splitlines():
        if 'AUDIT-INCOMPLETE:' in line or 'AUDIT-SYNTAX:' in line:
            flags.append(line.strip().removeprefix('#').strip())
    if not flags and ('TODO' in source or 'unimplemented' in source):
        flags.append('incomplete: source contains TODO or unimplemented; review required')
    if diagnostics:
        flags.append('blocked: inspect the diagnostic against the manual/SIP before changing syntax')
    return flags


def run_tests(record, output, target, pointer_bits, dependencies, initializers, timeout):
    tests = record.pop('_tests', ())
    local_initializers = record.pop('_initializers', ())
    record['tests'] = []
    if record['status'] != 'compiled-object':
        return
    linker = companion('clang', verifier_path())
    for index, (symbol, test) in enumerate(tests):
        stem = output / record['source']
        binary = stem.with_suffix(f'.test-{index:03}')
        try:
            if test.get('skip'):
                records = execute_test_case(binary, test, timeout=timeout, env=os.environ)
                for item in records:
                    item['location'] = test.get('location')
                record['tests'].extend(records)
                continue
            entry = entry_source([*initializers, *local_initializers], symbol,
                                 benchmark=test['mode'] == 'bench')
            artifact = compile_object(TestEntry(entry, target, pointer_bits))
            obj = binary.with_suffix(binary.suffix + '.o')
            obj.write_bytes(artifact.object_bytes)
            from py_compiler.syntax.complex.profile import linker_options
            linked = subprocess.run([linker, str(obj), record['object'],
                                     *dict.fromkeys([*dependencies, *record.get('dependencies', ())]),
                                     *linker_options(record.get('dependencies', ())),
                                     '-lm', '-o', str(binary)], capture_output=True, text=True, timeout=timeout)
            if linked.returncode:
                raise ValueError('link failed: ' + linked.stderr)
            # Independent cwd prevents file tests from sharing scratch paths.
            working = binary.with_suffix(binary.suffix + '.work')
            working.mkdir(exist_ok=True)
            records = execute_test_case(binary, test, timeout=timeout, env=os.environ, cwd=working)
            for item in records:
                item['location'] = test.get('location')
                check_snapshots(item, ROOT)
            record['tests'].extend(records)
        except (ValueError, OSError, subprocess.TimeoutExpired) as failure:
            record['tests'].append({'label': test['label'], 'mode': test['mode'],
                                    'location': test.get('location'), 'passed': False,
                                    'error': str(failure)})
    record['status'] = ('tested' if all(item['passed'] for item in record['tests']) else 'test-failed') if tests else 'compiled-no-tests'


def check_counts(tests):
    executed = [test for test in tests if not test.get('skipped')]
    runtime = [test for test in executed if test.get('mode') != 'compiler']
    fixtures = [test for test in executed if test.get('mode') == 'compiler']
    return {'tests': len(tests), 'tests-passed': sum(test['passed'] for test in executed),
            'tests-failed': sum(not test['passed'] for test in executed),
            'tests-skipped': len(tests) - len(executed),
            'runtime-checks': len(runtime), 'runtime-passed': sum(test['passed'] for test in runtime),
            'compiler-checks': len(fixtures), 'compiler-passed': sum(test['passed'] for test in fixtures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--prefix', default='', help='optional example path prefix')
    parser.add_argument('--test', action='store_true', help='link and run every compiled test block')
    parser.add_argument('--resume', action='store_true', help='reuse matching results from an interrupted audit')
    parser.add_argument('--timeout', type=float, default=5, help='seconds per c test or link')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    target, pointer_bits = 'x86_64-unknown-linux-gnu', 64
    manifest = json.loads((ROOT / 'package.json').read_text())
    built = {}
    for alias, specification in manifest.get('dependencies', {}).items():
        dependency = (ROOT / specification['path']).resolve()
        if dependency not in built:
            print('Library:', alias, flush=True)
            result = build_package(dependency, _built=built, jobs=args.jobs,
                                   target=target, pointer_bits=pointer_bits)
            if not result['success']:
                print(json.dumps(result, indent=2), flush=True)
                return 1
            built[dependency] = result
    packages = c_dependencies(ROOT, manifest, target, pointer_bits)
    dependencies = list(dict.fromkeys(obj for package in packages.values() for obj in package['objects']))
    initializers = list(dict.fromkeys(symbol for package in packages.values() for symbol in package['initializers']))
    dependency_hash = sha256(json.dumps(packages, sort_keys=True).encode())
    for obj in dependencies:
        dependency_hash.update(Path(obj).read_bytes())
    dependency_fingerprint = dependency_hash.hexdigest()
    runner_fingerprint = sha256(Path(__file__).read_bytes()).hexdigest()
    sources = sorted(path for path in ROOT.rglob('*.sev')
                     if not any(part.endswith('.pkg') for part in path.relative_to(ROOT).parts)
                     and path.relative_to(ROOT).as_posix().startswith(args.prefix))
    output = ROOT / 'package.pkg' / 'example-audit'
    output.mkdir(parents=True, exist_ok=True)
    compiler = compiler_identity()
    report_path = output / ('report.json' if not args.prefix else 'selected-report.json')
    previous = {}
    if args.resume and report_path.exists():
        saved = json.loads(report_path.read_text())
        if (saved.get('compiler') == compiler and saved.get('executed-tests') == args.test
                and saved.get('prefix') == args.prefix
                and saved.get('dependencies-sha256') == dependency_fingerprint
                and saved.get('runner-sha256') == runner_fingerprint
                and saved.get('timeout') == args.timeout):
            previous = {item['source']: item for item in saved['sources']}

    def compile_one(path):
        relative = path.relative_to(ROOT).as_posix()
        source = path.read_text()
        fingerprint = sha256(source.encode()).hexdigest()
        cached = previous.get(relative)
        if cached and cached.get('source-sha256') == fingerprint:
            if 'object' not in cached or Path(cached['object']).is_file():
                return {**cached, '_reused': True}
        record = {'source': relative, 'source-sha256': fingerprint, 'frontend-accepted': False,
                  'status': 'failed', 'diagnostics': [], 'flags': review_flags(source, ())}
        try:
            result = compile_source(relative, source, Syntax(pointer_bits, include_tests=True),
                                    c_packages=packages, debug_path=str(path))
            if result.diagnostics:
                record['diagnostics'] = list(map(str, result.diagnostics))
                record['flags'] = review_flags(source, record['diagnostics'])
                return record
            record['frontend-accepted'] = True
            record['placements'] = [placement for body in result.program.bodies
                                    for placement in getattr(body, 'placements', ())]
            lir = lower(result.program, target, pointer_bits)
            artifact = compile_object(lir)
            stem = output / relative
            stem.parent.mkdir(parents=True, exist_ok=True)
            object_path = stem.with_suffix('.o')
            object_path.write_bytes(artifact.object_bytes)
            stem.with_suffix('.mlir').write_text(lir.text())
            record.update(status='compiled-object', object=str(object_path),
                          dependencies=list(artifact.dependencies),
                          object_sha256=sha256(artifact.object_bytes).hexdigest())
            if args.test:
                record['_tests'] = [(body.name, body.test) for body in result.program.bodies if body.test]
                record['_initializers'] = [body.name for body in result.program.bodies if not body.declaration]
                run_tests(record, output, target, pointer_bits, dependencies, initializers, args.timeout)
        except Exception as failure:
            record['diagnostics'] = [type(failure).__name__ + ': ' + str(failure)]
            record['flags'] = review_flags(source, record['diagnostics'])
        return record

    records = []
    report_path = output / ('report.json' if not args.prefix else 'selected-report.json')
    def save(complete):
        compiled = sum('object' in item for item in records)
        tests = [test for item in records for test in item.get('tests', ())]
        report = {'compiler': compiler, 'target': target, 'pointer-bits': pointer_bits,
                  'includes-tests': True, 'executed-tests': args.test, 'complete': complete,
                  'dependencies-sha256': dependency_fingerprint, 'runner-sha256': runner_fingerprint,
                  'timeout': args.timeout,
                  'prefix': args.prefix, 'compiled': compiled, 'total': len(sources),
                  'processed': len(records),
                  'frontend-accepted': sum(item.get('frontend-accepted', False) for item in records),
                  **check_counts(tests), 'sources': records}
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        return report
    save(False)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for record in executor.map(compile_one, sources):
            reused = record.pop('_reused', False)
            if args.test and not reused and '_tests' in record:
                run_tests(record, output, target, pointer_bits, dependencies, initializers, args.timeout)
            records.append(record)
            save(False)
            print(record['status'] + ': ' + record['source'], flush=True)
    report = save(True)
    print(f"Frontend {report['frontend-accepted']}/{len(records)}; objects {report['compiled']}/{len(records)}; "
          f"runtime {report['runtime-passed']}/{report['runtime-checks']}; "
          f"compiler {report['compiler-passed']}/{report['compiler-checks']}; "
          f"skipped {report['tests-skipped']}. Report: {report_path}", flush=True)
    return 0 if report['compiled'] == len(records) and report['tests-failed'] == 0 else 1


import unittest


class AuditRunnerTests(unittest.TestCase):
    def test_skips_are_not_runtime_passes_or_linked(self):
        import tempfile
        from unittest.mock import patch
        test = {'label': 'unavailable', 'mode': 'unit', 'skip': 'requires GPU'}
        record = {'source': 'example.sev', 'status': 'compiled-object', '_tests': [('test', test)]}
        with tempfile.TemporaryDirectory() as directory, patch(__name__ + '.compile_object') as compile_c:
            run_tests(record, Path(directory), 'x86_64-unknown-linux-gnu', 64, [], [], 1)
        compile_c.assert_not_called()
        self.assertEqual(record['status'], 'tested')
        counts = check_counts(record['tests'])
        self.assertEqual(counts['tests-skipped'], 1)
        self.assertEqual(counts['tests-passed'], 0)
        self.assertEqual(counts['runtime-checks'], 0)

    def test_runtime_execution_checks_output_and_repeats_in_isolated_directory(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        test = {'label': 'output', 'mode': 'unit', 'repeat': 3, 'timeout': 0.25,
                'output-expectations': [{'stream': 'stdout', 'operator': '==', 'value': 'expected'}]}
        record = {'source': 'example.sev', 'status': 'compiled-object', 'object': 'example.o',
                  '_tests': [('test', test)]}
        success = subprocess.CompletedProcess([], 0, 'expected', '')
        wrong = subprocess.CompletedProcess([], 0, 'wrong', '')
        with tempfile.TemporaryDirectory() as directory, \
                patch(__name__ + '.compile_object', return_value=SimpleNamespace(object_bytes=b'object')), \
                patch(__name__ + '.subprocess.run', return_value=success), \
                patch('py_compiler.packaging.testing.execute', side_effect=[success, wrong]) as execute:
            run_tests(record, Path(directory), 'x86_64-unknown-linux-gnu', 64, [], [], 1)
            self.assertEqual(execute.call_args.args[1], 0.25)
            self.assertTrue(execute.call_args.kwargs['cwd'].is_dir())
        self.assertEqual(len(record['tests']), 2)
        self.assertEqual(record['status'], 'test-failed')
        self.assertEqual(check_counts(record['tests'])['runtime-passed'], 1)

    def test_snapshot_mismatch_fails_the_audit(self):
        import struct
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        test = {'label': 'snapshot', 'mode': 'unit'}
        record = {'source': 'example.sev', 'status': 'compiled-object', 'object': 'example.o',
                  '_tests': [('test', test)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / 'snapshots'
            snapshots.mkdir()
            (snapshots / 'value.txt').write_bytes(b'expected')
            captured = root / 'captured.bin'
            name, value = b'value.txt', b'wrong'
            captured.write_bytes(struct.pack('=QQ', len(name), len(value)) + name + value)
            with patch(__name__ + '.ROOT', root), \
                    patch(__name__ + '.compile_object', return_value=SimpleNamespace(object_bytes=b'object')), \
                    patch(__name__ + '.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')), \
                    patch(__name__ + '.execute_test_case', return_value=[{
                        'label': 'snapshot', 'mode': 'unit', 'passed': True,
                        'snapshot-records': str(captured)}]):
                run_tests(record, root, 'x86_64-unknown-linux-gnu', 64, [], [], 1)
        self.assertEqual(record['status'], 'test-failed')
        self.assertIn('snapshot differs', record['tests'][0]['error'])

    def test_compiler_checks_have_a_separate_count(self):
        counts = check_counts([{'mode': 'compiler', 'passed': True},
                               {'mode': 'unit', 'passed': True},
                               {'mode': 'unit', 'passed': False}])
        self.assertEqual(counts['compiler-passed'], 1)
        self.assertEqual(counts['runtime-checks'], 2)
        self.assertEqual(counts['runtime-passed'], 1)


if __name__ == '__main__':
    raise SystemExit(main())
