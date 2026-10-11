"""Package test blocks execute in isolated native processes built through MLIR."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import pty
import re
import resource
import signal
import subprocess
import threading

from py_compiler.package import build, encode, write_atomic
from py_compiler.lir.src.lib import compile_object, companion
from py_compiler.mlir.src.lib import symbol_ref, verifier_path


def build_package(root, *, include_tests=False, _active=(), _built=None, **options):
    root = Path(root).resolve()
    if root in _active:
        raise ValueError('cyclic package dependency: ' + ' -> '.join(map(str, (*_active, root))))
    built = {} if _built is None else _built
    manifest = json.loads((root / 'package.json').read_text())
    for specification in manifest.get('dependencies', {}).values():
        if not isinstance(specification, dict) or not isinstance(specification.get('path'), str):
            raise ValueError('native dependencies require a local path')
        dependency = (root / specification['path']).resolve()
        if dependency not in built:
            dependency_options = {key: value for key, value in options.items() if key != 'stop_after'}
            result = build_package(dependency, _active=(*_active, root), _built=built, **dependency_options)
            if not result['success']:
                return result
            built[dependency] = result
    return build(root, include_tests=include_tests, **options)


@dataclass(frozen=True)
class TestEntry:
    source: str
    target: str
    pointer_bits: int
    dependencies: tuple = ()

    def text(self):
        return self.source


def entry_source(initializers, test):
    calls = [*initializers, test]
    declarations = [f'  func.func private {symbol_ref(symbol)}()' for symbol in dict.fromkeys(calls)]
    instructions = [f'    func.call {symbol_ref(symbol)}() : () -> ()' for symbol in calls]
    return ('module {\n' + '\n'.join(declarations) + '\n  func.func @main() -> i32 {\n'
            + '\n'.join(instructions) + '\n    %zero = arith.constant 0 : i32\n'
            '    func.return %zero : i32\n  }\n}\n')


def test_passed(result, diagnostic=None):
    if diagnostic is None:
        return result.returncode == 0
    return result.returncode == -signal.SIGABRT and diagnostic in result.stdout + result.stderr


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def execute(binary, timeout, env):
    # A terminal preserves MLIR assertion messages printed just before abort.
    # Drain concurrently so verbose tests cannot fill the terminal buffer.
    master, slave = pty.openpty()
    chunks = []
    def drain():
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
    try:
        process = subprocess.Popen([str(binary)], stdout=slave, stderr=subprocess.PIPE,
                                   env=env, preexec_fn=no_core)
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        reader.join(timeout=1)
        return subprocess.CompletedProcess([str(binary)], process.returncode,
                                           b''.join(chunks).decode(errors='replace'), stderr.decode(errors='replace'))
    finally:
        os.close(master)


def test_package(root, *, sanitize=None, timeout=30, **options):
    target = options.get('target', 'x86_64-unknown-linux-gnu')
    machine = {'AMD64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine(), platform.machine())
    if platform.system() != 'Linux' or target.split('-')[0] != machine or 'linux' not in target:
        raise ValueError('native package tests require a matching Linux host target')
    if timeout <= 0:
        raise ValueError('test timeout must be positive')
    result = build_package(root, include_tests=True, **options)
    if not result['success']:
        return result
    package = Path(root).resolve() / 'package.pkg'
    build_id = result['build-id']
    metadata = json.loads((package / 'metadata/realizations' / (build_id + '.json')).read_text())
    symbols = json.loads((package / 'metadata/symbols' / (build_id + '.json')).read_text())
    output = package / 'debug/test' / build_id
    output.mkdir(parents=True, exist_ok=True)
    report = {'success': False, 'build-id': build_id, 'pipeline': 'MLIR -> LLVM -> object', 'tests': []}
    report_path = output / 'results.json'
    write_atomic(report_path, encode(report))
    optimizer = options.get('mlir_opt') or verifier_path()
    linker = companion('clang', optimizer)

    def tool(command):
        process = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise ValueError(f'{command[0]} failed:\n{process.stdout}{process.stderr}')

    objects = [result['object'], *result['dependencies']]
    if sanitize:
        instrumented = []
        for index, obj in enumerate(objects):
            llvm = Path(obj).parent.parent / 'ir/package.ll'
            text = re.sub(r'^(define [^\n]*?) \{$', r'\1 sanitize_address {', llvm.read_text(), flags=re.MULTILINE)
            source, lowered, destination = (output / f'asan-{index}{suffix}' for suffix in ('.ll', '.instrumented.ll', '.o'))
            source.write_text(text)
            tool([companion('opt', optimizer), '-mtriple=' + target, '-passes=asan', '-S', source, '-o', lowered])
            tool([companion('llc', optimizer), '-filetype=obj', '-relocation-model=pic', '-mtriple=' + target, lowered, '-o', destination])
            instrumented.append(str(destination))
        objects = instrumented
    for index, symbol in enumerate(s for s in symbols if s.get('test')):
        test = symbol['test']
        stem = output / f'test-{index:04}'
        source = entry_source(metadata.get('dependency-initializers', []) + metadata['initializers'], symbol['entry'])
        artifact = compile_object(TestEntry(source, target, options.get('pointer_bits', 64)), optimizer)
        stem.with_suffix('.mlir').write_text(source)
        stem.with_suffix('.o').write_bytes(artifact.object_bytes)
        tool([linker, *(['-fsanitize=address'] if sanitize else []), stem.with_suffix('.o'), *objects, '-o', stem])
        expected = test.get('expected-failure')
        env = {**os.environ, **({'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'} if sanitize else {})}
        try:
            process = execute(stem, timeout, env)
            passed = test_passed(process, expected)
            record = {'label': test['label'], 'mode': test['mode'], 'passed': passed,
                      'exit': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
        except subprocess.TimeoutExpired:
            record = {'label': test['label'], 'mode': test['mode'], 'passed': False, 'error': 'test timed out'}
        report['tests'].append(record)
        write_atomic(report_path, encode(report))
        print(('PASS ' if record['passed'] else 'FAIL ') + (test['label'] or symbol['entry']))
    report['success'] = all(test['passed'] for test in report['tests'])
    write_atomic(report_path, encode(report))
    return {'success': report['success'], 'tests': len(report['tests']),
            'failed': sum(not t['passed'] for t in report['tests']), 'report': str(report_path)}


import unittest


class PackageTestRunnerTests(unittest.TestCase):
    def test_build_orders_local_dependencies_before_consumer(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / 'library'
            dependency.mkdir()
            (dependency / 'package.json').write_text('{}')
            (root / 'package.json').write_text(encode({'dependencies': {'vector': {'path': 'library'}}}))
            with patch(__name__ + '.build', return_value={'success': True}) as compile_package:
                result = build_package(root, include_tests=True)
            self.assertTrue(result['success'])
            self.assertEqual([call.args[0] for call in compile_package.call_args_list], [dependency, root])
            self.assertFalse(compile_package.call_args_list[0].kwargs['include_tests'])
            self.assertTrue(compile_package.call_args_list[1].kwargs['include_tests'])

    def test_dependency_cycles_fail_before_compilation(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'package.json').write_text(encode({'dependencies': {'self': {'path': '.'}}}))
            with patch(__name__ + '.build') as compile_package:
                with self.assertRaisesRegex(ValueError, 'cyclic package dependency'):
                    build_package(root)
                compile_package.assert_not_called()

    def test_failure_requires_abort_and_matching_diagnostic(self):
        def result(code, message):
            return subprocess.CompletedProcess([], code, stdout=message, stderr='')
        self.assertTrue(test_passed(result(-signal.SIGABRT, 'out of bounds'), 'out of bounds'))
        self.assertFalse(test_passed(result(0, 'out of bounds'), 'out of bounds'))
        self.assertFalse(test_passed(result(-signal.SIGSEGV, 'out of bounds'), 'out of bounds'))
        self.assertFalse(test_passed(result(-signal.SIGABRT, 'unrelated failure'), 'out of bounds'))

    def test_entry_initializes_package_before_test(self):
        text = entry_source(['initialize'], 'test with spaces')
        self.assertLess(text.index('func.call @initialize'), text.index('func.call @"test with spaces"'))
        self.assertIn('func.return %zero : i32', text)
