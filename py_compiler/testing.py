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
import shutil
import subprocess
import sys
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
    for _, specification in sorted(manifest.get('dependencies', {}).items()):
        if not isinstance(specification, dict) or not isinstance(specification.get('path'), str):
            raise ValueError('native dependencies require a local path')
        dependency = (root / specification['path']).resolve()
        if dependency not in built:
            dependency_options = {key: value for key, value in options.items() if key not in ('stop_after', 'publish')}
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
    calls = [*dict.fromkeys(initializers), test]
    declarations = [f'  func.func private {symbol_ref(symbol)}()' for symbol in dict.fromkeys(calls)]
    instructions = [f'    func.call {symbol_ref(symbol)}() : () -> ()' for symbol in calls]
    return ('module {\n' + '\n'.join(declarations) + '\n  func.func @main() -> i32 {\n'
            + '\n'.join(instructions) + '\n    %zero = arith.constant 0 : i32\n'
            '    func.return %zero : i32\n  }\n}\n')


def test_passed(result, diagnostic=None):
    if diagnostic is None:
        return result.returncode == 0
    return result.returncode == -signal.SIGABRT and diagnostic in result.stdout + result.stderr


def test_environment(environ, sanitize):
    env = dict(environ)
    if sanitize:
        # Honor explicit runtime settings (e.g. LSan under ptrace), while
        # keeping memory errors fatal and leak checking enabled by default.
        options = env.get('ASAN_OPTIONS', '')
        env['ASAN_OPTIONS'] = 'detect_leaks=1:' + options + ':halt_on_error=1'
    return env


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
        command = list(map(str, binary)) if isinstance(binary, (list, tuple)) else [str(binary)]
        process = subprocess.Popen(command, stdout=slave, stderr=subprocess.PIPE,
                                   env=env, preexec_fn=no_core, start_new_session=True)
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
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        reader.join(timeout=1)
        return subprocess.CompletedProcess(command, process.returncode,
                                           b''.join(chunks).decode(errors='replace'), stderr.decode(errors='replace'))
    finally:
        os.close(master)


GDB_SCRIPT = '''import gdb, json, signal
result = {"exit": None, "backtrace": ""}
def exited(event):
    result["exit"] = getattr(event, "exit_code", None)
def stopped(event):
    if isinstance(event, gdb.SignalEvent):
        result["exit"] = -int(getattr(signal, event.stop_signal))
        result["backtrace"] = gdb.execute("thread apply all bt", to_string=True)
gdb.events.exited.connect(exited)
gdb.events.stop.connect(stopped)
gdb.execute("run")
print("SEV_DEBUG_RESULT=" + json.dumps(result))
'''


def debug_execute(binary, timeout, env, script):
    debugger = shutil.which('gdb')
    if not debugger:
        raise ValueError('--backtrace requires gdb')
    process = execute([debugger, '--quiet', '--nx', '--batch',
                       '-ex', 'set pagination off', '-ex', 'set confirm off',
                       '-ex', 'set disable-randomization off', '-x', script,
                       '--args', binary], timeout, env)
    marker = 'SEV_DEBUG_RESULT='
    records = [line[len(marker):] for line in process.stdout.splitlines() if line.startswith(marker)]
    if process.returncode or len(records) != 1:
        raise ValueError('debugger could not run the test:\n' + process.stdout + process.stderr)
    result = json.loads(records[0])
    if result['exit'] is None:
        raise ValueError('debugger did not report a test exit or fatal signal')
    process.returncode = result['exit']
    process.stdout = '\n'.join(line for line in process.stdout.splitlines() if not line.startswith(marker))
    return process, result['backtrace']


def test_location(root, test):
    location = test['location']
    path = (Path(root) / location['file']).resolve()
    return {**location, 'file': str(path), 'uri': path.as_uri()}


def format_test_result(record, fallback='', *, hyperlinks=None):
    location = record['location']
    address = f"{location['file']}:{location['line']}:{location['column']}"
    if hyperlinks is None:
        hyperlinks = sys.stdout.isatty() and os.environ.get('TERM') != 'dumb'
    if hyperlinks:
        address = f"\033]8;;{location['uri']}\033\\{address}\033]8;;\033\\"
    status = 'PASS' if record['passed'] else 'FAIL'
    return f"{status} {record['label'] or fallback} ({address})"


def test_package(root, *, sanitize=None, timeout=30, backtrace=False, **options):
    if backtrace and sanitize:
        raise ValueError('run --backtrace and --sanitize separately; LeakSanitizer cannot run under ptrace')
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
    build_id = result['output-slot']
    metadata = json.loads((package / 'metadata/realizations' / (build_id + '.json')).read_text())
    symbols = json.loads((package / 'metadata/symbols' / (build_id + '.json')).read_text())
    output = package / 'debug/test' / build_id
    output.mkdir(parents=True, exist_ok=True)
    report = {'success': False, 'build-id': result['build-id'], 'output-slot': build_id,
              'pipeline': 'MLIR -> LLVM -> object', 'tests': []}
    report_path = output / 'results.json'
    write_atomic(report_path, encode(report))
    script = output / 'backtrace.py'
    if backtrace:
        script.write_text(GDB_SCRIPT)
    optimizer = options.get('mlir_opt') or verifier_path()
    linker = companion('clang', optimizer)

    def tool(command):
        process = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise ValueError(f'{command[0]} failed:\n{process.stdout}{process.stderr}')

    objects = [*result['objects'], *result['dependencies']]
    original_objects = list(objects)
    if sanitize:
        instrumented = []
        for index, obj in enumerate(objects):
            llvm = Path(str(obj).replace('/object/', '/ir/')).with_suffix('.ll')
            text = re.sub(r'^(define [^\n]*?)( !dbg ![0-9]+)? \{$',
                          r'\1 sanitize_address\2 {', llvm.read_text(), flags=re.MULTILINE)
            source, lowered, destination = (output / f'asan-{index}{suffix}' for suffix in ('.ll', '.instrumented.ll', '.o'))
            source.write_text(text)
            tool([companion('opt', optimizer), '-mtriple=' + target, '-passes=asan', '-S', source, '-o', lowered])
            tool([companion('llc', optimizer), '-filetype=obj', '-relocation-model=pic', '-mtriple=' + target, lowered, '-o', destination])
            instrumented.append(str(destination))
        objects = instrumented
    linked_objects = dict(zip(original_objects, objects))
    partitions = {part['submodule']: part for part in metadata.get('partitions', [])}
    symbol_owners = {name: key for key, part in partitions.items() for name in part['symbols']}
    for index, symbol in enumerate(s for s in symbols if s.get('test')):
        test = symbol['test']
        stem = output / f'test-{index:04}'
        source = entry_source(metadata.get('dependency-initializers', []) + metadata['initializers'], symbol['entry'])
        artifact = compile_object(TestEntry(source, target, options.get('pointer_bits', 64)), optimizer)
        stem.with_suffix('.mlir').write_text(source)
        stem.with_suffix('.o').write_bytes(artifact.object_bytes)
        # LLVM lowers floating-point operations such as trunc to libm calls.
        # Native package tests already require a Linux host above.
        roots = [symbol['entry'], *metadata['initializers']]
        pending = [symbol_owners[name] for name in roots if name in symbol_owners]
        selected = set()
        while pending:
            owner = pending.pop()
            if owner not in selected:
                selected.add(owner)
                pending.extend(partitions[owner]['dependencies'])
        local = [str(package / part['object']) for key, part in partitions.items() if key in selected]
        selected_objects = [*local, *result['dependencies']] if partitions else original_objects
        tool([linker, *(['-fsanitize=address'] if sanitize else []), stem.with_suffix('.o'),
              *(linked_objects[obj] for obj in selected_objects), '-lm', '-o', stem])
        expected = test.get('expected-failure')
        env = test_environment(os.environ, sanitize)
        try:
            trace = ''
            if backtrace:
                process, trace = debug_execute(stem, timeout, env, script)
            else:
                process = execute(stem, timeout, env)
            passed = test_passed(process, expected)
            record = {'label': test['label'], 'mode': test['mode'], 'passed': passed,
                      'exit': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr,
                      'backtrace': trace}
            if trace:
                stem.with_suffix('.backtrace.txt').write_text(trace)
                print(trace, end='')
        except subprocess.TimeoutExpired:
            record = {'label': test['label'], 'mode': test['mode'], 'passed': False, 'error': 'test timed out'}
        record['location'] = test_location(root, test)
        record['objects'] = selected_objects
        report['tests'].append(record)
        write_atomic(report_path, encode(report))
        print(format_test_result(record, symbol['entry']))
    report['success'] = all(test['passed'] for test in report['tests'])
    write_atomic(report_path, encode(report))
    return {'success': report['success'], 'tests': len(report['tests']),
            'failed': sum(not t['passed'] for t in report['tests']), 'report': str(report_path)}


import unittest


class PackageTestRunnerTests(unittest.TestCase):
    def test_sanitizer_environment_preserves_explicit_runtime_options(self):
        source = {'ASAN_OPTIONS': 'detect_leaks=0:halt_on_error=0', 'OTHER': 'value'}
        env = test_environment(source, 'address')
        options = dict(option.split('=', 1) for option in env['ASAN_OPTIONS'].split(':') if option)
        self.assertEqual(options['detect_leaks'], '0')
        self.assertEqual(options['halt_on_error'], '1')
        self.assertEqual(env['OTHER'], 'value')
        self.assertEqual(source['ASAN_OPTIONS'], 'detect_leaks=0:halt_on_error=0')
        self.assertEqual(test_environment(source, None), source)
        self.assertIn('detect_leaks=1', test_environment({}, 'address')['ASAN_OPTIONS'])

    def test_result_locations_and_links(self):
        root = Path('/tmp/package with spaces')
        location = test_location(root, {'location': {'file': 'src/lib.sev', 'line': 58, 'column': 5}})
        self.assertEqual(location['file'], str(root / 'src/lib.sev'))
        self.assertEqual(location['uri'], 'file:///tmp/package%20with%20spaces/src/lib.sev')
        for passed in (True, False):
            record = {'label': 'vector growth', 'passed': passed, 'location': location}
            status = 'PASS' if passed else 'FAIL'
            plain = format_test_result(record, hyperlinks=False)
            self.assertEqual(plain, f'{status} vector growth ({root}/src/lib.sev:58:5)')
            linked = format_test_result(record, hyperlinks=True)
            self.assertIn(f"\033]8;;{location['uri']}\033\\", linked)
            self.assertIn(f'{root}/src/lib.sev:58:5', linked)
        record.update(label='', passed=False, error='test timed out')
        self.assertIn('FAIL test_symbol (', format_test_result(record, 'test_symbol', hyperlinks=False))

    def test_debugger_success_preserves_inferior_failure(self):
        from unittest.mock import patch
        payload = {'exit': -signal.SIGABRT, 'backtrace': '#0 leaf at example.sev:3\n'}
        process = subprocess.CompletedProcess([], 0, 'assertion failed\nSEV_DEBUG_RESULT=' + json.dumps(payload), '')
        with patch(__name__ + '.shutil.which', return_value='/usr/bin/gdb'), patch(__name__ + '.execute', return_value=process):
            result, trace = debug_execute('binary', 30, {}, 'script.py')
        self.assertEqual(result.returncode, -signal.SIGABRT)
        self.assertEqual(trace, payload['backtrace'])
        self.assertNotIn('SEV_DEBUG_RESULT', result.stdout)
        self.assertFalse(test_passed(result))
        self.assertTrue(test_passed(result, 'assertion failed'))

    def test_debugger_startup_failure_is_not_a_test_pass(self):
        from unittest.mock import patch
        process = subprocess.CompletedProcess([], 0, 'ptrace: Operation not permitted', '')
        with patch(__name__ + '.shutil.which', return_value='/usr/bin/gdb'), patch(__name__ + '.execute', return_value=process):
            with self.assertRaisesRegex(ValueError, 'debugger could not run'):
                debug_execute('binary', 30, {}, 'script.py')

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

    def test_diamond_builds_shared_dependency_once(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = {'app': ('right', 'left'), 'left': ('base',), 'right': ('base',), 'base': ()}
            for name, dependencies in graph.items():
                path = root / name
                path.mkdir()
                (path / 'package.json').write_text(encode({'dependencies': {
                    dep: {'path': '../' + dep} for dep in dependencies}}))
            with patch(__name__ + '.build', return_value={'success': True}) as compile_package:
                build_package(root / 'app')
            self.assertEqual([call.args[0].name for call in compile_package.call_args_list],
                             ['base', 'left', 'right', 'app'])

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
        diamond = entry_source(['base', 'left', 'base', 'right'], 'test')
        self.assertEqual(diamond.count('func.call @base()'), 1)
        self.assertLess(diamond.index('func.call @base()'), diamond.index('func.call @left()'))
