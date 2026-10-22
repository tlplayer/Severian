"""Package test blocks execute in isolated c processes built through MLIR."""
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

from py_compiler.packaging.package import build, encode, write_atomic
from py_compiler.lir.src.lib import compile_object, companion
from py_compiler.mlir.src.lib import symbol_ref, verifier_path


def build_package(root, *, include_tests=False, _active=(), _built=None, _manifest=None, _packages=None, _registry_cache=None, **options):
    root = Path(root).resolve()
    if root in _active:
        raise ValueError('cyclic package dependency: ' + ' -> '.join(map(str, (*_active, root))))
    built = {} if _built is None else _built
    manifest = _manifest if _manifest is not None else json.loads((root / 'package.json').read_text())
    if _packages is not None:
        return build(root, include_tests=include_tests, _manifest=manifest, _packages=_packages, **options)
    from py_compiler.packaging.executable import DemandPackages
    dependency_options = {key: value for key, value in options.items()
                          if key in ('target', 'pointer_bits', 'jobs', 'mlir_opt')}
    packages = DemandPackages(root, manifest, built=built, registry_cache=_registry_cache,
                              **dependency_options)
    return build(root, include_tests=include_tests, _manifest=_manifest, _packages=packages, **options)



@dataclass(frozen=True)
class TestEntry:
    source: str
    target: str
    pointer_bits: int
    dependencies: tuple = ()

    def text(self):
        return self.source


def entry_source(initializers, test, *, benchmark=False):
    if benchmark:
        return benchmark_entry(initializers, test)
    calls = [*dict.fromkeys(initializers), test]
    declarations = [f'  func.func private {symbol_ref(symbol)}()' for symbol in dict.fromkeys(calls)]
    instructions = [f'    func.call {symbol_ref(symbol)}() : () -> ()' for symbol in calls]
    return ('module {\n' + '\n'.join(declarations) + '\n  func.func @main() -> i32 {\n'
            + '\n'.join(instructions) + '\n    %zero = arith.constant 0 : i32\n'
            '    func.return %zero : i32\n  }\n}\n')


def benchmark_entry(initializers, test):
    """Time the c body after warmup, excluding launch and initialization."""
    calls = list(dict.fromkeys(initializers))
    declarations = '\n'.join(f'func.func private {symbol_ref(symbol)}()' for symbol in dict.fromkeys([*calls, test]))
    initializations = '\n'.join(f'func.call {symbol_ref(symbol)}() : () -> ()' for symbol in calls)
    warmup = '\n'.join(f'func.call {symbol_ref(test)}() : () -> ()' for _ in range(3))
    samples = []
    for index in range(10):
        samples.append(f'''%start{index} = func.call @__sev_bench_now() : () -> i64
func.call {symbol_ref(test)}() : () -> ()
%end{index} = func.call @__sev_bench_now() : () -> i64
%duration{index} = arith.subi %end{index}, %start{index} : i64
%slot{index} = llvm.getelementptr %samples[{index}] : (!llvm.ptr) -> !llvm.ptr, i64
llvm.store %duration{index}, %slot{index} : i64, !llvm.ptr''')
    return '''module {
llvm.mlir.global private constant @__sev_bench_key("SEV_BENCHMARK_RESULT\\00") : !llvm.array<21 x i8>
func.func private @clock_gettime(i32, !llvm.ptr) -> i32
func.func private @getenv(!llvm.ptr) -> !llvm.ptr
func.func private @creat(!llvm.ptr, i32) -> i32
func.func private @write(i32, !llvm.ptr, i64) -> i64
func.func private @close(i32) -> i32
func.func private @__sev_bench_now() -> i64 {
    %two = arith.constant 2 : i64
    %stamp = llvm.alloca %two x i64 : (i64) -> !llvm.ptr
    %clock = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %status = func.call @clock_gettime(%clock, %stamp) : (i32, !llvm.ptr) -> i32
    %ok = arith.cmpi eq, %status, %zero : i32
    cf.assert %ok, "benchmark monotonic clock failed"
    %seconds = llvm.load %stamp : !llvm.ptr -> i64
    %subsecond = llvm.getelementptr %stamp[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %nanos = llvm.load %subsecond : !llvm.ptr -> i64
    %scale = arith.constant 1000000000 : i64
    %whole = arith.muli %seconds, %scale : i64
    %value = arith.addi %whole, %nanos : i64
    func.return %value : i64
}
''' + declarations + '\nfunc.func @main() -> i32 {\n' + initializations + '\n' + warmup + '''
    %count = arith.constant 10 : i64
    %samples = llvm.alloca %count x i64 : (i64) -> !llvm.ptr
''' + '\n'.join(samples) + '''
    %key = llvm.mlir.addressof @__sev_bench_key : !llvm.ptr
    %path = func.call @getenv(%key) : (!llvm.ptr) -> !llvm.ptr
    %null = llvm.mlir.zero : !llvm.ptr
    %configured = llvm.icmp "ne" %path, %null : !llvm.ptr
    cf.assert %configured, "benchmark result path is missing"
    %mode = arith.constant 384 : i32
    %file = func.call @creat(%path, %mode) : (!llvm.ptr, i32) -> i32
    %zero = arith.constant 0 : i32
    %opened = arith.cmpi sge, %file, %zero : i32
    cf.assert %opened, "benchmark result file could not be opened"
    %bytes = arith.constant 80 : i64
    %written = func.call @write(%file, %samples, %bytes) : (i32, !llvm.ptr, i64) -> i64
    %complete = arith.cmpi eq, %written, %bytes : i64
    cf.assert %complete, "benchmark result write failed"
    %status = func.call @close(%file) : (i32) -> i32
    %closed = arith.cmpi eq, %status, %zero : i32
    cf.assert %closed, "benchmark result close failed"
    func.return %zero : i32
}
}
'''


def test_passed(result, diagnostic=None, stdout_contains=(), stdout_excludes=(), output_expectations=()):
    for expectation in output_expectations:
        stream, operator = expectation['stream'], expectation['operator']
        if stream not in ('stdout', 'stderr') or operator not in ('==', '!='):
            raise ValueError('invalid output expectation')
        equal = getattr(result, stream) == expectation['value']
        if equal != (operator == '=='):
            return False
    if any(text not in result.stdout for text in stdout_contains) or any(text in result.stdout for text in stdout_excludes):
        return False
    if diagnostic is None:
        return result.returncode == 0
    return result.returncode == -signal.SIGABRT and diagnostic in result.stdout + result.stderr


def timeout_output(value):
    return value.decode(errors='replace') if isinstance(value, bytes) else value or ''


def test_environment(environ, sanitize, *, memory=False):
    env = dict(environ)
    if sanitize:
        # Symbolize from local debug information. Remote symbol downloads can
        # turn a sanitizer finding into a test timeout with no diagnostic.
        env['DEBUGINFOD_URLS'] = ''
        # Honor explicit runtime settings (e.g. LSan under ptrace), while
        # keeping memory errors fatal and leak checking enabled by default.
        options = env.get('ASAN_OPTIONS', '')
        env['ASAN_OPTIONS'] = 'detect_leaks=1:' + options + ':halt_on_error=1'
    if memory:
        env['ASAN_OPTIONS'] = env.get('ASAN_OPTIONS', '') + ':detect_leaks=1:halt_on_error=1:detect_stack_use_after_return=1:exitcode=86'
        env['LSAN_OPTIONS'] = env.get('LSAN_OPTIONS', '') + ':exitcode=86'
    return env


def memory_result(stderr):
    diagnostics = re.findall(r'(?:ERROR|SUMMARY): (?:AddressSanitizer|LeakSanitizer):[^\n]*', stderr)
    leaks = re.search(r'(\d+) byte\(s\) leaked in (\d+) allocation\(s\)', stderr)
    return {'backend': 'address+leak', 'errors': sum(d.startswith('ERROR:') for d in diagnostics),
            'leaks': int(leaks[2]) if leaks else 0, 'leaked-bytes': int(leaks[1]) if leaks else 0,
            'diagnostics': diagnostics}


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def execute(binary, timeout, env, *, seekable_stdout=False, cwd=None):
    if seekable_stdout:
        import tempfile
        command = list(map(str, binary)) if isinstance(binary, (list, tuple)) else [str(binary)]
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.PIPE,
                                       env=env, cwd=cwd, preexec_fn=no_core, start_new_session=True)
            try:
                _, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as failure:
                os.killpg(process.pid, signal.SIGKILL)
                _, stderr = process.communicate()
                output.seek(0)
                failure.output = output.read().decode(errors='replace')
                failure.stderr = stderr.decode(errors='replace')
                raise
            output.seek(0)
            return subprocess.CompletedProcess(command, process.returncode,
                output.read().decode(errors='replace'), stderr.decode(errors='replace'))
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
                                   env=env, cwd=cwd, preexec_fn=no_core, start_new_session=True)
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
        except subprocess.TimeoutExpired as failure:
            os.killpg(process.pid, signal.SIGKILL)
            _, stderr = process.communicate()
            reader.join(timeout=1)
            failure.output = b''.join(chunks).decode(errors='replace')
            failure.stderr = stderr.decode(errors='replace')
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
    status = 'SKIP' if record.get('skipped') else 'PASS' if record['passed'] else 'FAIL'
    reason = ': ' + record['reason'] if record.get('skipped') else ''
    if not record['passed']:
        detail = record.get('error')
        if not detail:
            failed = next((case for case in record.get('campaign', {}).get('cases', []) if not case['passed']), {})
            detail = failed.get('error')
        diagnostics = record.get('memory', {}).get('diagnostics', [])
        if diagnostics:
            detail = (detail + '; ' if detail else '') + diagnostics[0]
        if detail:
            reason = ': ' + detail
    return f"{status} {record['label'] or fallback} ({address}){reason}"


def execute_test_case(stem, test, *, timeout, env, backtrace=False, script=None, cwd=None):
    if test.get('skip'):
        return [{'label': test['label'], 'mode': test['mode'], 'passed': True,
                 'skipped': True, 'reason': test['skip'], 'exit': None, 'stdout': '', 'stderr': ''}]
    records = []
    count = test.get('repeat', 1)
    deadline = test.get('timeout', timeout)
    for iteration in range(count):
        label = test['label'] + (f' [run {iteration + 1}/{count}]' if count > 1 else '')
        record = {'label': label, 'mode': test['mode'], 'iteration': iteration + 1, 'timeout': deadline}
        if test.get('fuzz'):
            record['fuzz'] = {**test['fuzz'], 'reproduction-command': str(stem)}
        benchmark_path = stem.with_suffix(f'.run-{iteration + 1}.benchmark.bin')
        snapshot_path = stem.with_suffix(f'.run-{iteration + 1}.snapshots.bin')
        snapshot_path.unlink(missing_ok=True)
        execution_env = {**env, 'SEV_SNAPSHOT_RECORDS': str(snapshot_path)}
        if 'SEV_COVERAGE_FILE' in env:
            coverage_path = stem.with_suffix(f'.run-{iteration + 1}.coverage.bin')
            coverage_path.unlink(missing_ok=True)
            execution_env['SEV_COVERAGE_FILE'] = str(coverage_path)
            record['_coverage-path'] = str(coverage_path)
        if test['mode'] == 'bench':
            benchmark_path.unlink(missing_ok=True)
            execution_env['SEV_BENCHMARK_RESULT'] = str(benchmark_path)
        try:
            trace = ''
            if backtrace:
                process, trace = debug_execute(stem, deadline, execution_env, script)
            else:
                process = execute(stem, deadline, execution_env, seekable_stdout=True, cwd=cwd)
            record.update(passed=test_passed(process, test.get('expected-failure'), test.get('stdout-contains', ()),
                                            test.get('stdout-excludes', ()), test.get('output-expectations', ())),
                          exit=process.returncode, stdout=process.stdout, stderr=process.stderr, backtrace=trace)
            if test['mode'] == 'bench' and record['passed']:
                import statistics
                import struct
                measurements = benchmark_path.read_bytes() if benchmark_path.exists() else b''
                if len(measurements) != 80:
                    record.update(passed=False, error='benchmark did not produce ten c timing samples')
                else:
                    samples = struct.unpack('=10q', measurements)
                    if min(samples) < 0:
                        record.update(passed=False, error='benchmark clock moved backwards')
                    else:
                        record['benchmark'] = {'warmup': 3, 'samples-ns': list(samples),
                            'median-ns': statistics.median(samples), 'minimum-ns': min(samples),
                            'measurement': 'c test body; process launch and module initialization excluded'}
            if trace:
                stem.with_suffix(f'.run-{iteration + 1}.backtrace.txt' if count > 1 else '.backtrace.txt').write_text(trace)
        except subprocess.TimeoutExpired as failure:
            record.update(passed=False, exit=None, error=f'test exceeded its {deadline:g}s deadline',
                          stdout=timeout_output(failure.output), stderr=timeout_output(failure.stderr))
        if snapshot_path.exists():
            record['snapshot-records'] = str(snapshot_path)
        records.append(record)
        if not record['passed']:
            break
    return records


def check_snapshots(record, root, update=False):
    """Compare bytes only after a successful c execution; updates are explicit."""
    if not record.get('passed') or 'snapshot-records' not in record:
        return
    import struct
    data = Path(record['snapshot-records']).read_bytes()
    package_root = Path(root).resolve()
    directory = (package_root / 'snapshots').resolve()
    position, pending, names = 0, [], set()
    try:
        if not directory.is_relative_to(package_root):
            raise ValueError('snapshot directory escapes the package')
        while position < len(data):
            if len(data) - position < 16:
                raise ValueError('truncated snapshot header')
            name_size, value_size = struct.unpack_from('=QQ', data, position)
            position += 16
            if name_size > len(data) - position or value_size > len(data) - position - name_size:
                raise ValueError('truncated snapshot value')
            name = data[position:position + name_size].decode('utf-8')
            position += name_size
            value = data[position:position + value_size]
            position += value_size
            path = Path(name)
            if not name or path.is_absolute() or '..' in path.parts or '\0' in name:
                raise ValueError('snapshot name must be a relative path inside snapshots/')
            destination = (directory / path).resolve()
            if not destination.is_relative_to(directory) or destination == directory:
                raise ValueError('snapshot path escapes snapshots/')
            if any(existing == destination for _, existing, _ in pending):
                raise ValueError('duplicate snapshot name in one test: ' + name)
            names.add(name)
            pending.append((name, destination, value))
        failures = []
        for name, destination, value in pending:
            if update:
                destination.parent.mkdir(parents=True, exist_ok=True)
                import tempfile
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    output.write(value)
                    temporary = output.name
                os.replace(temporary, destination)
            elif not destination.is_file():
                failures.append(name + ': missing snapshot; run test --update-snapshots to accept it')
            elif destination.read_bytes() != value:
                import difflib
                expected = destination.read_bytes().decode('utf-8', errors='replace').splitlines(keepends=True)
                actual = value.decode('utf-8', errors='replace').splitlines(keepends=True)
                failures.append(name + ': snapshot differs\n' + ''.join(difflib.unified_diff(expected, actual,
                    fromfile='expected/' + name, tofile='actual/' + name)))
        record['snapshots'] = {'names': sorted(names), 'updated': update}
        if failures:
            record.update(passed=False, error='\n'.join(failures))
    except (ValueError, UnicodeError, OSError) as error:
        record.update(passed=False, error=str(error))


def test_package(root, *, sanitize=None, memory=False, timeout=30, backtrace=False, parallel=1, update_snapshots=False,
                 coverage=False, fuzz=False, profile=False, differential=False, progress=None, **options):
    root = Path(root).resolve()
    source = None
    if not root.is_dir() and (root.is_file() or root.suffix == '.sev'):
        source = root.resolve(strict=True)
        if not source.is_file() or source.suffix != '.sev':
            raise ValueError('test source must be a .sev file')
        from py_compiler.packaging.executable import source_package
        root, manifest = source_package(source)
        manifest.setdefault('package', {'name': 'source.tests', 'version': '0.0.0'})
        manifest['sources'] = [source.relative_to(root).as_posix()]
        options['_manifest'] = manifest
    if backtrace and (sanitize or memory):
        raise ValueError('run --backtrace and --sanitize separately; LeakSanitizer cannot run under ptrace')
    target = options.get('target', 'x86_64-unknown-linux-gnu')
    machine = {'AMD64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine(), platform.machine())
    if platform.system() != 'Linux' or target.split('-')[0] != machine or 'linux' not in target:
        raise ValueError('c package tests require a matching Linux host target')
    if timeout <= 0:
        raise ValueError('test timeout must be positive')
    if parallel < 1:
        raise ValueError('parallel worker count must be positive')
    if source is not None:
        from py_compiler.packaging.executable import DemandPackages
        if progress:
            progress('Preparing tests: ' + str(source))
        options['_packages'] = DemandPackages(root, manifest, target=target,
                                              pointer_bits=options.get('pointer_bits', 64),
                                              jobs=options.get('jobs', 4), mlir_opt=options.get('mlir_opt'),
                                              progress=progress)
    if progress:
        progress('Compiling tests: ' + str(source or root))
    result = build_package(root, include_tests=True, **options)
    if not result['success']:
        return result
    package = Path(root).resolve() / 'package.pkg'
    build_id = result['output-slot']
    metadata = json.loads((package / 'metadata/realizations' / (build_id + '.json')).read_text())
    symbols = json.loads((package / 'metadata/symbols' / (build_id + '.json')).read_text())
    requested_memory, requested_sanitize = memory, sanitize
    memory = memory or any((symbol.get('test') or {}).get('memory') for symbol in symbols)
    if memory:
        if backtrace:
            raise ValueError('test with memory cannot run under --backtrace (LeakSanitizer requires no ptrace)')
        sanitize = 'address'
    from py_compiler.packaging.coverage import Coverage, RUNTIME, instrument
    collecting = coverage or fuzz or differential
    collector = Coverage(root, result.get('build-id')) if collecting else None
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
    coverage_runtime = []
    if collector:
        coverage_objects = []
        excluded = {s['entry'] for s in symbols if s.get('test')}
        runtime_source, runtime_object = output / 'coverage.c', output / 'coverage-runtime.o'
        runtime_source.write_text(RUNTIME)
        coverage_runtime = [str(runtime_object)]
        for index, obj in enumerate(objects):
            llvm = Path(str(obj).replace('/object/', '/ir/')).with_suffix('.ll')
            if obj not in result['objects'] or not llvm.exists():
                coverage_objects.append(obj)
                continue
            source, destination = output / f'coverage-{index}.ll', output / f'coverage-{index}.o'
            source.write_text(instrument(llvm.read_text(), root, collector.points, excluded, existing_sources_only=True))
            tool([companion('llc', optimizer), '-filetype=obj', '-relocation-model=pic',
                  '-mtriple=' + target, source, '-o', destination])
            coverage_objects.append(str(destination))
        slots = max(64, 1 << (2 * len(collector.points)).bit_length())
        tool([linker, '-c', '-O2', '-fPIC', f'-DSEV_COVERAGE_SLOTS={slots}', runtime_source, '-o', runtime_object])
        objects = coverage_objects
        from hashlib import sha256
        for point in collector.points.values():
            name = point['file']
            collector.hashes[name] = sha256((Path(root) / name).read_bytes()).hexdigest()
    def sanitized_objects(inputs, prefix):
        instrumented = []
        for index, obj in enumerate(inputs):
            llvm = Path(str(obj).replace('/object/', '/ir/')).with_suffix('.ll')
            if not llvm.exists():
                instrumented.append(obj)
                continue
            text = re.sub(r'^(define [^\n]*?)( !dbg ![0-9]+)? \{$',
                          r'\1 sanitize_address\2 {', llvm.read_text(), flags=re.MULTILINE)
            source, lowered, destination = (output / f'{prefix}-{index}{suffix}' for suffix in ('.ll', '.instrumented.ll', '.o'))
            source.write_text(text)
            tool([companion('opt', optimizer), '-mtriple=' + target, '-passes=asan', '-S', source, '-o', lowered])
            tool([companion('llc', optimizer), '-filetype=obj', '-relocation-model=pic', '-mtriple=' + target, lowered, '-o', destination])
            instrumented.append(str(destination))
        return instrumented
    unchecked_objects = dict(zip(original_objects, objects))
    baseline_objects = original_objects
    if sanitize:
        objects = sanitized_objects(objects, 'asan')
        if collector and any((s.get('test') or {}).get('mode') in ('profile', 'bench') for s in symbols):
            baseline_objects = sanitized_objects(original_objects, 'asan-baseline')
    measured_objects = dict(zip(original_objects, baseline_objects))
    linked_objects = dict(zip(original_objects, objects))
    partitions = {part['submodule']: part for part in metadata.get('partitions', [])}
    symbol_owners = {name: key for key, part in partitions.items() for name in part['symbols']}
    executions = []
    for index, symbol in enumerate(s for s in symbols if s.get('test')):
        test = {**symbol['test'], '_sanitize': bool(requested_sanitize or requested_memory or symbol['test'].get('memory'))}
        if test.get('campaign') and not test['campaign'].get('replay-cases') and not {'fuzz':fuzz,'profile':profile,'differential':differential}.get(test['mode'],False):
            continue
        if test.get('fuzz') and not fuzz and test['fuzz']['case'] != 0:
            continue
        stem = output / f'test-{index:04}'
        if test.get('skip'):
            executions.append((stem, test, [], symbol['entry']))
            continue
        if progress:
            progress('Linking test: ' + str(test.get('label') or symbol['entry']))
        source = entry_source(metadata.get('dependency-initializers', []) + metadata['initializers'], symbol['entry'],
                              benchmark=test['mode'] == 'bench')
        artifact = compile_object(TestEntry(source, target, options.get('pointer_bits', 64)), optimizer)
        stem.with_suffix('.mlir').write_text(source)
        stem.with_suffix('.o').write_bytes(artifact.object_bytes)
        # LLVM lowers floating-point operations such as trunc to libm calls.
        # C package tests already require a Linux host above.
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
        from py_compiler.syntax.complex.profile import linker_options
        tool([linker, *(['-fsanitize=address'] if test['_sanitize'] else []), *linker_options(selected_objects), stem.with_suffix('.o'),
              *((linked_objects if test['_sanitize'] else unchecked_objects)[obj] for obj in selected_objects), *coverage_runtime, '-pthread', '-lm', '-o', stem])
        if collector and test['mode'] in ('profile', 'bench'):
            coverage_binary = stem.with_suffix('.coverage-exe')
            shutil.copyfile(stem, coverage_binary)
            coverage_binary.chmod(0o700)
            test = {**test, '_coverage-binary': str(coverage_binary)}
            tool([linker, *(['-fsanitize=address'] if test['_sanitize'] else []), *linker_options(selected_objects), stem.with_suffix('.o'),
                  *((measured_objects[obj] if test['_sanitize'] else obj) for obj in selected_objects), '-pthread', '-lm', '-o', stem])
        executions.append((stem, test, selected_objects, symbol['entry']))
    from concurrent.futures import ThreadPoolExecutor
    discoveries, campaign_hits, campaign_failures, baseline_hits = [], {}, set(), set()
    def record_results(execution, records):
        _, test, objects, entry = execution
        for record in records:
            if test.get('_sanitize') and not record.get('skipped'):
                details = record.get('stderr', '') + '\n'.join(
                    case.get('stderr', '') for case in record.get('campaign', {}).get('cases', []))
                record['memory'] = memory_result(details)
                if record['memory']['diagnostics']:
                    record['passed'] = False
            check_snapshots(record, root, update_snapshots)
            if collector and not record.get('skipped'):
                hits = collector.add(record, record.pop('_coverage-path'))
                info = test.get('fuzz')
                if not info and record['passed']:
                    baseline_hits.update(hits)
                if fuzz and info:
                    campaign = (test['location']['file'], info['origin-label'])
                    seen = campaign_hits.setdefault(campaign, set(baseline_hits))
                    new = hits - seen
                    failure = (campaign, record.get('exit'), record.get('error'),
                               record.get('stdout'), record.get('stderr'))
                    novel_failure = not record['passed'] and failure not in campaign_failures
                    if novel_failure:
                        campaign_failures.add(failure)
                    if new or novel_failure:
                        confirmation = run(execution)[0]
                        from py_compiler.packaging.coverage import read_hits
                        confirmed_hits = set(read_hits(confirmation['_coverage-path']))
                        same_result = (confirmation['passed'] == record['passed']
                                       and confirmation.get('exit') == record.get('exit')
                                       and confirmation.get('error') == record.get('error')
                                       and confirmation.get('stdout') == record.get('stdout')
                                       and confirmation.get('stderr') == record.get('stderr'))
                        if same_result and (novel_failure or new <= confirmed_hits):
                            seen.update(hits)
                            discoveries.append({**info, 'file': test['location']['file'],
                                                'reason': 'failure' if novel_failure else 'coverage',
                                                'new-points': sorted(new)})
                        else:
                            record['fuzz']['persistence'] = 'unconfirmed; replay differed'
            record['location'] = test_location(root, test)
            record['objects'] = objects
            report['tests'].append(record)
            write_atomic(report_path, encode(report))
            if record.get('backtrace'):
                print(record['backtrace'], end='')
            print(format_test_result(record, entry))
    def run(execution):
        checked = execution[1].get('_sanitize')
        env = test_environment(os.environ, 'address' if checked else None,
                               memory=bool(requested_memory or execution[1].get('memory')))
        if execution[1].get('campaign'):
            from py_compiler.packaging.campaigns import run as run_campaign
            stem, test, _, _ = execution
            trace = stem.with_suffix('.campaign-coverage.bin')
            from py_compiler.packaging.coverage import artifact_slot
            spec = {**test['campaign'], 'artifact-slot':artifact_slot(result.get('build-id'))}
            campaign, found = run_campaign(stem, spec, execute, test.get('timeout', timeout), env,
                                           str(trace) if collector and test['mode'] in ('fuzz','differential') else None)
            discoveries.extend({**item, 'file':test['location']['file']} for item in found)
            record = {'label':test['label'],'mode':test['mode'],'passed':campaign['passed'],
                      'campaign':campaign,'stdout':'','stderr':''}
            if collector:
                if not trace.exists(): trace.write_bytes(b'')
                record['_coverage-path'] = str(trace)
            return [record]
        trace = execution[0].with_suffix('.coverage.bin')
        trace.unlink(missing_ok=True)
        records = execute_test_case(execution[0], execution[1], timeout=timeout,
                                    env={**env, **({'SEV_COVERAGE_FILE': str(trace)} if collector else {})},
                                    backtrace=backtrace, script=script if backtrace else None)
        if execution[1].get('_coverage-binary'):
            secondary = execute_test_case(Path(execution[1]['_coverage-binary']), execution[1],
                timeout=timeout, env={**env, 'SEV_COVERAGE_FILE': str(trace)})
            for record, instrumented in zip(records, secondary):
                record['_coverage-path'] = instrumented['_coverage-path']
                record['_coverage-partial'] = not instrumented['passed']
                if not instrumented['passed']:
                    record.update(passed=False, error='instrumented coverage execution failed',
                                  stderr=record.get('stderr', '') + instrumented.get('stderr', ''))
        return records
    if progress:
        progress('Running tests: ' + str(len(executions)))
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        pending = []
        def finish_pending():
            for execution, future in pending:
                record_results(execution, future.result())
            pending.clear()
        for execution in sorted(executions, key=lambda item: bool(item[1].get('fuzz'))):
            if execution[1].get('parallel') and parallel > 1:
                pending.append((execution, pool.submit(run, execution)))
            else:
                finish_pending()
                record_results(execution, run(execution))
        finish_pending()
    if collector:
        report['coverage'] = collector.save()
    if fuzz or profile or differential:
        from py_compiler.syntax.complex.fuzz import persist
        persistence=persist(root, discoveries, result.get('build-id'),mode='fuzz' if fuzz or profile else 'differential')
        for mode,enabled in (('fuzz',fuzz or profile),('differential',differential)):
            if enabled:
                identities={item['regression-id'] for item in discoveries if item.get('mode','fuzz')==mode}
                report[mode+'-persistence'] = [{**item,'appended':[i for i in item['appended'] if i in identities],
                                              'pending':[i for i in item['pending'] if i in identities]} for item in persistence]
    report['success'] = all(test['passed'] for test in report['tests'])
    write_atomic(report_path, encode(report))
    return {'success': report['success'], 'tests': len(report['tests']),
            'failed': sum(not t['passed'] for t in report['tests']),
            'skipped': sum(bool(t.get('skipped')) for t in report['tests']), 'report': str(report_path)}


import unittest


class SourceSelectionTests(unittest.TestCase):
    def test_build_compiles_selected_source_with_its_original_debug_path(self):
        import tempfile
        from unittest.mock import patch
        from py_compiler.frontend.src.lib import compile_source
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'selected.sev'
            source.write_text('test:\n    assert(unknown_selected_value)\n')
            (root / 'sibling.sev').write_text('invalid sibling')
            manifest = {'package': {'name': 'selection', 'version': '1.0.0'},
                        'sources': ['selected.sev']}
            with patch('py_compiler.packaging.package.compile_source', wraps=compile_source) as frontend:
                with patch('py_compiler.packaging.package.lower') as c:
                    result = build_package(root, include_tests=True, _manifest=manifest)
            self.assertFalse(result['success'])
            self.assertIn('unknown_selected_value', '\n'.join(result['sample']))
            self.assertEqual(frontend.call_count, 1)
            self.assertEqual(frontend.call_args.args[0], 'selected.sev')
            self.assertEqual(frontend.call_args.kwargs['debug_path'], str(source))
            c.assert_not_called()
            self.assertFalse((root / 'package.json').exists())

    def test_file_selects_only_itself_and_keeps_original_package(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'nested').mkdir()
            source = root / 'nested/selected.sev'
            source.write_text('import custom\ntest:\n    assert(true)\n')
            (root / 'broken.sev').write_text('this sibling must not compile')
            manifest = {'package': {'name': 'selection', 'version': '1.0.0'},
                        'sources': ['broken.sev', 'nested/selected.sev'],
                        'dependencies': {'custom': {'path': 'library'}}}
            original = encode(manifest)
            (root / 'package.json').write_text(original)
            failure = {'success': False, 'sample': ['selected diagnostic']}
            with patch(__name__ + '.build_package', return_value=failure) as builder:
                with patch('py_compiler.packaging.executable.prepare_dependencies') as prepare:
                    result = test_package(source, coverage=True, fuzz=True, jobs=2)
                prepare.assert_not_called()
                self.assertEqual(builder.call_args.kwargs['_packages'].resolved, {})
            self.assertIs(result, failure)
            self.assertEqual(builder.call_args.args, (root,))
            options = builder.call_args.kwargs
            self.assertTrue(options['include_tests'])
            self.assertEqual(options['jobs'], 2)
            self.assertEqual(options['_manifest']['sources'], ['nested/selected.sev'])
            self.assertEqual(options['_manifest']['dependencies']['custom'], {'path': 'library'})
            self.assertEqual((root / 'package.json').read_text(), original)
            self.assertEqual(source.read_text(), 'import custom\ntest:\n    assert(true)\n')

    def test_standalone_file_has_manifest_and_standard_prelude_in_memory(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'standalone.sev'
            source.write_text('test:\n    print(string(42))\n')
            with patch(__name__ + '.build_package', return_value={'success': False}) as builder:
                with patch('py_compiler.packaging.executable.prepare_dependencies'):
                    test_package(source)
            manifest = builder.call_args.kwargs['_manifest']
            self.assertEqual(manifest['sources'], ['standalone.sev'])
            self.assertTrue(manifest['package']['name'])
            self.assertEqual(manifest['dependencies'], {})
            self.assertTrue(callable(builder.call_args.kwargs['_packages'].resolve_prelude))
            self.assertFalse((root / 'package.json').exists())

    def test_source_override_does_not_replace_dependency_sources(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / 'library'
            dependency.mkdir()
            (dependency / 'package.json').write_text(encode({'sources': ['lib.sev']}))
            manifest = {'sources': ['selected.sev'], 'dependencies': {'lib': {'path': 'library'}}}
            (dependency / 'lib.sev').write_text('def answer() -> int:\n    return 42\n')
            dependency_manifest = {'package': {'name': 'library', 'version': '1.0.0'}, 'sources': ['lib.sev']}
            (dependency / 'package.json').write_text(encode(dependency_manifest))
            manifest['package'] = {'name': 'app', 'version': '1.0.0'}
            (root / 'selected.sev').write_text('import lib\ntest:\n    assert(lib.answer() == 42)\n')
            with patch(__name__ + '.build', wraps=build) as builder:
                result = build_package(root, include_tests=True, _manifest=manifest)
            self.assertTrue(result['success'], result)
            source_call, dependency_call = builder.call_args_list
            self.assertEqual(dependency_call.args, (dependency,))
            self.assertIsNone(dependency_call.kwargs['_manifest'])
            self.assertFalse(dependency_call.kwargs['include_tests'])
            self.assertEqual(source_call.args, (root,))
            self.assertIs(source_call.kwargs['_manifest'], manifest)
            self.assertTrue(source_call.kwargs['include_tests'])

    def test_missing_source_and_wrong_extension_fail_before_build(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = root / 'input.txt'
            other.write_text('test:\n    assert(true)\n')
            with patch(__name__ + '.build_package') as builder:
                with self.assertRaises(FileNotFoundError):
                    test_package(root / 'missing.sev')
                with self.assertRaisesRegex(ValueError, 'must be a .sev file'):
                    test_package(other)
                builder.assert_not_called()


class PackageTestRunnerTests(unittest.TestCase):
    def test_c_snapshots_require_explicit_updates_and_detect_changes(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='sev-snapshots-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(encode({'package': {'name': 'snapshots', 'version': '0.1.0'}}))
            (root / 'src/lib.sev').write_text('test "snapshots":\n'
                '    snapshot("nested/value.txt", "hello\\nworld")\n'
                '    snapshot("empty.txt", "")\n')
            missing = test_package(root)
            self.assertFalse(missing['success'])
            self.assertFalse((root / 'snapshots').exists())
            accepted = test_package(root, update_snapshots=True)
            self.assertTrue(accepted['success'], accepted)
            self.assertEqual((root / 'snapshots/nested/value.txt').read_bytes(), b'hello\nworld')
            matching = test_package(root)
            self.assertTrue(matching['success'], matching)
            (root / 'snapshots/nested/value.txt').write_text('different')
            changed = test_package(root)
            self.assertFalse(changed['success'])
            records = json.loads(Path(changed['report']).read_text())['tests']
            self.assertIn('snapshot differs', records[0]['error'])
            self.assertEqual((root / 'snapshots/nested/value.txt').read_text(), 'different')

    def test_snapshot_names_cannot_escape_the_baseline_directory(self):
        import struct
        import tempfile
        with tempfile.TemporaryDirectory(prefix='sev-snapshot-path-') as directory:
            root = Path(directory)
            artifact = root / 'records.bin'
            name = b'../outside.txt'
            artifact.write_bytes(struct.pack('=QQ', len(name), 1) + name + b'x')
            record = {'passed': True, 'snapshot-records': str(artifact)}
            check_snapshots(record, root, True)
            self.assertFalse(record['passed'])
            self.assertFalse((root / 'outside.txt').exists())

    def test_c_benchmark_warms_up_and_measures_body_only(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='sev-benchmark-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'package.json').write_text(encode({'package': {'name': 'benchmark', 'version': '0.1.0'}}))
            (root / 'src/lib.sev').write_text('calls = 0\ntest with bench "counter":\n'
                '    module.calls += 1\n    assert(module.calls <= 13)\n')
            result = test_package(root)
            self.assertTrue(result['success'], result)
            report = json.loads(Path(result['report']).read_text())
            timing = report['tests'][0]['benchmark']
            self.assertEqual(timing['warmup'], 3)
            self.assertEqual(len(timing['samples-ns']), 10)
            self.assertGreaterEqual(timing['minimum-ns'], 0)

    def test_repeat_deadlines_skip_and_failure(self):
        from unittest.mock import patch
        success = subprocess.CompletedProcess([], 0, '', '')
        failure = subprocess.CompletedProcess([], 1, '', 'failure')
        test = {'label': 'repeat', 'mode': 'unit', 'repeat': 3, 'timeout': 0.25}
        with patch(__name__ + '.execute', return_value=success) as run:
            records = execute_test_case(Path('binary'), test, timeout=30, env={})
            self.assertEqual(len(records), 3)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(run.call_args.args[1], 0.25)
        with patch(__name__ + '.execute', side_effect=[success, failure]) as run:
            records = execute_test_case(Path('binary'), test, timeout=30, env={})
            self.assertEqual(len(records), 2)
            self.assertFalse(records[-1]['passed'])
        with patch(__name__ + '.execute') as run:
            records = execute_test_case(Path('binary'), {**test, 'skip': 'no backend'}, timeout=30, env={})
            run.assert_not_called()
            self.assertTrue(records[0]['skipped'])
        with patch(__name__ + '.execute', side_effect=subprocess.TimeoutExpired('binary', 0.25)):
            records = execute_test_case(Path('binary'), test, timeout=30, env={})
            self.assertFalse(records[0]['passed'])
            self.assertIn('0.25s', records[0]['error'])

    def test_c_cases_repeat_skip_and_parallel(self):
        import tempfile
        source = '''test square(value, expected) with cases {(2, 4), (3, 9)}:
    assert(value * value == expected)
test with repeat(3) and timeout(2s) "repeat":
    assert(true)
test with skip("backend unavailable") "skip":
    unsupported_backend()
test with parallel "first":
    assert(true)
test with parallel "second":
    assert(true)
'''
        with tempfile.TemporaryDirectory(prefix='sev-runner-options-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(encode({'package': {'name': 'runner-options', 'version': '0.1.0'}}))
            result = test_package(root, parallel=2)
            self.assertTrue(result['success'], result)
            self.assertEqual(result['tests'], 8)
            self.assertEqual(result['skipped'], 1)

    def test_seekable_stdout_preserves_exact_bytes_and_restores_position(self):
        source = ('import os\n'
                  'os.write(1, b"first\\n")\n'
                  'end = os.lseek(1, 0, 1)\n'
                  'os.lseek(1, 0, 0)\n'
                  'assert os.read(1, end) == b"first\\n"\n'
                  'os.lseek(1, end, 0)\n'
                  'os.write(1, b"second\\n")\n')
        result = execute([sys.executable, '-c', source], 5, os.environ, seekable_stdout=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'first\nsecond\n')

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

    def test_sanitizer_symbolization_cannot_wait_on_remote_debug_servers(self):
        source = {'DEBUGINFOD_URLS': 'https://debug.invalid/', 'OTHER': 'value'}
        env = test_environment(source, 'address', memory=True)
        self.assertEqual(env['DEBUGINFOD_URLS'], '')
        self.assertIn('detect_leaks=1', env['ASAN_OPTIONS'])
        self.assertEqual(source['DEBUGINFOD_URLS'], 'https://debug.invalid/')
        self.assertEqual(test_environment(source, None), source)

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
            for name, text, dependencies in (
                ('library', 'def answer() -> int:\n    return 42\n', {}),
                ('app', 'import vector\ntest:\n    assert(vector.answer() == 42)\n', {'vector': {'path': '../library'}}),
            ):
                path = root / name
                (path / 'src').mkdir(parents=True)
                (path / 'src/lib.sev').write_text(text)
                (path / 'package.json').write_text(encode({'package': {'name': name, 'version': '1.0.0'}, 'dependencies': dependencies}))
            completed = []
            c_build = build
            def compile_package(path, **options):
                result = c_build(path, **options)
                completed.append((path.name, options['include_tests']))
                return result
            with patch(__name__ + '.build', side_effect=compile_package):
                result = build_package(root / 'app', include_tests=True)
            self.assertTrue(result['success'], result)
            self.assertEqual(completed, [('library', False), ('app', True)])

    def test_diamond_builds_shared_dependency_once(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = {'app': ('left', 'right'), 'left': ('base',), 'right': ('base',), 'base': ()}
            for name, dependencies in graph.items():
                path = root / name
                (path / 'src').mkdir(parents=True)
                (path / 'src/lib.sev').write_text(''.join('import ' + dep + '\n' for dep in dependencies) + 'def answer() -> int:\n    return 42\n')
                (path / 'package.json').write_text(encode({'package': {'name': name, 'version': '1.0.0'}, 'dependencies': {
                    dep: {'path': '../' + dep} for dep in dependencies}}))
            completed = []
            c_build = build
            def compile_package(path, **options):
                result = c_build(path, **options)
                completed.append(path.name)
                return result
            with patch(__name__ + '.build', side_effect=compile_package):
                result = build_package(root / 'app')
            self.assertTrue(result['success'], result)
            self.assertEqual(completed, ['base', 'left', 'right', 'app'])

    def test_referenced_dependency_cycles_fail(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text('import loop\n')
            (root / 'package.json').write_text(encode({'package': {'name': 'cycle', 'version': '1.0.0'},
                                                      'dependencies': {'loop': {'path': '.'}}}))
            result = build_package(root)
            self.assertFalse(result['success'])
            self.assertIn('cyclic package dependency', '\n'.join(result['sample']))

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


class OutputExpectationTests(unittest.TestCase):
    def test_exact_stream_expectations(self):
        process = subprocess.CompletedProcess([], 0, 'hello\n', '')
        expected = ({'stream': 'stderr', 'operator': '==', 'value': ''},)
        self.assertTrue(test_passed(process, output_expectations=expected))
        process.stderr = 'unexpected warning'
        self.assertFalse(test_passed(process, output_expectations=expected))
        self.assertTrue(test_passed(process, output_expectations=({'stream': 'stdout', 'operator': '!=', 'value': ''},)))

    def test_captured_output_is_required_even_when_program_exits_successfully(self):
        process = subprocess.CompletedProcess([], 0, stdout='hello\n', stderr='')
        self.assertTrue(test_passed(process, stdout_contains=('hello',)))
        self.assertFalse(test_passed(process, stdout_contains=('missing',)))
        self.assertFalse(test_passed(process, stdout_contains=('hello', 'missing')))


    def test_forbidden_output_fails_even_on_success(self):
        process = subprocess.CompletedProcess([], 0, stdout='good\nbad\n', stderr='')
        self.assertFalse(test_passed(process, stdout_contains=('good',), stdout_excludes=('bad',)))
        self.assertTrue(test_passed(process, stdout_contains=('good',), stdout_excludes=('missing',)))
        self.assertFalse(test_passed(process, stdout_excludes=('',)))


class MemoryRunnerTests(unittest.TestCase):
    def test_memory_cannot_disable_leaks_or_successfully_exit_on_violation(self):
        env = test_environment({'ASAN_OPTIONS': 'detect_leaks=0:exitcode=0',
                                'LSAN_OPTIONS': 'exitcode=0'}, 'address', memory=True)
        options = dict(item.split('=', 1) for item in env['ASAN_OPTIONS'].split(':') if item)
        self.assertEqual(options['detect_leaks'], '1')
        self.assertEqual(options['exitcode'], '86')
        self.assertTrue(env['LSAN_OPTIONS'].endswith('exitcode=86'))

    def test_leak_and_invalid_access_diagnostics(self):
        result = memory_result('ERROR: LeakSanitizer: detected memory leaks\n'
            'SUMMARY: AddressSanitizer: 32 byte(s) leaked in 2 allocation(s).')
        self.assertEqual((result['errors'], result['leaks'], result['leaked-bytes']), (1, 2, 32))
        self.assertEqual(memory_result('ERROR: AddressSanitizer: heap-use-after-free')['errors'], 1)
        self.assertEqual(memory_result('')['errors'], 0)

    def test_c_memory_detects_leak_and_invalid_access_in_called_functions(self):
        import tempfile
        for operation, diagnostic in [
                ('values[100] = 42\n    drop values', 'AddressSanitizer'),
                ('values[0] = 42', 'LeakSanitizer')]:
            with self.subTest(diagnostic=diagnostic), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'main.sev'
                source.write_text('def broken():\n    unsafe:\n'
                    '        values = allocate[int](1)\n        ' + operation.replace('\n    ', '\n        ') +
                    '\ntest with memory "called function":\n    broken()\n')
                result = test_package(source)
                self.assertFalse(result['success'], result)
                report = json.loads(Path(result['report']).read_text())
                self.assertIn(diagnostic, report['tests'][0]['stderr'])
                self.assertGreater(report['tests'][0]['memory']['errors'], 0)


class TimeoutDiagnosticTests(unittest.TestCase):
    def test_deadline_keeps_stdout_and_memory_diagnostic(self):
        from unittest.mock import patch
        failure = subprocess.TimeoutExpired('test', 0.1, output=b'last operation\n',
            stderr=b'ERROR: AddressSanitizer: heap-use-after-free\n')
        with patch(__name__ + '.execute', side_effect=failure):
            record = execute_test_case(Path('/tmp/sev-timeout'), {'mode':'unit', 'label':'memory'},
                                       timeout=0.1, env={})[0]
        self.assertEqual(record['stdout'], 'last operation\n')
        self.assertIn('heap-use-after-free', record['stderr'])
        self.assertFalse(record['passed'])
        record['location'] = {'file':'main.sev', 'line':1, 'column':1}
        record['memory'] = memory_result(record['stderr'])
        displayed = format_test_result(record, hyperlinks=False)
        self.assertIn('0.1s deadline', displayed)
        self.assertIn('heap-use-after-free', displayed)

    def test_execute_drains_output_before_raising_timeout(self):
        command = [sys.executable, '-c',
                   'import os,time; os.write(1,b"before timeout"); os.write(2,b"diagnostic"); time.sleep(10)']
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            execute(command, 0.2, os.environ, seekable_stdout=True)
        self.assertEqual(caught.exception.output, 'before timeout')
        self.assertEqual(caught.exception.stderr, 'diagnostic')
