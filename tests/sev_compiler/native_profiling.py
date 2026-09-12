"""Exercise profiling on concrete native compiler binaries, bypassing launchers."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeProfiling(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='sev-native-profiling-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        rust = ROOT / 'package.pkg/release/sev'
        if not rust.exists():
            rust = ROOT / 'package.pkg/debug/sev'
        self.binaries = {
            'rust': Path(os.environ.get('SEVERIAN_NATIVE_RUST_COMPILER', rust)),
            'source': Path(os.environ.get('SEVERIAN_NATIVE_SOURCE_COMPILER', ROOT / 'sev_compiler/package.pkg/host/dev/bin/sev_compiler')),
        }
        # If a compiler accidentally delegates profiling to Python, fail even
        # on machines with Python installed. The harness itself uses Python.
        blocked = self.directory / 'bin'
        blocked.mkdir()
        for name in ('python', 'python3'):
            path = blocked / name
            path.write_text('#!/bin/sh\necho "unexpected Python delegation" >&2\nexit 99\n')
            path.chmod(0o755)
        self.environment = {**os.environ, 'SEVERIAN_SYSROOT': str(ROOT),
                            'PATH': str(blocked) + os.pathsep + os.environ['PATH']}

    def invoke(self, compiler, arguments, report):
        result = subprocess.run([str(self.binaries[compiler]), *map(str, arguments),
                                 '--profile-output', str(report)], cwd=ROOT,
                                env=self.environment, capture_output=True, text=True, timeout=60)
        data = json.loads((report / 'report.json').read_text())
        self.assertEqual(data['compiler'], compiler)
        self.assertGreaterEqual(data['user_seconds'], 0)
        self.assertGreaterEqual(data['system_seconds'], 0)
        self.assertGreater(data['peak_rss_kib'], 0)
        self.assertIn('CPU:', result.stderr)
        self.assertIn('Memory:', result.stderr)
        self.assertIn('Time:', result.stderr)
        self.assertNotIn('unexpected Python delegation', result.stderr)
        return result, data

    def test_memory_rankings_and_native_source_hint(self):
        source = self.directory / 'subject.sev'
        source.write_text('test:\n    assert(true)\n')
        tools = self.directory / 'bin'
        recorder = tools / 'heaptrack'
        recorder.write_text('#!/bin/sh\nprefix=$3\nshift 3\n"$@"\nstatus=$?\ntouch "$prefix.zst"\nexit "$status"\n')
        recorder.chmod(0o755)
        analyzer = tools / 'heaptrack_print'
        analyzer.write_text('#!/bin/sh\nmetric=allocations\noutput=\n'
                            'while [ "$#" -gt 0 ]; do\n'
                            'case "$1" in\n'
                            '--flamegraph-cost-type) metric=$2; shift ;;\n'
                            '--print-flamegraph) output=$2; shift ;;\n'
                            'esac\nshift\ndone\n'
                            'case "$metric" in allocations) cost=7 ;; peak) cost=64 ;; leaked) cost=32 ;; esac\n'
                            'printf "root;recursive;recursive;allocate; %s\\n" "$cost" > "$output"\n'
                            'printf "7 calls to allocation functions with 64B peak consumption from\\n  __sev_fn_123_test\\n    at %s:2\\n\\n" "$SEV_HINT_SOURCE"\n')
        analyzer.chmod(0o755)
        self.environment['SEV_HINT_SOURCE'] = str(source)
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                report = self.directory / compiler
                result, data = self.invoke(compiler, ['check', source, '--profile', 'memory'], report)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(data['exit_code'], 0)
                self.assertEqual((report / 'analysis.status').read_text(), '0\n')
                table_path, = report.glob('*.functions.tsv')
                table = table_path.read_text()
                self.assertIn('recursive\t0\t7\t0\t64\t0\t32\n', table)
                self.assertIn('allocate\t7\t7\t64\t64\t32\t32\n', table)
                hints_path, = report.glob('*.hints.txt')
                hints = hints_path.read_text()
                self.assertIn('Allocation hotspot: test', hints)
                self.assertIn(str(source) + ':2', hints)
                self.assertIn('assert(true)', hints)

    def test_directory_workers_reuse_prelude_with_isolated_subjects(self):
        sources = self.directory / 'sources'
        sources.mkdir()
        for index in range(2):
            (sources / f'{index}.sev').write_text(f'def value() -> int:\n    return {index}\ntest:\n    assert(value() == {index})\n')
        report = self.directory / 'directory-profile'
        result, data = self.invoke('source', ['test', sources, '--profile'], report)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('2 passed; 0 failed', result.stdout)
        self.assertIn('prelude-snapshot', (report / 'stages.tsv').read_text())
        workers = sorted(report.glob('stages.tsv.unit-*'))
        self.assertEqual(len(workers), 2)
        for worker in workers:
            self.assertIn('prelude-reused', worker.read_text())

    def test_direct_file_execution_and_live_native_accounting(self):
        source = ROOT / 'docs/examples/01-types/01-basic/00-constants.sev'
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                result, data = self.invoke(compiler, [source, '--profile'], self.directory / compiler)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, '3\n3.1415926\n')
                self.assertEqual(data['exit_code'], 0)
                self.assertEqual(data['arguments'], [str(source)])

    def test_build_and_test_commands(self):
        source = self.directory / 'subject.sev'
        source.write_text('test:\n    assert(20 + 22 == 42)\n')
        for compiler in self.binaries:
            for command in ('build', 'test'):
                with self.subTest(compiler=compiler, command=command):
                    result, data = self.invoke(compiler, [command, source, '--profile', 'time',
                                                         '-o', self.directory / f'{compiler}-{command}'],
                                               self.directory / f'{compiler}-{command}-report')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(data['mode'], 'time')

    def test_diagnostics_and_existing_report_are_preserved(self):
        missing = self.directory / 'missing.sev'
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                report = self.directory / compiler
                result, data = self.invoke(compiler, ['check', missing, '--profile'], report)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotEqual(data['exit_code'], 0)
                original = (report / 'report.json').read_bytes()
                repeated = subprocess.run([str(self.binaries[compiler]), 'check', str(missing),
                                           '--profile', '--profile-output', str(report)], cwd=ROOT,
                                          env=self.environment, capture_output=True, timeout=15)
                self.assertNotEqual(repeated.returncode, 0)
                self.assertEqual((report / 'report.json').read_bytes(), original)

    def test_invalid_source_still_writes_a_report(self):
        source = self.directory / 'invalid.sev'
        source.write_text('def main():\n    print(unknown_profile_test_symbol)\n')
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                result, data = self.invoke(compiler, ['check', source, '--profile'],
                                           self.directory / compiler)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(data['exit_code'], 1)
                self.assertIn('unknown_profile_test_symbol', result.stderr)

    def test_application_profile_flag_is_not_consumed(self):
        source = ROOT / 'docs/examples/01-types/01-basic/00-constants.sev'
        for compiler, binary in self.binaries.items():
            with self.subTest(compiler=compiler):
                report = self.directory / compiler
                result = subprocess.run([str(binary), 'run', str(source), '--profile',
                                         '--profile-output', str(report), '--', '--profile'],
                                        cwd=ROOT, env=self.environment, capture_output=True,
                                        text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, '3\n3.1415926\n')
                self.assertEqual(json.loads((report / 'report.json').read_text())['arguments'][-2:],
                                 ['--', '--profile'])

    def test_native_capture_reexecutes_and_preserves_failure(self):
        # A profiler protocol fixture tests dispatch without requiring kernel
        # perf permissions. Actual allocation capture is checked separately.
        perf = self.directory / 'bin' / 'perf'
        perf.write_text('#!/bin/sh\ncase "$1" in\nrecord)\n'
                        '  while [ "$1" != -- ]; do shift; done\n'
                        '  shift\n  exec "$@"\n  ;;\n'
                        'report) printf "test analysis\\n" ;;\nesac\n')
        perf.chmod(0o755)
        for compiler in self.binaries:
            with self.subTest(compiler=compiler):
                report = self.directory / compiler
                result, data = self.invoke(compiler, ['check', self.directory / 'missing.sev',
                                                      '--profile', 'cpu'], report)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(data['exit_code'], 1)
                self.assertEqual((report / 'analysis.status').read_text(), '0\n')
                self.assertEqual((report / 'cpu.txt').read_text(), 'test analysis\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
