#!/usr/bin/env python3
"""Runner modifiers must execute repetitions and interrupt blocked bodies."""
import time
import unittest

from migration import COMPILER, MigrationCase, ROOT


class RunnerModes(MigrationCase):
    def test_benchmark_warms_up_measures_and_reports(self):
        path = self.write('''
            test with bench "sample":
                print("iteration")
        ''')
        output = self.succeeds([COMPILER, 'test', path, '--sysroot', ROOT,
                                '-o', self.directory / 'benchmark'])
        self.assertEqual(output.count('iteration\n'), 11)
        self.assertRegex(output, r'benchmark: sample iterations: 10 total: [0-9]+ ns mean:')

    def test_documented_benchmark(self):
        path = ROOT / 'docs/examples/03-testing/02-with-tests/03-benchmark.sev'
        output = self.succeeds([COMPILER, 'test', path, '--sysroot', ROOT,
                                '-o', self.directory / 'fib'])
        self.assertIn('benchmark: recursive fibonacci throughput iterations: 10', output)

    def test_documented_runner_modes(self):
        for name in ['11-timeout', '13-repeat-parallel', '20-eventually']:
            with self.subTest(example=name):
                self.native((ROOT / 'docs/examples/03-testing/02-with-tests' / (name + '.sev')).read_text())

    def test_repetitions_have_fresh_locals(self):
        self.native('''
            test with repeat(3):
                value := 0
                value += 1
                print(value)
        ''', expected='1\n1\n1\n')

    def test_timeout_composes_with_repeat_and_cleanup(self):
        self.native('''
            test with repeat(3) and timeout(1s):
                defer print("cleanup")
                print("body")
        ''', expected='body\ncleanup\n' * 3)

    def test_blocked_body_and_cleanup_are_bounded(self):
        for body in ['pause()', 'defer pause()']:
            with self.subTest(body=body):
                subject = self.write('''
                    @c(symbol="sleep")
                    def sleep(seconds: u32) -> u32
                    def pause():
                        sleep(10)
                    test with timeout(25ms) "blocked test":
                        ''' + body)
                executable = self.directory / 'blocked'
                self.succeeds([COMPILER, 'test', subject, '--emit', 'mlir', '--sysroot', ROOT])
                # Compile through the same native path used by the CLI. Timing
                # includes compilation, with ample margin below the 10s sleep.
                started = time.monotonic()
                result = self.invoke([COMPILER, 'test', subject, '--sysroot', ROOT, '-o', executable])
                self.assertLess(time.monotonic() - started, 8)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('test timeout: blocked test exceeded 25000000 ns', result.stderr)

    def test_child_failure_is_not_reported_as_success(self):
        path = self.write('''
            test with timeout(1s):
                assert(false, "child failed")
        ''')
        result = self.invoke([COMPILER, 'test', path, '--sysroot', ROOT,
                              '-o', self.directory / 'child'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('child failed', result.stderr)

    def test_modifier_arguments_are_checked(self):
        for source, diagnostic in [
            ('test with timeout:\n    pass', 'timeout requires 1 arguments'),
            ('test with repeat(1, 2):\n    pass', 'repeat requires 1 arguments'),
            ('test with parallel(2):\n    pass', 'parallel requires 0 arguments'),
            ('test with repeat(1) and repeat(2):\n    pass', 'duplicate test mode'),
        ]:
            path = self.write(source)
            result = self.invoke([COMPILER, 'test', path, '--emit', 'mlir', '--sysroot', ROOT])
            self.assertGreater(result.returncode, 0)
            self.assertIn(diagnostic, result.stderr)


if __name__ == '__main__':
    unittest.main()
