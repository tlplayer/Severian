#!/usr/bin/env python3
"""Compiler regressions found by running the basic type and class examples."""
import unittest

from migration import COMPILER, MigrationCase, ROOT


class ExampleFailures(MigrationCase):
    def run_tests(self, text, name='subject.sev'):
        path = self.write(text, name)
        result = self.invoke([COMPILER, 'test', path, '--sysroot', ROOT,
                              '-o', self.directory / (path.stem + '.exe')])
        return path, result

    def test_prelude_policies_and_reports_preserve_captured_streams(self):
        for mode in ['integ', 'integration']:
            with self.subTest(mode=mode):
                path, result = self.run_tests(f'''
                    test with {mode} "captured output":
                        print("captured")
                        assert(stdout == "captured\\n")
                        assert(stderr == "")
                    test with {mode} "fresh capture":
                        assert(stdout == "")
                        assert(stderr == "")
                    test with profile "measurement":
                        assert(not (time < 0ns))
                ''', name=mode + '.sev')
                self.assertEqual(result.returncode, 0, result.stderr)
                for label in [f'captured output [{mode}]', f'fresh capture [{mode}]',
                              'measurement [profile]']:
                    self.assertIn('  RUN ' + label, result.stderr)
                    self.assertIn('  ok ' + label, result.stderr)
                self.assertIn(f'{path}:1', result.stderr)
                self.assertIn('Tests: 3 passed', result.stderr)

    def test_overloaded_results_drive_numeric_inference_and_evaluate_once(self):
        _, result = self.run_tests('''
            def duration() -> Duration:
                print("duration")
                return 2s
            test "dimensionless results":
                assert(duration() / 1ms == 2000)
                assert(4096B / 1KiB == 4)
                assert(2000 == 2s / 1ms)
                assert(2s / 1ms + 1 == 2001)
                assert(100MB / 2s == 50MB / 1s)
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'duration\n')

    def test_chain_layout_ends_before_sibling_statements_and_declarations(self):
        _, result = self.run_tests('''
            class Counter:
                value: int
                def next() -> Counter:
                    return Counter(value + 1)
            def make() -> Counter:
                value = Counter(0)
                    .next()
                    .next()

                return value

            test "chains":
                value = make()
                    .next()
                    .next()

                assert(value.value == 4)

            test "sibling":
                assert(make().value == 2)
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Tests: 2 passed', result.stderr)

    def test_compiler_cases_keep_delimited_maps_inside_case(self):
        _, result = self.run_tests('''
            class Point:
                x: int
                y: int
            test with compiler "group updates":
                accept:
                    point := Point(1, 2)
                    point.set({
                    "x": 10,
                    "y": 20,
                })
                reject:
                    point := Point(1, 2)
                    point.set({
                    "x": 10,
                    "z": 20,
                })
            test "following case":
                point := Point(1, 2)
                point.set({
                    "x": 3,
                    "y": 4,
                })
                assert(point.x == 3)
                assert(point.y == 4)
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('group updates [compiler]', result.stderr)
        self.assertIn('Tests: 2 passed', result.stderr)

    def test_reports_name_location_early_return_and_failed_case(self):
        path, result = self.run_tests('''
            test "":
                return
            test "failed case":
                assert(false, "intentional failure")
            test "never reached":
                pass
        ''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f'  ok unnamed test ({path}:1)', result.stderr)
        self.assertIn(f'  RUN failed case ({path}:3)', result.stderr)
        self.assertNotIn('  ok failed case', result.stderr)
        self.assertNotIn('never reached', result.stderr)
        self.assertNotIn('Tests: 3 passed', result.stderr)

    def test_missing_policy_diagnostic_has_source_and_remedy(self):
        path, result = self.run_tests('''
            test with missing "missing policy":
                pass
        ''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('E000215', result.stderr)
        self.assertIn(f'{path}:1:1', result.stderr)
        self.assertIn('TestMode', result.stderr)

    def test_operator_diagnostic_includes_operand_types(self):
        _, result = self.run_tests('''
            test "wrong operands":
                value: int = 1
                assert(value == "one")
        ''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('operator == (i64, string)', result.stderr)

    def test_unclosed_expression_does_not_consume_next_compiler_case(self):
        _, result = self.run_tests('''
            test with compiler "syntax rejection":
                reject:
                    value = (1 +
                accept:
                    value = 2
            test "following runtime case":
                assert(true)
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Tests: 2 passed', result.stderr)

    def test_zero_argument_mock_covers_all_calls_and_stays_local(self):
        _, result = self.run_tests('''
            def enabled() -> bool:
                return false
            def indirect() -> bool:
                return enabled()
            test "mocked":
                mock(
                    enabled() -> true
                    else throw Error("unexpected call")
                )
                assert(enabled())
                assert(indirect())
            test "original":
                assert(not enabled())
                assert(not indirect())
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_file_without_test_declarations_reports_zero(self):
        _, result = self.run_tests('''
            def main():
                print("application")
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Tests: 0 (no test declarations)', result.stderr)

    def test_throwing_fallback_preserves_owned_string_layout(self):
        _, result = self.run_tests('''
            def name(found: bool) -> string | None:
                if found:
                    return "present"
                return None
            test "throwing string fallback":
                value = name(true) else throw Error("missing")
                assert(value == "present")
                throws(name(false) else throw Error("missing"))
                print("continued")
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'continued\n')

    def test_integration_assertion_restores_streams_before_diagnostic(self):
        path, result = self.run_tests('''
            test with integ "captured failure":
                print("actual")
                if true:
                    assert("expected" in stdout)
        ''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('assertion failed: "expected" in stdout', result.stderr)
        self.assertIn(f'{path}:4:16', result.stderr)
        self.assertNotIn('  ok captured failure', result.stderr)

    def test_compiler_expectation_diagnostic_names_case_and_original_error(self):
        _, result = self.run_tests('''
            test with compiler "accepted program":
                accept:
                    value = missing_value
        ''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('compiler test expectation failed in accepted program', result.stderr)
        self.assertIn('expected acceptance', result.stderr)
        self.assertIn('unknown name missing_value', result.stderr)

    def test_case_tables_and_runner_modifiers_report_each_logical_case(self):
        _, result = self.run_tests('''
            test sample(value) with cases {(1,), (2,)}:
                assert(value > 0)
            test with repeat(2) and timeout(1s) "repeated":
                print("iteration")
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'iteration\niteration\n')
        self.assertIn('  ok sample [0]', result.stderr)
        self.assertIn('  ok sample [1]', result.stderr)
        self.assertIn('  ok repeated [repeat, timeout]', result.stderr)
        self.assertIn('Tests: 3 passed', result.stderr)

    def test_mocks_and_indirect_copies_have_independent_native_signatures(self):
        _, result = self.run_tests('''
            def foo(value: int) -> int:
                return value
            def calculate(value: int) -> int:
                return foo(value) * 2
            test "first mock":
                mock(foo(0) -> 10, else throw Error("first"))
                assert(calculate(0) == 20)
                throws(foo(1) -> Error("first"))
            test "second mock":
                mock(foo(0) -> 30, else throw Error("second"))
                assert(calculate(0) == 60)
                throws(foo(1) -> Error("second"))
            test "original signature":
                assert(foo(3) == 3)
                assert(calculate(3) == 6)
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
