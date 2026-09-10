#!/usr/bin/env python3
"""Case tables and Cartesian products produce independent executable tests."""
import unittest

from migration import MigrationCase, ROOT


class TestCases(MigrationCase):
    def test_documented_cases_and_expect(self):
        for path in [
            '03-testing/01-basics/07-parameterized.sev',
            '03-testing/02-with-tests/10-parameterized.sev',
            '03-testing/01-basics/06-approximate.sev',
        ]:
            with self.subTest(path=path):
                self.native((ROOT / 'docs/examples' / path).read_text())

    def test_rows_are_ordered_and_bindings_are_isolated(self):
        self.native('''
            test sample(value, expected) with cases {
                (1, 2),
                (4, 5),
            }:
                local := value + 1
                print(local)
                expect(local == expected)
        ''', expected='2\n5\n')

    def test_cartesian_product_visits_every_pair(self):
        self.native('''
            test [a in [1, 2], b in [3, 4, 5]] with cases:
                print(a, b)
        ''', expected='1 3\n1 4\n1 5\n2 3\n2 4\n2 5\n')

    def test_invalid_tables_have_diagnostics(self):
        for text, diagnostic in [
            ('test sample(a, b) with cases {(1,)}:\n    pass', 'parameter count'),
            ('test sample(a, a) with cases {(1, 2)}:\n    pass', 'duplicate case parameter'),
            ('test sample(a) with cases {}:\n    pass', 'at least one case'),
            ('test [a in []] with cases:\n    pass', 'at least one case'),
        ]:
            with self.subTest(source=text):
                self.rejects(text, diagnostic)

    def test_expect_and_later_case_failures_are_not_suppressed(self):
        path = self.write('''
            test sample(value) with cases {(1,), (2,)}:
                expect(value == 1)
        ''')
        from migration import COMPILER
        result = self.invoke([COMPILER, 'test', path, '--sysroot', ROOT,
                              '-o', self.directory / 'cases'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('assert', result.stderr.lower())

    def test_membership_uses_element_and_container_types(self):
        self.native('''
            test:
                assert(2 in [1, 2, 3])
                assert(4 not in [1, 2, 3])
                empty: list[int] = []
                assert(2 not in empty)
                assert(0.5 in [0.25, 0.5])
                assert(true in [false, true])
                assert('λ' in ['a', 'λ'])
                assert("λ" in "aλz")
        ''')

    def test_membership_evaluates_each_operand_once_in_order(self):
        self.native('''
            def element() -> int:
                print("element")
                return 2
            def container() -> list[int]:
                print("container")
                return [1, 2, 3]
            test:
                assert(element() in container())
        ''', expected='element\ncontainer\n')

    def test_assertion_diagnostic_survives_redirected_output(self):
        from migration import COMPILER
        path = self.write('''
            test:
                assert(false, "specific failure")
        ''')
        result = self.invoke([COMPILER, 'test', path, '--sysroot', ROOT,
                              '-o', self.directory / 'diagnostic'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('assertion failed: specific failure', result.stderr)

    def test_assertion_message_is_lazy_and_condition_runs_once(self):
        self.native('''
            def condition() -> bool:
                print("condition")
                return true
            def message() -> string:
                print("message")
                return "unexpected"
            test:
                assert(condition(), message())
        ''', expected='condition\n')


if __name__ == '__main__':
    unittest.main()
