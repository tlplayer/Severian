#!/usr/bin/env python3
"""Conversions dispatch over every member of a union and evaluate once."""
import unittest

from migration import COMPILER, MigrationCase, ROOT


class UnionConversions(MigrationCase):
    def test_union_parameter_example(self):
        path = ROOT / 'docs/examples/02-functions/01-basic/03-union-parameters.sev'
        self.native(path.read_text())

    def test_union_conversion_evaluates_its_subject_once(self):
        self.native('''
            def value(which: bool) -> int | float:
                print("subject")
                if which:
                    return 42
                return 2.5
            test:
                assert(float(value(true)) == 42.0)
                assert(float(value(false)) == 2.5)
        ''', expected='subject\nsubject\n')

    def test_string_conversion_accepts_whitespace_and_exponents(self):
        self.native('''
            test:
                assert(float("  -1.25e2 ") == -125.0)
                assert(f64("4.5") == 4.5)
                assert(float("0") == 0.0)
        ''')

    def test_invalid_string_conversion_fails(self):
        for value in ['', ' ', '4.5junk', 'abc', '4.5λ']:
            with self.subTest(value=value):
                path = self.write('test:\n    float("' + value + '")')
                result = self.invoke([COMPILER, 'test', path, '--sysroot', ROOT,
                                      '-o', self.directory / 'invalid'])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid floating-point string', result.stderr)

    def test_all_union_members_must_be_convertible(self):
        self.rejects('''
            def convert(value: int | bool) -> float:
                return float(value)
        ''', 'conversion')


if __name__ == '__main__':
    unittest.main()
