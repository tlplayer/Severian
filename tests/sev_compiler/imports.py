#!/usr/bin/env python3
"""Explicit file import syntax through the source compiler and native backend."""
import unittest

from migration import MigrationCase


class ExplicitImports(MigrationCase):
    def setUp(self):
        super().setUp()
        self.write('def answer() -> int:\n    return 42\n', 'array.sev')

    def test_wildcard_reexports(self):
        self.write('import * from "array.sev"\n', 'facade.sev')
        self.native('''
            import * from "facade.sev"
            test:
                assert(answer() == 42)
        ''')

    def test_qualified_wildcard(self):
        self.native('''
            import * from "array.sev" as arrays
            def answer() -> int:
                return 7
            test:
                assert(arrays.answer() == 42)
                assert(answer() == 7)
        ''')

    def test_implicit_import_requires_migration(self):
        for suffix in ('', ' as arrays'):
            with self.subTest(suffix=suffix):
                self.rejects('import "array.sev"' + suffix + '\n',
                             'file imports require explicit')

    def test_malformed_wildcard(self):
        for declaration, message in [
            ('import * "array.sev"', 'expected `from`'),
            ('import * from', 'expected a locator string'),
            ('import * from array', 'expected a locator string'),
        ]:
            with self.subTest(declaration=declaration):
                self.rejects(declaration + '\n', message)


if __name__ == '__main__':
    unittest.main()
