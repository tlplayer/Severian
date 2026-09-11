#!/usr/bin/env python3
"""Excluding a call name preserves primitive types and private prelude users."""
import unittest
from migration import COMPILER, ROOT, MigrationCase

class IntrinsicExclusions(MigrationCase):
    def policy(self, names):
        import json
        self.write('[package]\nname = "exclusions"\n[prelude]\nexclude = ' + json.dumps(names), name='package.toml')

    def test_replacements_use_ordinary_lookup_and_result_types(self):
        names = ['int', 'float', 'bool', 'char', 'string', 'type', 'len', 'bytes', 'list', 'tuple', 'borrow', 'clone']
        self.policy(names)
        declarations = '\n'.join(f'def {name}(value: int) -> int:\n    return value + 1\n' for name in names)
        assertions = '\n'.join(f'    assert({name}(41) == 42)' for name in names)
        self.native(declarations + '\ntest:\n' + assertions)

    def test_excluded_assert_is_an_ordinary_statement_call(self):
        self.policy(['assert'])
        self.native('''
            def assert():
                print("replacement")
            test:
                assert()
        ''', expected='replacement\n')

    def test_missing_replacement_does_not_reach_intrinsic(self):
        for name in ['int', 'string', 'type', 'len', 'bytes', 'list', 'tuple', 'borrow', 'clone', 'assert']:
            with self.subTest(name=name):
                self.policy([name])
                self.rejects(f'def main():\n    {name}(42)\n', 'unknown callable ' + name)

    def test_owned_strings_and_prefix_ownership_keep_their_semantics(self):
        self.policy(['string', 'clone', 'borrow', 'len', 'int'])
        self.native('''
            def clone(value: int) -> int:
                return value
            def string(value: int) -> bool:
                return value == 42
            test:
                original = "source"
                copied = clone original
                assert(copied == original)
                assert(string(clone(42)))
                print(copied)
        ''', expected='source\n')

if __name__ == '__main__':
    unittest.main()
