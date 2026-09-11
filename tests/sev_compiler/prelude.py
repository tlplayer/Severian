#!/usr/bin/env python3
"""The selected prelude reserves names while retaining typed extension points."""
import unittest

from migration import COMPILER, ROOT, MigrationCase


class PreludePolicy(MigrationCase):
    def test_reserved_free_functions_fail_in_check_build_and_test(self):
        for text in (
            'def max() -> int:\n    return 42\n',
            'def max[T](value: T) -> T:\n    return value\n',
            'def view(value: borrow string) -> borrow string:\n    return value\n',
            'def len(value: string) -> int:\n    return 42\n',
            'def __buffer_load[T](value: T) -> T:\n    return value\n',
        ):
            for command in ('check', 'build', 'test'):
                with self.subTest(text=text, command=command):
                    result = self.invoke([COMPILER, command, self.write(text),
                                          '--emit', 'mlir', '--sysroot', ROOT])
                    self.assertGreater(result.returncode, 0)
                    self.assertRegex(result.stderr, r'error: E[0-9]+:.*reserved prelude function')

    def test_polymorphism_and_member_methods_preserve_prelude_behavior(self):
        self.native('''
            class Counted:
                count: int
                def max() -> int:
                    return count
            def len(value: Counted) -> int:
                return value.count
            operator max(left: Counted, right: Counted) -> Counted:
                return left if left.count > right.count else right
            test:
                assert(len(Counted(42)) == 42)
                assert(len("text") == 4)
                assert(max(Counted(7), Counted(9)).max() == 9)
        ''')
        self.rejects('''
            def len(value: int) -> string:
                return "wrong result contract"
        ''', 'not a distinct polymorphic overload')

    def test_package_exclusion_removes_reservation_and_provider(self):
        self.write('''
            [package]
            name = "custom-prelude"
            version = "0.1.0"
            [[bin]]
            name = "custom-prelude"
            path = "main.sev"
            [prelude]
            exclude = ["max", "round"]
        ''', name='package.toml')
        self.rejects('test:\n    assert(round(1.5) == 2.0)\n', 'round')
        self.native('''
            def max() -> int:
                return 42
            def round(value: float, digits: int = 0) -> float:
                return value
            test:
                assert(max() == 42)
                assert(round(1.5) == 1.5)
        ''')

        self.write('''
            def max() -> int:
                return 42
            def main():
                print(max())
        ''', name='main.sev')
        result = self.invoke([COMPILER, 'run', self.directory, '--sysroot', ROOT])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '42\n')
        manifest = self.directory / 'package.toml'
        manifest.write_text(manifest.read_text().replace('["max", "round"]', '[]'))
        result = self.invoke([COMPILER, 'check', self.directory, '--sysroot', ROOT])
        self.assertGreater(result.returncode, 0)
        self.assertIn('reserved prelude function max', result.stderr)

    def test_exclusions_are_validated(self):
        for names, diagnostic in (
            ('["missing"]', 'unknown prelude exclusion'),
            ('["max", "max"]', 'duplicate prelude exclusion'),
            ('["string"]', 'requires extracting compiler intrinsic'),
        ):
            with self.subTest(names=names):
                self.write('[package]\nname = "policy"\n[prelude]\nexclude = ' + names,
                           name='package.toml')
                self.rejects('def main():\n    pass\n', diagnostic)

    def test_qualified_package_function_has_a_distinct_identity(self):
        self.write('''
            def max() -> int:
                return 42
        ''', name='provider.sev')
        self.native('''
            import * from "provider.sev" as provider
            test:
                assert(provider.max() == 42)
        ''')


if __name__ == '__main__':
    unittest.main()
