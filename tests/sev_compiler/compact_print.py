#!/usr/bin/env python3
"""Printing and selected imports must not materialize unrelated runtime bodies."""
import os
import unittest
from unittest.mock import patch

from migration import COMPILER, ROOT, MigrationCase


class CompactPrint(MigrationCase):
    def setUp(self):
        super().setUp()
        # Exercise checkout providers, not an older published prelude archive.
        self.environment = patch.dict(os.environ, {
            'SEVERIAN_HOME': str(self.directory / 'home'),
            'SEVERIAN_REGISTRY': str(self.directory / 'registry'),
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def build_ir(self, text):
        self.subject = self.write(text)
        return self.succeeds([COMPILER, 'build', self.subject, '--emit', 'mlir', '--sysroot', ROOT])

    def execute(self, expected):
        executable = self.directory / 'subject'
        self.succeeds([COMPILER, 'build', self.subject, '--sysroot', ROOT, '-o', executable])
        self.assertEqual(self.succeeds([executable]), expected)

    def test_empty_program_has_no_prelude_runtime(self):
        mlir = self.build_ir('')
        self.assertNotIn('llvm.func', mlir)
        self.assertNotIn('__sev_initializer', mlir)
        self.assertNotIn('2.718281828459045', mlir)

    def test_integer_print_is_one_mlir_operation(self):
        mlir = self.build_ir('def main():\n    print(1 + 1)\n')
        self.assertEqual(mlir.count('"vector.print"'), 1)
        self.assertNotIn('llvm.func', mlir)
        self.assertNotIn('memref.', mlir)
        self.assertNotIn('__sev_initializer', mlir)
        self.assertLess(len(mlir.splitlines()), 40)
        self.execute('2\n')

    def test_literal_output_needs_no_formatting_or_allocations(self):
        mlir = self.build_ir('def main():\n    print("hello λ")\n')
        self.assertNotIn('"memref.alloc"', mlir)
        self.assertNotIn('"arith.div', mlir)
        self.assertLess(len(mlir.splitlines()), 350)
        self.execute('hello λ\n')

    def test_mixed_output_and_options_preserve_behavior(self):
        self.build_ir('''
            def main():
                print()
                print("count", 42, true, 'λ', 0.5, None)
                print(-9223372036854775808, 9223372036854775807)
                print("a", "b", sep="|", end="!")
                print(7, 8, sep="/", end="done", flush=true)
                print()
        ''')
        self.execute('\ncount 42 true λ 0.5 None\n-9223372036854775808 9223372036854775807\na|b!7/8done\n')

    def test_selected_import_retains_callees_but_not_unrelated_functions(self):
        self.write('''
            @c(symbol="unused_external")
            def external() -> int
            def helper(value: int) -> int:
                return value + 1
            def answer() -> int:
                return helper(41)
            def unused() -> int:
                return external()
        ''', name='provider.sev')
        mlir = self.build_ir('''
            import answer from "provider.sev" as selected
            def main():
                print(selected())
        ''')
        self.assertNotIn('unused_external', mlir)
        self.execute('42\n')
        self.rejects('import answer from "provider.sev"\ndef main():\n    print(helper(1))\n', 'helper')

    def test_selected_math_keeps_its_dependencies(self):
        self.build_ir('def main():\n    print(round(1.5))\n')
        self.execute('2\n')


if __name__ == '__main__':
    unittest.main(verbosity=2)
