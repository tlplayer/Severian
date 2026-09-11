#!/usr/bin/env python3
"""Native scalar numerics and standalone standard-library import regressions."""
import unittest

from migration import MigrationCase


class Numerics(MigrationCase):
    def test_f32_rounding_storage_arithmetic_and_promotion(self):
        mlir = self.native('''
            def add(left: f32, right: f32) -> f32:
                return left + right
            test:
                rounded: f32 = 16777217.0
                assert(f64(rounded) == 16777216.0)
                assert(type(rounded) == "f32")
                left: f32 = 1.5
                right: f32 = 2.5
                assert(add(left, right) == 4.0)
                assert(left * right == 3.75)
                assert(-left == -1.5)
                wide: f64 = 0.25
                assert(left + wide == 1.75)
                assert(type(left + wide) == "float")
                assert(int(right) == 2)
                assert(f32(3) == 3.0)
                values: list[f32] = [left, right]
                values.append(f32(4.0))
                assert(values[2] == 4.0)
                print(left, right, f32(0.5))
        ''', expected='1.5 2.5 0.5\n')
        self.assertIn('f32', mlir)
        self.assertIn('arith.extf', mlir)
        self.assertIn('arith.truncf', mlir)

    def test_float_conversion_policies(self):
        self.native('''
            test:
                narrow: f32 = 1.25
                assert(f64(narrow, promote) == 1.25)
                assert(f32(1.25, lossy) == narrow)
                reject:
                    value = f32(1.25, promote)
                reject:
                    value: f32 = true
        ''')
        self.rejects('''
            def narrow(value: f64) -> f32:
                return value + 1.0
        ''', 'expected scalar type')

    def test_standard_library_alias_and_math_boundaries(self):
        self.native('''
            import math as mathematics
            test:
                assert(mathematics.sin(0.0) == 0.0)
                assert(mathematics.log2(8.0) == 3.0)
                assert(mathematics.floor(-1.25) == -2)
                assert(mathematics.ceil(-1.25) == -1)
                assert(mathematics.round(1.23456, 3) == 1.235)
                assert(mathematics.isnan(0.0 / 0.0))
                assert(not mathematics.isfinite(1.0 / 0.0))
                assert(mathematics.isfinite(-42.5))
        ''')

    def test_selective_import_spellings(self):
        for declaration in ['from math import sin as sine',
                            'import sin from math as sine']:
            with self.subTest(declaration=declaration):
                self.native(declaration + '\ntest:\n    assert(sine(0.0) == 0.0)\n')

    def test_unknown_standard_library_is_a_diagnostic(self):
        self.rejects('import missing_numerics_library', 'unknown standard library')

    def test_multiple_selected_imports_share_module_declarations(self):
        self.native('''
            from math import sin as sine
            from math import cos as cosine
            test:
                assert(sine(0.0) == 0.0)
                assert(cosine(0.0) == 1.0)
        ''')

    def test_unknown_imported_member_is_a_diagnostic(self):
        self.rejects('from math import missing_function', 'imported declaration')

    def test_read_only_recursion_ignores_other_modules_mutation(self):
        self.write('''
            def step(value: string, remaining: int) -> int:
                if remaining == 0:
                    assert(value == "hello")
                    return 0
                return step(value, remaining - 1)
        ''', 'reader.sev')
        self.write('''
            def step(value: list[int], remaining: int) -> int:
                value.append(remaining)
                return remaining
        ''', 'writer.sev')
        self.native('''
            import * from "reader.sev" as reader
            import * from "writer.sev" as writer
            test:
                message = "hel" + "lo"
                assert(reader.step(message, 3) == 0)
                assert(message == "hello")
                values: list[int] = []
                assert(writer.step(values, 7) == 7)
                assert(values == [7])
        ''')


if __name__ == '__main__':
    unittest.main()
