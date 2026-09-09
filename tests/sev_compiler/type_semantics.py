#!/usr/bin/env python3
"""Regressions for the shared semantics used by docs/examples/01-types."""
import unittest

from migration import MigrationCase, ROOT


class TypeSemantics(MigrationCase):
    def test_unsigned_widths_and_full_u64_range(self):
        self.native('''
            test:
                small: u16 = 65535
                word: u32 = 4294967295
                large: u64 = 18446744073709551615
                assert(small > 32767)
                assert(word > 2147483647)
                assert(large > 9223372036854775807)
                assert(string(large) == "18446744073709551615")
                assert(large / 10 == 1844674407370955161)
                assert(large % 10 == 5)
                assert(u32(small) == 65535)
                assert(i64(word) == 4294967295)
                assert(u64(i64(42)) == 42)
        ''')
        self.rejects('test:\n    value: u64 = 18446744073709551616\n', "outside u64")
        self.rejects('test:\n    value: u64 = -1\n', "outside u64")

    def test_unsigned_generic_examples(self):
        for name in ("06-type-generic", "22-kind-generic", "25-compiler-term-generic"):
            with self.subTest(example=name):
                path = ROOT / f"docs/examples/01-types/04-generics/{name}.sev"
                self.native(path.read_text())

    def test_power(self):
        self.native('''
            test:
                assert(2 ** 10 == 1024)
                assert(2.0 ** 0.5 > 1.414)
                assert(9.0 ** 0.5 == 3.0)
                assert(2 ** 0 == 1)
        ''')

    def test_constructor_initializes_fields(self):
        path = ROOT / "docs/examples/01-types/02-classes/01-inferred-field-mutation.sev"
        self.native(path.read_text())

    def test_constructor_overloads(self):
        path = ROOT / "docs/examples/01-types/02-classes/05-overloaded-constructors.sev"
        self.native(path.read_text() + '''
test:
    assert(X(20, 22).value == 42)
    assert(X(17).value == 17)
''')

    def test_field_visibility_and_compiler_test_setup(self):
        path = ROOT / "docs/examples/01-types/02-classes/02-read-write-fields.sev"
        self.native(path.read_text())

    def test_constructor_checks_field_types_and_initialization(self):
        self.rejects('''
            class Wrong:
                value: int
                def Wrong():
                    value := true
            test:
                value = Wrong()
        ''', "boolean cannot initialize an integer")
        self.rejects('''
            class Missing:
                value: int
                def Missing():
                    pass
            test:
                value = Missing()
        ''', "unknown name value")

    def test_expression_generic_name_is_not_a_compiler_capability(self):
        path = ROOT / "docs/examples/01-types/04-generics/09-expression-generic.sev"
        self.native(path.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
