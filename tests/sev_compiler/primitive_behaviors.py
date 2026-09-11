#!/usr/bin/env python3
"""Native acceptance for primitive migration slices; no compiler rebuilds."""
import os
import shutil
import signal
import unittest

from migration import COMPILER, MigrationCase, ROOT
from resource_guard import run as guarded_run


class PrimitiveBehaviors(MigrationCase):
    def traps(self, expression, prefix=""):
        path = self.write(prefix + "\ndef main():\n    " + expression + "\n")
        executable = self.directory / "trap.exe"
        self.succeeds([COMPILER, "build", path, "--sysroot", ROOT, "-o", executable])
        result = self.invoke([executable])
        self.assertEqual(result.returncode, -signal.SIGABRT,
                         "expected the assertion boundary, not success or an arithmetic fault")

    def test_integer_division_and_remainder_all_active_widths(self):
        lines = ["test:"]
        for typename, minimum in (("i8", -128), ("i16", -32768),
                                  ("i32", -2147483648), ("i64", -9223372036854775808)):
            lines += [
                f"    assert({typename}(-7) / {typename}(3) == {typename}(-2))",
                f"    assert({typename}(-7) // {typename}(3) == {typename}(-3))",
                f"    assert({typename}(7) // {typename}(-3) == {typename}(-3))",
                f"    assert({typename}(-7) // {typename}(-3) == {typename}(2))",
                f"    assert({typename}(-7) % {typename}(3) == {typename}(-1))",
                f"    assert({typename}({minimum}) % {typename}(-1) == {typename}(0))",
                f"    assert({typename}({minimum}) / {typename}(1) == {typename}({minimum}))",
            ]
        lines += ["    assert(u8(255) / u8(2) == u8(127))",
                  "    assert(u8(255) // u8(2) == u8(127))",
                  "    assert(u8(255) % u8(2) == u8(1))"]
        self.native("\n".join(lines))

    def test_division_guards_precede_undefined_operations(self):
        for typename, minimum in (("i8", -128), ("i16", -32768),
                                  ("i32", -2147483648), ("i64", -9223372036854775808)):
            for operator in ("/", "//"):
                with self.subTest(type=typename, operator=operator, case="overflow"):
                    self.traps(f"value = {typename}({minimum}) {operator} {typename}(-1)")
        for typename in ("i8", "i16", "i32", "i64", "u8"):
            for operator in ("/", "//", "%"):
                with self.subTest(type=typename, operator=operator, case="zero"):
                    self.traps(f"value = {typename}(7) {operator} {typename}(0)")

    def test_source_operator_edit_changes_native_behavior(self):
        sysroot = self.directory / "sysroot"
        for name in ("sev_compiler", "library"):
            shutil.copytree(ROOT / name, sysroot / name, ignore=shutil.ignore_patterns("package.pkg"))
        path = sysroot / "sev_compiler/universal/primitive/numeric/operators.sev"
        program = "test:\n    assert(i32(7) / i32(3) == i32(2))\n"
        self.native(program, sysroot=sysroot)
        original = path.read_text()
        self.assertEqual(original.count("return __signed_divide(self, right)"), 1)
        path.write_text(original.replace("return __signed_divide(self, right)", "return self"))
        self.native(program.replace("== i32(2)", "== i32(7)"), sysroot=sysroot)
        # MigrationCase verifies the compiler digest remained unchanged.

    def test_direct_prelude_subjects_keep_their_namespace(self):
        for name in ("numeric/operators.sev", "numeric/conversion.sev", "collections.sev"):
            subject = ROOT / "sev_compiler/universal/primitive" / name
            for spelling in (subject, subject.relative_to(ROOT)):
                with self.subTest(file=name, spelling=str(spelling)):
                    result = self.invoke([COMPILER, "check", spelling, "--sysroot", ROOT], cwd=ROOT)
                    self.assertEqual(result.returncode, 0, result.stderr)
        subject = ROOT / "sev_compiler/universal/primitive/numeric/operators.sev"
        result = guarded_run([str(COMPILER), "check", os.path.relpath(subject, self.directory),
                                 "--sysroot", str(ROOT)], cwd=self.directory,
                                env={**os.environ, "PWD": "/unrelated"},
                                timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_documentation_and_character_literals(self):
        self.write("'''\nclass Fictional:\n    value: Missing\n'''\ndef answer() -> int:\n    return 42\n", "documented.sev")
        self.native("import * from \"documented.sev\"\ntest:\n    assert(answer() == 42)\n    assert('λ' == 'λ')\n    assert('''\n        λ\n        text\n    ''' == \"λ\\ntext\\n\")\n")
        self.rejects("value = '''unfinished\n", "unterminated block string")
        self.rejects("value = 'ab'\n", "character literal requires one Unicode scalar")

    def test_ordinary_declarations_do_not_consume_specialization_budget(self):
        functions = "\n".join(f"def ordinary_{i}() -> int:\n    return {i}\n" for i in range(600))
        self.native(functions + '\ntest:\n    print(ordinary_599())\n', expected="599\n")

    def test_invalid_utf8_sequences_fail_in_shared_decoder(self):
        subject = ROOT / "sev_compiler/universal/primitive/char/encoding.sev"
        # Source locators are relative to the fixture, including in a temp dir.
        prefix = f'import * from "{os.path.relpath(subject, self.directory)}" as encoding\n'
        for expression in ("decode_two(192, 128)", "decode_two(194, 127)",
                           "decode_three(224, 159, 191)", "decode_three(237, 160, 128)",
                           "decode_three(225, 128, 256)", "decode_four(240, 143, 191, 191)",
                           "decode_four(244, 144, 128, 128)", "decode_four(245, 128, 128, 128)",
                           "decode_four(240, 144, 128, -1)"):
            with self.subTest(sequence=expression):
                self.traps("value = encoding." + expression, prefix)

    def test_range_overflow_and_empty_directions(self):
        self.native('''
            test:
                count := 0
                for value in range(9223372036854775806, 9223372036854775807, 2):
                    assert(value == 9223372036854775806)
                    count += 1
                assert(count == 1)
                for value in range(-9223372036854775807, -9223372036854775808, -2):
                    assert(value == -9223372036854775807)
                    count += 1
                assert(count == 2)
                for value in range(0, -9223372036854775808, -9223372036854775808):
                    assert(value == 0)
                    count += 1
                assert(count == 3)
                for value in range(3, 0):
                    assert(false)
                for value in range(0, 3, -1):
                    assert(false)
        ''')
        self.traps("value = range(0, 3, 0)")


if __name__ == "__main__":
    unittest.main()
