#!/usr/bin/env python3
"""Native regressions for supported documentation examples and their boundaries."""
import unittest

import known_good_spine
from migration import COMPILER, ROOT


class ExampleProgress(known_good_spine.KnownGoodSpine):
    def test_strings_example(self):
        self.check_example("01-types/01-basic/04-strings.sev",
                           "Severian\n8\nSEVERIAN\ntrue\nHello, Severian!\nSever\n", tests=False)

    def test_identity_example(self):
        self.check_example("01-types/01-basic/06-is.sev")

    def test_string_slice_boundaries(self):
        self.native('''
            test:
                value = "aλ😀z"
                assert(value[:] == value)
                assert(value[:2] == "aλ")
                assert(value[-2:] == "😀z")
                assert(value[::-1] == "z😀λa")
                assert(value[::2] == "a😀")
                assert(value[3:0:-2] == "zλ")
                assert(value[:-1:-1] == "")
                assert(value[-100:100] == value)
                assert(value[100:-100:-1] == "z😀λa")
                assert(""[::-1] == "")
                assert(value[::9223372036854775807] == "a")
                assert(value[::-9223372036854775808] == "z")
                assert(value.contains("λ😀"))
                assert(value.contains(""))
                assert(not value.contains("😀λ"))
                assert("λ😀" in value)
                assert("😀λ" not in value)
                assert("Mixed 123!".upper() == "MIXED 123!")
        ''')

    def test_slice_operands_execute_once_in_order(self):
        self.native('''
            def subject() -> string:
                print("value")
                return "abcdef"
            def bound(label: string, value: int) -> int:
                print(label)
                return value
            test:
                assert(subject()[bound("start", 1):bound("stop", 5):bound("step", 2)] == "bd")
        ''', expected="value\nstart\nstop\nstep\n")

    def test_slice_rejects_invalid_steps(self):
        self.rejects('print("abc"[::true])', r"boolean cannot initialize an integer")
        subject = self.write('print("abc"[::0])')
        executable = self.directory / "zero-step"
        self.succeeds([COMPILER, "build", subject, "--sysroot", ROOT, "-o", executable])
        self.assertNotEqual(self.invoke([executable]).returncode, 0)

    def test_copy_has_independent_storage(self):
        self.native('''
            test:
                original = [1, 2, 3]
                alias = original
                duplicate = copy original
                assert(alias is original)
                assert(!(duplicate is original))
                assert(duplicate == original)
                duplicate[0] = 9
                assert(original[0] == 1)
                assert(!(duplicate == original))
                assert(!false)
                assert(!(!true))
        ''')

    def test_while_initializer_example(self):
        self.check_example("02-functions/02-control-flow/01-while-initializer.sev",
                           "0\n1\n2\ntrue\n", tests=False)

    def test_if_while_for_example(self):
        self.check_example("02-functions/02-control-flow/02-if-while-for.sev",
                           "0\n1\n2\neven\nodd\neven\nodd\n", tests=False)

    def test_continue_break_example(self):
        self.check_example("02-functions/02-control-flow/03-continue-break.sev", "1\n", tests=False)

    def test_while_initializer_runs_once_and_is_scoped(self):
        self.native('''
            def initial() -> int:
                print("initial")
                return 0
            test:
                while count < 3 with count := initial():
                    count += 1
                    if count == 1:
                        continue
                    print(count)
        ''', expected="initial\n2\n3\n")
        self.rejects('''
            def main():
                while count < 1 with count := 0:
                    count += 1
                print(count)
        ''', r"unknown name count")


if __name__ == "__main__":
    unittest.main(verbosity=2)
