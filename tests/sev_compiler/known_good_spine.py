#!/usr/bin/env python3
"""Preserve the documentation baseline using the built source compiler.

Build first: package.pkg/debug/sev build sev_compiler
Run: python3 tests/sev_compiler/known_good_spine.py

Subjects are compiled in place so relative imports keep their real paths.
Hello is checked through its native stdout; its integration test mode is
outside this baseline. All other subjects also run their source tests.
"""
import unittest

from migration import COMPILER, ROOT, MigrationCase, tool


class KnownGoodSpine(MigrationCase):
    def check_example(self, relative, expected="", *, tests=True):
        subject = ROOT / "docs/examples" / relative
        for mode in ("build", "test") if tests else ("build",):
            with self.subTest(example=relative, mode=mode):
                arguments = [COMPILER, mode, subject, "--sysroot", ROOT]
                emitted = self.directory / f"{mode}.mlir"
                emitted.write_text(self.succeeds(arguments + ["--emit", "mlir"]))
                self.succeeds([tool("SEVERIAN_MLIR_OPT", "mlir-opt-21"),
                               "--verify-each", emitted, "-o", self.directory / "verified.mlir"])
                executable = self.directory / mode
                output = self.succeeds(arguments + ["-o", executable])
                if mode == "test":
                    self.assertEqual(output, "")
                # `test` executes the generated test entry itself and must not
                # call source main. Run its artifact too to check its output.
                self.assertEqual(self.succeeds([executable]), expected if mode == "build" else "")

    def test_hello(self):
        self.check_example("00-getting-started/01-hello.sev", "hello, severian\n", tests=False)

    def test_conversion(self):
        self.check_example("01-types/01-basic/03-conversion.sev",
                           "10\n0.5\n10.5\ntrue\nseverian\n10.5!\n")

    def test_conditional_expression(self):
        self.check_example("02-functions/02-control-flow/07-conditional-expression.sev")

    def test_ordinary_and_named(self):
        self.check_example("03-testing/01-basics/01-ordinary-and-named.sev")

    def test_building_library(self):
        self.check_example("05-building/src/lib.sev")

    def test_building_binary(self):
        self.check_example("05-building/src/main.sev", "42\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
