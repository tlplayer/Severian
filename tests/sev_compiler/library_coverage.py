#!/usr/bin/env python3
"""Native coverage for source prelude providers and collection families."""
import shutil
import os
import unittest

import known_good_spine
from migration import MigrationCase, ROOT


class LibraryCoverage(MigrationCase):
    check_example = known_good_spine.KnownGoodSpine.check_example

    def test_approximate_example(self):
        self.check_example("03-testing/02-with-tests/09-approximate.sev")

    def test_capture_preserves_initializer_and_isolates_cases(self):
        capture = os.path.relpath(ROOT / "library/testing/src/capture.sev", self.directory)
        self.native(f'''
            import "{capture}" as capture
            @c(symbol="perror")
            def emit_error(prefix: native_ptr)
            test:
                initial = capture.begin()
                print("initial λ")
                context = capture.ready(initial)
                print("first")
                first = capture.stdout(context)
                second_context = capture.reset(context)
                print("second")
                emit_error(capture.__pointer(0))
                second = capture.stdout(second_context)
                errors = capture.stderr(second_context)
                capture.finish(second_context)
                assert(first == "initial λ\\nfirst\\n")
                assert(second == "initial λ\\nsecond\\n")
                assert(errors != "")
                print("restored")
        ''', expected="restored\n")

    def test_snapshot_example(self):
        self.check_example("03-testing/02-with-tests/19-snapshots.sev")

    def test_approximate_special_values(self):
        self.native('''
            test:
                assert(approximate(0.1 + 0.2, 0.3))
                assert(not approximate(1.0, 2.0))
                assert(approximate(0.000001, 0.0, rtol=0.0, atol=0.00001))
                infinity = 1.0 / 0.0
                nan = 0.0 / 0.0
                assert(approximate(infinity, infinity))
                assert(not approximate(1.0, infinity))
                assert(not approximate(infinity, -infinity))
                assert(not approximate(nan, nan))
        ''')

    def test_list_family_slice_and_copy(self):
        self.native('''
            test:
                values = [1, 2, 3, 4]
                assert(values[::-1] == [4, 3, 2, 1])
                assert(values[::-1][::-1] == values)
                assert(values[-3:-1] == [2, 3])
                assert(values[::2] == [1, 3])
                assert(values[100:-100:-1] == [4, 3, 2, 1])
                assert(values[::9223372036854775807] == [1])
                assert(values[::-9223372036854775808] == [4])
                duplicate = values[:]
                duplicate[0] = 9
                assert(values[0] == 1)
                assert([1.5, 2.5][::-1] == [2.5, 1.5])
                assert([true, false][::-1] == [false, true])
                assert(['λ', '😀'][::-1] == ['😀', 'λ'])
        ''')

    def test_source_declared_macro_family(self):
        self.native('''
            union Samples:
                i16
                f64
            -> choices[T: Samples]():
                def identity(value: T) -> T:
                    return value
                test "family member":
                    assert(identity(T(7)) == T(7))
                    print(type(identity(T(7))))
            choices[Samples]()
        ''', expected="i16\nfloat\n")

    def test_prelude_import_adds_library_without_rebuilding(self):
        root = self.directory / "sysroot"
        (root / "sev_compiler").mkdir(parents=True)
        shutil.copytree(ROOT / "sev_compiler/universal", root / "sev_compiler/universal")
        (root / "library").symlink_to(ROOT / "library", target_is_directory=True)
        helper = root / "helper.sev"
        helper.write_text("def library_answer() -> int:\n    return 42\n")
        prelude = root / "sev_compiler/universal/prelude.sev"
        prelude.write_text(prelude.read_text() + '\nimport "../../helper.sev"\n')
        self.native("test:\n    assert(library_answer() == 42)\n", sysroot=root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
