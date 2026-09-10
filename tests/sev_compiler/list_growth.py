#!/usr/bin/env python3
"""Source extension methods and buffer receiver borrowing through sev_compiler."""
import os
import re
import subprocess
import unittest

from bootstrap_mlir import tool
from migration import MigrationCase, ROOT


class ListGrowth(MigrationCase):
    def test_method_identity_comes_from_source(self):
        origin = ROOT / "sev_compiler/universal/prelude.sev"
        sysroot = self.directory / "sysroot"
        prelude = sysroot / "sev_compiler/universal/prelude.sev"
        prelude.parent.mkdir(parents=True)
        provider = prelude.parent / "collections.sev"
        collection_source = origin.parent / "primitive/collections.sev"
        body = collection_source.read_text().replace(
            '"slicing.sev"',
            f'"{os.path.relpath(collection_source.parent / "slicing.sev", provider.parent)}"')
        provider.write_text(body.replace("def append(", "def grow("))
        imports = []
        for line in origin.read_text().splitlines():
            match = re.fullmatch(r'import "([^"]+)"(.*)', line)
            if match:
                path = (origin.parent / match[1]).resolve()
                if path == collection_source:
                    path = provider
                imports.append(f'import "{os.path.relpath(path, prelude.parent)}"{match[2]}\n')
        prelude.write_text("".join(imports))
        program = '''
            def read() -> int:
                values: list[int] = []
                values.append(42)
                return values[0]
            test:
                print(read())
        '''
        self.rejects(program, r"unknown callable values.append", sysroot=sysroot)
        self.native(program.replace(".append(", ".grow("), sysroot=sysroot, expected="42\n")

    def test_unchanged_generic_examples(self):
        for number in (12, 16, 18, 24):
            path, = (ROOT / "docs/examples/01-types/05-generics").glob(f"{number:02d}-*.sev")
            with self.subTest(example=path.name):
                self.native(path.read_text(), name=path.name)

    def test_growth_lifetimes_and_layouts(self):
        self.native('''
            class Point:
                flag: bool
                x: int
                y: float
            class Empty:
                pass
            def points(count: int) -> list[Point]:
                values: list[Point] = []
                for index in range(count):
                    values.append(Point(true, index, float(index)))
                return values
            test:
                values := points(32)
                for index in range(len(values)):
                    assert(values[index].flag)
                    assert(values[index].x == index)
                    assert(values[index].y == float(index))
                retained = values
                values.append(values[0])
                assert(len(values) == 33)
                assert(values[32].x == 0)
                assert(len(retained) == 32)
                assert(retained[31].x == 31)
                duplicate = copy values
                duplicate.append(Point(false, 100, 1.5))
                assert(len(duplicate) == 34)
                assert(len(values) == 33)
                for count in range(12):
                    values = points(count)
                    values.append(Point(false, 99, 9.5))
                    assert(values[count].x == 99)
                empty: list[Empty] = []
                empty.append(Empty())
                empty.append(Empty())
                assert(len(empty) == 2)
                numbers: list[int] = []
                for index in range(20):
                    numbers.append(index)
                assert(numbers[19] == 19)
                flags: list[bool] = []
                flags.append(true)
                flags.append(false)
                assert(flags[0])
                assert(not flags[1])
                pairs = [(1, true)]
                pairs.append((2, false))
                assert(pairs[1][0] == 2)
                assert(not pairs[1][1])
        ''')
        executable = self.directory / "sanitized"
        self.succeeds([tool("SEVERIAN_CLANG", "clang-21"), self.directory / "subject.ll",
                       "-fsanitize=address", "-o", executable, "-lm"])
        result = subprocess.run([executable], capture_output=True, text=True,
                                env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        tracker = self.directory / "allocations.c"
        tracker.write_text('''
            #include <stddef.h>
            #include <stdlib.h>
            #include <stdio.h>
            void *__real_malloc(size_t);
            void __real_free(void *);
            static long live;
            void *__wrap_malloc(size_t n) {
                void *p = __real_malloc(n);
                if (p) ++live;
                return p;
            }
            void __wrap_free(void *p) {
                if (p) --live;
                __real_free(p);
            }
            __attribute__((destructor)) static void check(void) {
                if (live) {
                    fprintf(stderr, "outstanding allocations: %ld\\n", live);
                    _Exit(97);
                }
            }
        ''')
        tracked = self.directory / "tracked"
        self.succeeds([tool("SEVERIAN_CLANG", "clang-21"), self.directory / "subject.ll",
                       tracker, "-Wl,--wrap=malloc", "-Wl,--wrap=free", "-o", tracked, "-lm"])
        self.succeeds([tracked])

    def test_general_receiver_replacement_and_returns(self):
        self.native('''
            type Sequence[T] = buffer[T]
            type Integers = list[int]
            extend Integers:
                def reset():
                    self = []
            def traced(value: int) -> int:
                print(value)
                return value
            extend Sequence[T]:
                def replace(values: Sequence[T]) -> int:
                    previous = len(self)
                    self = values
                    return previous
                def repeated(value: T, count: int = 1) -> int:
                    if count < 1:
                        return len(self)
                    index := 0
                    while true:
                        self.append(value)
                        index += 1
                        if index == count:
                            break
                    return len(self)
                def first_or(value: T) -> T:
                    if len(self) != 0:
                        return self[0]
                    self.append(value)
                    return value
            test:
                values := [1, 2]
                assert(values.replace([7]) == 2)
                assert(len(values) == 1)
                assert(values[0] == 7)
                assert(values.repeated(9, 0) == 1)
                assert(values.repeated(9, 3) == 4)
                assert(values[3] == 9)
                assert(values.repeated(count=traced(2), value=traced(5)) == 6)
                assert(values[5] == 5)
                assert(values.repeated(6) == 7)
                values.reset()
                assert(len(values) == 0)
                assert(values.first_or(42) == 42)
                assert(values[0] == 42)
                assert(values.first_or(99) == 42)
                assert(len(values) == 1)
        ''', expected="2\n5\n")

    def test_receiver_patterns_do_not_match_other_types(self):
        self.native('''
            class Box[T]:
                item: T
            extend Box[T]:
                def item_value() -> T:
                    return self.item
            type Sequence[T] = buffer[T]
            extend Sequence[T]:
                def item_value() -> T:
                    return self[0]
            test:
                box = Box[int](7)
                assert(box.item_value() == 7)
                values = [9]
                assert(values.item_value() == 9)
        ''')

    def test_element_and_receiver_contracts(self):
        self.rejects('''
            class Point:
                x: int
            def bad():
                values: list[Point] = []
                values.append(3)
        ''', r"type mismatch|expected type")
        self.rejects('''
            def bad():
                value = 1
                value.append(2)
        ''', r"unknown callable")
        self.rejects('''
            type Sequence[T] = buffer[T]
            extend Sequence[T]:
                def collide() -> int:
                    return 1
            extend list[T]:
                def collide() -> int:
                    return 2
            def bad() -> int:
                values = [1]
                return values.collide()
        ''', r"ambiguous extension method")

    def test_unsupported_receiver_lifetimes_are_diagnosed(self):
        self.rejects('''
            extend list[T]:
                def recursive():
                    self.recursive()
            def bad(values: list[int]):
                values.recursive()
        ''', r"recursive buffer receiver methods require a reference ABI")
        self.native('''
            class Holder:
                values: list[int]
            def append(holder: Holder):
                holder.values.append(2)
            test:
                holder = Holder([1])
                append(holder)
                assert(holder.values[0] == 1)
                assert(holder.values[1] == 2)
        ''')


if __name__ == "__main__":
    unittest.main()
