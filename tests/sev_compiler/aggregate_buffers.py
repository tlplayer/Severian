#!/usr/bin/env python3
"""Copyable aggregate storage through source-defined list protocols."""
import os
import re
import subprocess
import unittest

from migration import MigrationCase, ROOT
from bootstrap_mlir import tool


class AggregateBuffers(MigrationCase):
    def test_record_generic_examples(self):
        for number in (7, 11, 13, 14, 19, 21):
            path, = (ROOT / "docs/examples/01-types/04-generics").glob(f"{number:02d}-*.sev")
            with self.subTest(example=path.name):
                self.native(path.read_text(), name=path.name)

    def test_returned_buffers_and_value_elements(self):
        self.native('''
            class Point:
                x: int
                y: int
            def make() -> list[Point]:
                return [Point(10, 20), Point(30, 40)]
            def replace(values: list[Point]):
                values[1] = Point(7, 9)
            test:
                points := make()
                assert(len(points) == 2)
                assert(size(points) == 2)
                replace(points)
                assert(points[1].x == 7)
                value := points[0]
                value.x = 99
                assert(points[0].x == 10)
                total := 0
                for point in points:
                    total += point.x + point.y
                assert(total == 46)
                empty: list[Point] = []
                assert(len(empty) == 0)
                if empty:
                    assert(false)
                if points:
                    total += 1
                assert(total == 47)
                alias = points
                duplicate = copy points
                assert(alias is points)
                assert(!(duplicate is points))
                duplicate[0] = Point(1, 2)
                assert(points[0].x == 10)
                reversed = points[::-1]
                assert(reversed[0].x == 7)
                assert(reversed[1].y == 20)
                assert(len(points[1:1]) == 0)
                for point in empty:
                    assert(false)
                points = make()
                assert(points[1].y == 40)
                count := 0
                while count < 8:
                    points = make()
                    points[0] = Point(count, 0)
                    assert(points[0].x == count)
                    count += 1
                assert(points[0].x == 7)
        ''')

        # Check that returning/reassigning an allocation preserves its lifetime.
        executable = self.directory / "sanitized"
        self.succeeds([tool("SEVERIAN_CLANG", "clang-21"), self.directory / "subject.ll",
                       "-fsanitize=address", "-o", executable, "-lm"])
        result = subprocess.run([executable], capture_output=True, text=True,
                                env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        # LeakSanitizer cannot run under the sandbox's tracing setup. Count
        # allocations emitted by MLIR, independently of the system allocator.
        tracker = self.directory / "allocations.c"
        tracker.write_text("""
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
        """)
        tracked = self.directory / "tracked"
        self.succeeds([tool("SEVERIAN_CLANG", "clang-21"), self.directory / "subject.ll",
                       tracker, "-Wl,--wrap=malloc", "-Wl,--wrap=free", "-o", tracked, "-lm"])
        self.succeeds([tracked])

    def test_nested_mixed_layout_and_empty_records(self):
        self.native('''
            class Small:
                flag: bool
                value: i16
            class Mixed:
                first: u8
                inner: Small
                amount: float
                last: u64
            class Empty:
                pass
            test:
                maximum: u64 = 18446744073709551615
                values = [Mixed(3, Small(true, -123), 1.25, maximum), Mixed(9, Small(false, 4), 2.5, 7)]
                assert(len(values) == 2)
                assert(values[0].inner.flag)
                assert(values[0].inner.value == -123)
                assert(values[0].amount == 1.25)
                assert(values[0].last == maximum)
                assert(values[1].first == 9)
                assert(not values[1].inner.flag)
                assert(values[1].amount == 2.5)
                empties = [Empty(), Empty()]
                assert(len(empties) == 2)
                count := 0
                for value in empties:
                    count += 1
                assert(count == 2)
        ''')

    def test_source_tuples_are_buffer_elements(self):
        self.native('''
            def sum_first(values: list[tuple[int, int]]) -> int:
                total := 0
                for pair in values:
                    total += pair[0]
                return total
            test:
                pairs = [(10, 20), (32, 40)]
                assert(sum_first(pairs) == 42)
                reversed = pairs[::-1]
                assert(reversed[0][1] == 40)
                assert(reversed[1][0] == 10)
        ''')

    def test_list_protocols_require_the_source_provider(self):
        origin = ROOT / "sev_compiler/universal/prelude.sev"
        sysroot = self.directory / "sysroot"
        target = sysroot / "sev_compiler/universal/prelude.sev"
        target.parent.mkdir(parents=True)
        providers = []
        complete = []
        for line in origin.read_text().splitlines():
            match = re.fullmatch(r'import "([^"]+)"(.*)', line)
            if match:
                path = (origin.parent / match[1]).resolve()
                entry = f'import "{os.path.relpath(path, target.parent)}"{match[2]}\n'
                complete.append(entry)
                if path.name != "collections.sev":
                    providers.append(entry)
        program = '''
            class Item:
                value: int
            def read() -> int:
                values = [Item(42)]
                return values[0].value
            test:
                print(read())
        '''
        target.write_text("".join(providers))
        self.rejects(program, "missing source protocol (implementation )?List.construction", sysroot=sysroot)
        target.write_text("".join(complete))
        self.native(program, sysroot=sysroot, expected="42\n")

    def test_result_inference_for_typed_generic_boundary(self):
        self.native('''
            @mlir("arith.index_cast")
            def position(value: int) -> index
            @mlir("arith.index_cast")
            def recover[T](value: index) -> T
            def answer() -> int:
                return recover(position(42))
            test:
                assert(answer() == 42)
                small: i16 = recover(position(7))
                assert(small == 7)
        ''')

    def test_type_qualified_methods_use_explicit_arguments(self):
        self.native('''
            trait Factory:
                def create(value: int) -> Self
            class Item: Factory
                value: int
                def create(value: int) -> Self:
                    return Item(value)
                def read() -> int:
                    return value
            def construct[T: Factory](value: int) -> T:
                return T.create(value)
            test:
                direct = Item.create(7)
                assert(direct.read() == 7)
                inferred: Item = construct(42)
                assert(inferred.read() == 42)
        ''')
        self.rejects('''
            class Item:
                value: int
                def read() -> int:
                    return value
            def main():
                print(Item.read())
        ''', "unknown name self")

    def test_alias_arguments_keep_the_callers_scope(self):
        self.write('type Sequence[T] = list[T]\n', "container.sev")
        self.native('''
            import "container.sev" as container
            class Local:
                value: int
            def identity[V](values: container.Sequence[V]) -> container.Sequence[V]:
                return values
            def make() -> container.Sequence[Local]:
                return [Local(42)]
            test:
                values = identity(make())
                assert(values[0].value == 42)
        ''')

    def test_storage_type_mismatch_and_owned_elements_rejected(self):
        self.rejects('''
            class A:
                value: int
            class B:
                value: int
            def consume(values: list[A]):
                pass
            def main():
                consume([B(1)])
        ''', "(type|overload)")
        self.rejects('''
            def main():
                values = ["owned"]
        ''', "owned elements require destruction lowering")
        self.rejects('''
            class Resource:
                handle: int
                def drop():
                    print(handle)
            def main():
                values = [Resource(1)]
        ''', "owned elements require destruction lowering")


if __name__ == "__main__":
    unittest.main()
