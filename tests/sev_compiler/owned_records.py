#!/usr/bin/env python3
"""Owned record fields remain visible to CFG and MLIR ownership lowering."""
import os
import re
import subprocess
import unittest

from bootstrap_mlir import tool
from migration import COMPILER, MigrationCase, ROOT


class OwnedRecords(MigrationCase):
    def test_string_behavior_comes_from_source(self):
        origin = ROOT / 'sev_compiler/universal/prelude.sev'
        sysroot = self.directory / 'sysroot'
        prelude = sysroot / 'sev_compiler/universal/prelude.sev'
        prelude.parent.mkdir(parents=True)
        prelude.with_suffix(".toml").write_text((ROOT / "sev_compiler/universal/prelude.toml").read_text())
        provider = prelude.parent / 'methods.sev'
        source = origin.parent / 'primitive/string/methods.sev'

        def imports(text, base, target, replacement=None):
            def rewrite(match):
                path = (base / match[1]).resolve()
                if replacement and path == source:
                    path = replacement
                return f'import * from "{os.path.relpath(path, target)}"'
            return re.sub(r'^import \* from "([^"]+)"', rewrite, text, flags=re.MULTILINE)

        body = imports(source.read_text(), source.parent, provider.parent)
        provider.write_text(body.replace('def replace(', 'def substitute(').replace(
            'operator int(value: string)', 'def parse_decimal(value: string)'))
        prelude.write_text(imports(origin.read_text(), origin.parent, prelude.parent, provider))
        self.rejects('def use():\n    value = "x".replace("x", "y")',
                     'unknown callable', sysroot=sysroot)
        self.rejects('def use():\n    value = int("42")',
                     'explicit conversion requires numeric primitive types', sysroot=sysroot)
        self.native('''
            test:
                assert("x".substitute("x", "y") == "y")
                assert(string_methods.parse_decimal("42") == 42)
        ''', sysroot=sysroot)

    def test_unchanged_examples(self):
        examples = ROOT / 'docs/examples/01-types/05-generics'
        for number in (26, 27):
            path, = examples.glob(f'{number:02d}-*.sev')
            with self.subTest(example=path.name):
                self.native(path.read_text(), name=path.name)
        path = examples / '04-boxed-generic.sev'
        executable = self.directory / 'boxed'
        self.succeeds([COMPILER, 'build', path, '--sysroot', ROOT, '-o', executable])
        output = self.succeeds([executable]).splitlines()
        self.assertEqual(len(output), 3)
        self.assertEqual(output[0], '30')
        self.assertEqual(float(output[1]), 7.0)
        self.assertEqual(output[2], 'Hello,World!')

    def test_nested_fields_mutation_returns_and_lifetimes(self):
        self.native('''
            class Span:
                start: int
                end: int
            class Text:
                text: string
                span: Span
                def append(suffix: string):
                    text += suffix
                    span.end += 1
            class Packet:
                item: Text
                numbers: list[int]
                def add(value: int):
                    numbers.append(value)
            def make(prefix: string, count: int) -> Packet:
                packet = Packet(Text(prefix + "!", Span(0, count)), [])
                for index in range(count):
                    packet.item.append("x")
                    packet.add(index)
                return packet
            def choose(left: Packet, right: Packet, take_left: bool) -> Packet:
                if take_left:
                    return left
                return right
            test:
                packet := make("start", 4)
                assert(packet.item.text == "start!xxxx")
                assert(packet.item.span.end == 8)
                assert(packet.numbers[3] == 3)
                saved = packet
                packet.item.append("z")
                packet.add(9)
                assert(saved.item.text == "start!xxxx")
                assert(len(saved.numbers) == 4)
                assert(packet.item.text == "start!xxxxz")
                other = make("other", 2)
                selected = choose(packet, other, false)
                assert(selected.item.text == "other!xx")
                conditional = packet if true else other
                assert(conditional.item.text == "start!xxxxz")
                for count in range(16):
                    packet = make("loop", count)
                    packet.item.append("end")
                    assert(len(packet.numbers) == count)
                assert(saved.item.text == "start!xxxx")
                assert(packet.item.text == "loop!xxxxxxxxxxxxxxxend")
        ''')
        self.check_allocations()

    def test_optional_and_error_payload_lifetimes(self):
        self.native((ROOT / 'docs/examples/01-types/01-basic/07-none-absent.sev').read_text())
        self.check_allocations()
        self.native('''
            class Failure: Error:
                message: string
            class Update:
                name: string | None | absent
            def checked(valid: bool) -> string | Failure:
                if not valid:
                    throw Failure("bad" + "!")
                return "good" + "!"
            def forward(valid: bool) -> string | Failure:
                return checked(valid)
            test:
                update = Update()
                assert(update.name is absent)
                update.name = "initial" + "!"
                saved = update
                for index in range(8):
                    update.name = None
                    assert(update.name is None)
                    update.name = "changed" + "!"
                assert(saved.name == "initial!")
                assert(update.name == "changed!")
                result ?= forward(false)
                if result is Failure:
                    assert(result.message == "bad!")
                else:
                    assert(false)
                assert(forward(true) == "good!")
        ''')
        self.check_allocations()

    def check_allocations(self):
        executable = self.directory / 'sanitized'
        self.succeeds([tool('SEVERIAN_CLANG', 'clang-21'), self.directory / 'subject.ll',
                       '-fsanitize=address', '-o', executable, '-lm'])
        result = subprocess.run([executable], capture_output=True, text=True,
                                env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'}, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        tracker = self.directory / 'allocations.c'
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
        tracked = self.directory / 'tracked'
        self.succeeds([tool('SEVERIAN_CLANG', 'clang-21'), self.directory / 'subject.ll',
                       tracker, '-Wl,--wrap=malloc', '-Wl,--wrap=free', '-o', tracked, '-lm'])
        self.succeeds([tracked])

    def test_generic_record_inference_preserves_types_and_copies(self):
        self.native('''
            class Pair[A, B]:
                left: A
                right: B
            class Box[T]:
                value: T
                def append(suffix: T):
                    value += suffix
            def same[T](value: T) -> T:
                return value
            test:
                box = Box("a" + "b")
                duplicate = same(box)
                duplicate.append("c")
                assert(box.value == "ab")
                assert(duplicate.value == "abc")
                pair = Pair(right=Box(5), left=box)
                assert(pair.left.value == "ab")
                assert(pair.right.value == 5)
                assert(Box("value").value == "value")
        ''')
        self.rejects('''
            class Pair[T]:
                left: T
                right: T
            def bad():
                value = Pair(1, "text")
        ''', 'conflicting generic type arguments')

    def test_source_string_replacement_and_integer_conversion(self):
        self.native('''
            test:
                assert("1_024".replace("_", "") == "1024")
                assert("aaaa".replace("aa", "b") == "bb")
                assert("λxλ".replace("λ", "😀") == "😀x😀")
                assert("λ😀".replace("", ":") == ":λ:😀:")
                assert("".replace("", "-") == "-")
                assert("abc".replace("absent", "x") == "abc")
                assert("abc".replace("", "") == "abc")
                assert(int("  +42 ") == 42)
                assert(int("-42") == -42)
                assert(int("-9223372036854775808") == -9223372036854775808)
                assert(int("9223372036854775807") == 9223372036854775807)
        ''')
        self.check_allocations()

    def test_invalid_integer_text_traps(self):
        for value in ('', '+', '12x', '9223372036854775808', '-9223372036854775809'):
            with self.subTest(value=value):
                path = self.write(f'def main():\n    value = int("{value}")\n')
                executable = self.directory / 'invalid'
                self.succeeds([COMPILER, 'build', path, '--sysroot', ROOT, '-o', executable])
                self.assertNotEqual(self.invoke([executable]).returncode, 0)

    def test_unhandled_owned_error_traps(self):
        path = self.write('''
            class Failure: Error:
                message: string
            def failed() -> string | Failure:
                throw Failure("bad" + "!")
            def main():
                value: string = failed()
                print(value)
                print("unreachable")
        ''')
        executable = self.directory / 'unhandled'
        self.succeeds([COMPILER, 'build', path, '--sysroot', ROOT, '-o', executable])
        result = self.invoke([executable])
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('unreachable', result.stdout)

    def test_unsupported_owned_boundaries_are_diagnosed(self):
        self.rejects('''
            class Text:
                value: string
            def recursive(value: Text) -> Text:
                return recursive(value)
            def use():
                value = recursive(Text("x"))
        ''', 'require a reference ABI')
        self.rejects('''
            class Text:
                value: string
            def replace(value: Text):
                value = Text("other")
            def use():
                value = Text("x")
                replace(value)
        ''', 'parameter rebinding requires a reference ABI')
        self.rejects('''
            class Text:
                value: string
            def use():
                values = [Text("x")]
        ''', 'owned elements require destruction lowering')


if __name__ == '__main__':
    unittest.main()
