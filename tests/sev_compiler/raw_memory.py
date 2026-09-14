#!/usr/bin/env python3
"""Native typed pointer behavior and unsafe-context rejection gates."""
import unittest
import os
import subprocess

from migration import MigrationCase, ROOT
from bootstrap_mlir import tool


class RawMemory(MigrationCase):
    def test_byte_literals_and_allocation_keep_element_counts_separate(self):
        self.native('''
            test:
                empty: byte = 0B
                width: byte = 1B
                assert(empty == 0b)
                assert(width == 8b)
                assert(1KB == 1000B)
                assert(1KiB == 1024B)
                unsafe:
                    raw = allocate(7B)
                    raw[0B] = u8(65)
                    raw[6B] = u8(90)
                    assert(raw[0] == u8(65))
                    assert(raw[6] == u8(90))
                    free(raw)
                    zero = allocate(0B)
                    free(zero)
                    typed = allocate[u32](2)
                    typed[0] = u32(23)
                    typed[4B] = u32(45)
                    assert(typed[0B] == u32(23))
                    assert(typed[1] == u32(45))
                    free(typed)
        ''')

    def test_container_requires_counts_and_byte_storage(self):
        self.native('''
            class Buffer: Container:
                def len() -> int:
                    return 3
                def size() -> int:
                    return 3
                def bytes() -> byte:
                    return 12B
            def storage[C: Container](value: C) -> byte:
                assert(value.len() == 3)
                assert(value.size() == 3)
                return bytes(value)
            test:
                assert(storage(Buffer()) == 12B)
                values: list[u32] = [1, 2, 3]
                assert(storage(values) == 12B)
        ''')
        for missing in ['len', 'size', 'bytes']:
            methods = {
                'len': 'def len() -> int:\n        return 3',
                'size': 'def size() -> int:\n        return 3',
                'bytes': 'def bytes() -> byte:\n        return 12B',
            }
            text = 'class Incomplete: Container:\n    ' + '\n    '.join(
                body for name, body in methods.items() if name != missing)
            with self.subTest(missing=missing):
                self.rejects(text, 'does not satisfy|does not implement|missing')

    def test_pointer_string_conversion_preserves_address_identity(self):
        self.native('''
            test:
                values: list[u8] = [65, 66, 67]
                unsafe:
                    p = &values[0]
                    address = string(p)
                    assert(address[:2] == "0x")
                    assert(address != "0x0")
                    assert(address == string(&values[0]))
                    assert(address != string(&values[1]))
                    assert(p[0] == u8(65))
        ''')

    def test_vector_reports_owned_bytes_separately_from_element_count(self):
        self.native(f'''
            import * from "{os.path.relpath(ROOT / 'library/core/collections/vector/src/vector.sev', self.directory)}" as vectors
            test:
                values = vectors.vector[u32, 2]()
                assert(len(values) == 0)
                assert(size(values) == 0)
                assert(bytes(values) == 8B)
                values.append(u32(1))
                values.append(u32(2))
                assert(len(values) == 2)
                assert(bytes(values) == 8B)
                values.append(u32(3))
                assert(size(values) == 3)
                assert(bytes(values) == 16B)
                values.clear()
                assert(len(values) == 0)
                assert(bytes(values) == 16B)
        ''')

    def test_byte_allocation_rejects_integer_sizes_and_safe_calls(self):
        self.rejects('def use():\n    memory = allocate(7B)', 'requires an unsafe scope')
        self.rejects('''
            def use():
                unsafe:
                    memory = allocate(7)
        ''', 'no overload|does not match|argument')
        self.rejects('''
            def use():
                unsafe:
                    memory = allocate[u32](7B)
        ''', 'does not match|expected|argument')

    def test_generic_type_layout_in_member_and_arithmetic_inference(self):
        self.native('''
            def width[T](value: T) -> int:
                return int(bytes(T).amount / 8.0)
            test:
                assert(width(i32(1)) == 4)
                assert(width(i64(1)) == 8)
                assert(width(u8(1)) == 1)
        ''')

    def test_value_byte_size_uses_its_source_method(self):
        self.native('''
            class Sized:
                count: int
                def bytes() -> byte:
                    return data_size(count)
            test:
                assert(bytes(Sized(23)) == 23B)
                assert(bytes(Sized(0)) == 0B)
        ''')

    def test_documented_allocation_and_pointer_examples(self):
        paths = [ROOT / 'docs/examples/01-types/04-memory/01-allocation.sev']
        paths.extend(sorted((ROOT / 'docs/examples/01-types/04-pointers').glob('*.sev')))
        for path in paths:
            with self.subTest(example=path.name):
                self.native(path.read_text(), name=path.name)

    def test_element_layout_aliasing_and_zero_allocation(self):
        self.native('''
            class Padded:
                head: u8
                value: int
            test:
                unsafe:
                    memory = allocate[Padded](2)
                    memory[0] = Padded(1, 23)
                    memory[1] = Padded(2, 45)
                    alias = memory
                    assert(alias[1].value == 45)
                    memory[0] = Padded(3, 67)
                    assert(alias[0].value == 67)
                    free(memory)
                    empty = allocate[int](0)
                    free(empty)
        ''')

    def test_buffer_address_writes_original_storage(self):
        self.native('''
            test:
                values := [10, 20, 30]
                unsafe:
                    address = &values[1]
                    address[0] = 99
                    assert(address[1] == 30)
                assert(values[1] == 99)
        ''')

    def test_buffer_owner_survives_pointer_control_flow(self):
        self.native('''
            def check(choose: bool):
                values := [10, 20, 30]
                unsafe:
                    address = &values[0]
                    if choose:
                        assert(address[1] == 20)
                    else:
                        assert(address[2] == 30)
            test:
                check(true)
                check(false)
        ''')
        executable = self.directory / 'sanitized'
        self.succeeds([tool('SEVERIAN_CLANG', 'clang-21'), self.directory / 'subject.ll', ROOT / 'library/core/memory/native/memory.c',
                       '-fsanitize=address', '-o', executable, '-lm'])
        result = subprocess.run([executable], capture_output=True, text=True,
                                env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'}, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsafe_scope_does_not_escape(self):
        for text in [
            'def use():\n    memory = allocate[int](1)',
            'def use(value: pointer[int]):\n    print(value[0])',
            'def use(value: pointer[int]):\n    free(value)',
            'def use():\n    value := 1\n    address = &value',
            'def use():\n    unsafe:\n        pass\n    memory = allocate[int](1)',
        ]:
            with self.subTest(source=text):
                self.rejects(text, 'requires an unsafe scope')

    def test_invalid_pointee_and_address_are_rejected(self):
        self.rejects('''
            def use():
                unsafe:
                    value: pointer[u8] = allocate[int](1)
        ''', 'expected pointee type')
        self.rejects('''
            def use():
                unsafe:
                    value = &(1 + 2)
        ''', 'address requires')
        self.rejects('''
            def use():
                unsafe:
                    memory = allocate[string](1)
        ''', 'copyable element storage')


if __name__ == '__main__':
    unittest.main()
