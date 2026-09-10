#!/usr/bin/env python3
"""Native typed pointer behavior and unsafe-context rejection gates."""
import unittest
import os
import subprocess

from migration import MigrationCase, ROOT
from bootstrap_mlir import tool


class RawMemory(MigrationCase):
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
        self.succeeds([tool('SEVERIAN_CLANG', 'clang-21'), self.directory / 'subject.ll',
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
