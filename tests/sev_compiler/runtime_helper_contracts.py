#!/usr/bin/env python3
"""SIP-0002: preserve typed collection contracts and owned-buffer rejection."""
import unittest

from migration import MigrationCase


class RuntimeHelperContracts(MigrationCase):
    def test_supported_buffer_types_in_both_orders(self):
        declarations = [
            'narrow: list[i32] = [1, 2]',
            'wide: list[i64] = [3, 4]',
        ]
        for ordered in (declarations, declarations[::-1]):
            with self.subTest(order=ordered):
                self.native('test:\n' + ''.join(f'    {line}\n' for line in ordered) + '''
    narrow.append(5)
    wide.append(6)
    assert(narrow[0] == 1)
    assert(narrow[2] == 5)
    assert(wide[0] == 3)
    assert(wide[2] == 6)
''')

    def test_nested_owned_buffers_still_require_destruction_lowering(self):
        for element, value in [('int', '1'), ('string', '"hello"')]:
            with self.subTest(element=element):
                self.rejects(
                    f'def use():\n    values: list[list[{element}]] = [[{value}]]',
                    'owned elements require destruction lowering',
                )

    def test_incompatible_elements_are_rejected(self):
        self.rejects('''
            def use():
                values: list[int] = [1]
                values.append("wrong")
        ''', 'expected scalar type')


if __name__ == '__main__':
    unittest.main()
