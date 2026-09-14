#!/usr/bin/env python3
"""Aliases preserve identity; applied extensions select a concrete receiver."""
import unittest

from migration import MigrationCase


class AliasExtensions(MigrationCase):
    def test_alias_and_direct_extension_share_the_same_type(self):
        self.native('''
            class Box[T]:
                value: T
            Box[int] as IntBox
            extend Box[int]:
                def doubled() -> int:
                    return value + value
            test:
                assert(IntBox(21).doubled() == 42)
                assert(Box[int](7).doubled() == 14)
        ''')

    def test_concrete_extension_does_not_apply_to_other_arguments(self):
        self.rejects('''
            class Box[T]:
                value: T
            extend Box[int]:
                def doubled() -> int:
                    return value + value
            test:
                Box[bool](true).doubled()
        ''', 'doubled')

    def test_old_alias_spelling_is_rejected(self):
        self.rejects('type Count = int\n', 'type aliases use B as X')


if __name__ == '__main__':
    unittest.main()
