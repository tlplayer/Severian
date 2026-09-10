#!/usr/bin/env python3
"""Explicit ownership operations must execute and enforce their contracts."""
import unittest

from migration import MigrationCase, ROOT
from owned_records import OwnedRecords


class ExplicitOwnership(MigrationCase):
    def test_unchanged_examples(self):
        directory = ROOT / 'docs/examples/06-ownership'
        for filename, expected in [
            ('01-borrow-clone-move.sev', 'severian\nseverian\nseverian\n'),
            ('03-shared-read-exclusive-write.sev', ''),
            ('13-conditional-moves.sev', 'hello\nhello\n'),
            ('15-compiler-rejections.sev', ''),
        ]:
            with self.subTest(example=filename):
                self.native((directory / filename).read_text(), expected=expected, name=filename)

    def test_clone_allocates_independent_collection_storage(self):
        self.native('''
            test:
                original := [1, 2, 3]
                duplicate := clone original
                duplicate[0] = 99
                assert(original[0] == 1)
                assert(duplicate[0] == 99)
                text = "allocated" + " text"
                copied = clone text
                assert(copied == "allocated text")
        ''')
        OwnedRecords.check_allocations(self)

    def test_move_and_borrow_rejections_reach_semantics(self):
        cases = [
            ('value := "owned"\n    consumed = move value\n    print(value)', 'use after drop or move'),
            ('value := [1]\n    view = borrow value\n    value[0] = 2', 'active borrow'),
            ('value := [1]\n    view = borrow value\n    other = borrow mut value', 'active borrow'),
            ('value := [1]\n    view = borrow mut value\n    print(value[0])', 'active borrow'),
            ('value := [1]\n    view = borrow value\n    drop(value)', 'active borrow'),
            ('value := [1]\n    view = borrow value\n    view[0] = 2', 'shared borrow'),
        ]
        for body, diagnostic in cases:
            with self.subTest(body=body):
                self.rejects('def use():\n    ' + body, diagnostic)

    def test_shared_parameter_cannot_mutate_or_escape(self):
        self.rejects('''
            def change(value: borrow list[int]):
                value[0] = 2
        ''', 'shared borrow parameter')
        self.rejects('''
            def bad() -> borrow string:
                value = "temporary"
                return borrow value
        ''', 'returned borrow')

    def test_temporary_borrows_end_after_the_call(self):
        self.native('''
            def read(value: borrow list[int]) -> int:
                return value[0]
            def change(value: borrow mut list[int]):
                value[0] += 1
            test:
                value := [1]
                assert(read(borrow value) == 1)
                change(borrow mut value)
                assert(read(borrow value) == 2)
        ''')

    def test_exclusive_scalar_borrow_updates_its_owner(self):
        self.native('''
            def increment(value: borrow mut int):
                value += 1
            test:
                value := 41
                increment(borrow mut value)
                assert(value == 42)
                alias := borrow mut value
                alias = 99
                drop(alias)
                assert(value == 99)
        ''')


if __name__ == '__main__':
    unittest.main()
