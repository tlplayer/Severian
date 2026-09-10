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

    def test_owned_builder_initializes_fields_before_escape(self):
        path = ROOT / 'docs/examples/06-ownership/02-owned-builder.sev'
        self.native(path.read_text())
        for body, diagnostic in [
            ('item := Pair()\n    print(item.name)', 'uninitialized field'),
            ('item := Pair()\n    item.name = "set"\n    return item', 'partially initialized'),
            ('item := Pair()\n    if flag:\n        item.name = "set"\n    item.count = 1\n    return item', 'partially initialized'),
        ]:
            with self.subTest(body=body):
                self.rejects('class Pair:\n    name: string\n    count: int\n'
                             'def make(flag: bool) -> Pair:\n    ' + body, diagnostic)

    def test_inferred_parameter_effects_example(self):
        path = ROOT / 'docs/examples/06-ownership/04-inferred-parameter-effects.sev'
        self.native(path.read_text())

    def test_owned_string_collection_storage_and_returns(self):
        self.native('''
            def make() -> list[string]:
                return ["a" + "λ", "😀", ""]
            test:
                values := make()
                assert(len(values) == 3)
                assert(values[0] == "aλ")
                assert(values[1] == "😀")
                saved = clone values
                values[0] = "changed"
                values.append("tail" + "!")
                assert(saved[0] == "aλ")
                assert(values[1] == "😀")
                assert(values[2] == "")
                assert(values[3] == "tail!")
                values.clear()
                assert(len(values) == 0)
                assert(len(saved) == 3)
        ''')
        OwnedRecords.check_allocations(self)

    def test_collection_moves_leave_other_elements_available(self):
        path = ROOT / 'docs/examples/06-ownership/06-collections.sev'
        self.native(path.read_text())
        self.native('''
            test:
                values := ["a", "b"]
                removed = move values[0]
                assert(values[1] == "b")
                reject:
                    print(values[0])
                values[0] = "replacement"
                assert(values.length() == 2)
                assert(values[0] == "replacement")
                for value in borrow values:
                    assert(value.length() > 0)
                values.append("after iteration")
                assert(values.length() == 3)
        ''')
        OwnedRecords.check_allocations(self)

    def test_scalar_closure_captures_snapshot_values(self):
        path = ROOT / 'docs/examples/06-ownership/08-closures.sev'
        self.native(path.read_text())
        self.native('''
            def argument() -> int:
                print("once")
                return 4
            test:
                offset := 3
                add = lambda value: value + offset
                offset = 99
                assert(add(argument()) == 7)
                assert(add(value=5) == 8)
                product = lambda left, right: left * right
                assert(product(6, 7) == 42)
        ''', expected='once\n')

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
