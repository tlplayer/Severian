#!/usr/bin/env python3
"""Deferred calls execute at every structured scope exit."""
import unittest

from migration import MigrationCase, ROOT


class DeferredCleanup(MigrationCase):
    def test_temporary_resources_example(self):
        self.native((ROOT / 'docs/examples/03-testing/02-with-tests/17-temporary-resources.sev').read_text())

    def test_reverse_order_on_fallthrough_and_early_return(self):
        self.native('''
            def work(early: bool):
                defer print("outer")
                if early:
                    defer print("inner")
                    return
                defer print("last")
            test:
                work(true)
                work(false)
        ''', expected='inner\nouter\nlast\nouter\n')

    def test_return_expression_is_evaluated_before_cleanup(self):
        self.native('''
            def change(values: list[int]):
                values[0] = 99
            def work() -> int:
                values := [42]
                defer change(values)
                return values[0]
            test:
                assert(work() == 42)
        ''')

    def test_cleanup_on_loop_fallthrough_continue_and_break(self):
        self.native('''
            test:
                defer print("outer")
                for value in range(0, 4):
                    defer print(value)
                    if value == 0:
                        continue
                    if value == 2:
                        break
                    print("body")
                print("done")
        ''', expected='0\nbody\n1\n2\ndone\nouter\n')

    def test_deferred_call_precedes_resource_destruction(self):
        self.native('''
            class Resource:
                id: int
                def show():
                    print(id)
                def drop():
                    print("drop")
            test:
                resource = Resource(42)
                defer resource.show()
        ''', expected='42\ndrop\n')

    def test_deferred_uses_cannot_outlive_their_bindings(self):
        self.rejects('''
            def work():
                value := "owned"
                defer print(value)
                drop(value)
        ''', 'use after drop or move')
        self.rejects('def work():\n    defer 42', 'defer requires a call')

    def test_error_and_optional_return_tags_survive_cleanup(self):
        self.native('''
            class Failure: Error:
                message: string
            def failed() -> string | Failure:
                defer print("error cleanup")
                throw Failure("bad" + "!")
            def optional(found: bool) -> int | None:
                defer print("optional cleanup")
                if found:
                    return 42
                return None
            test:
                result ?= failed()
                assert(result is Failure)
                present ?= optional(true)
                missing ?= optional(false)
                assert(present is int)
                assert(missing is None)
        ''', expected='error cleanup\noptional cleanup\noptional cleanup\n')


if __name__ == '__main__':
    unittest.main()
