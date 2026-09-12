#!/usr/bin/env python3
"""Native destructor order, transfers, and lifetime regression checks."""
import unittest
from migration import MigrationCase

RESOURCE = '''
class Resource:
    id: int
    operator drop(move self) -> unit:
        print(id)
'''

class Destruction(MigrationCase):
    def test_custom_cleanup_precedes_reverse_owned_fields(self):
        self.native(RESOURCE + '''
class Parent:
    first: Resource
    second: Resource
    operator drop(move self) -> unit:
        print(0)
test:
    parent = Parent(Resource(1), Resource(2))
''', expected='0\n2\n1\n')

    def test_default_cleanup_and_move(self):
        self.native(RESOURCE + '''
class Parent:
    child: Resource
test:
    resource = Resource(3)
    parent = Parent(move resource)
    moved = move parent
''', expected='3\n')

    def test_return_transfers_result_and_cleans_other_locals(self):
        self.native(RESOURCE + '''
def make() -> Resource:
    other = Resource(4)
    result = Resource(5)
    return result
test:
    result = make()
    print(0)
''', expected='4\n0\n5\n')

    def test_early_return_runs_hook_once(self):
        self.native('''
class Resource:
    id: int
    operator drop(move self) -> unit:
        if id == 1:
            print(1)
            return
        print(2)
test:
    first = Resource(1)
    second = Resource(2)
''', expected='2\n1\n')

    def test_loop_iteration_continue_and_break(self):
        self.native(RESOURCE + '''
test:
    index := 0
    while index < 3:
        item = Resource(index)
        index += 1
        if index == 1:
            continue
        if index == 2:
            break
''', expected='0\n1\n')

    def test_reassignment_destroys_previous_owner(self):
        self.native(RESOURCE + """
test:
    item := Resource(1)
    item = Resource(2)
    drop(item)
    item = Resource(3)
""", expected='1\n2\n3\n')

    def test_partial_move_and_explicit_field_drop(self):
        self.native(RESOURCE + """
class Parent:
    first: Resource
    second: Resource
test:
    parent = Parent(Resource(1), Resource(2))
    taken = move parent.first
    drop(parent.second)
""", expected='2\n1\n')

    def test_hook_can_destroy_its_own_field(self):
        self.native(RESOURCE + """
class Parent:
    first: Resource
    second: Resource
    operator drop(move self) -> unit:
        drop(self.first)
        print(0)
test:
    parent = Parent(Resource(1), Resource(2))
""", expected='1\n0\n2\n')

    def test_active_optional_payload(self):
        self.native(RESOURCE + """
def make(present: bool) -> Resource | None:
    if present:
        return Resource(1)
    return None
test:
    first ?= make(true)
    second ?= make(false)
""", expected='1\n')

    def test_returned_field_preserves_sibling_cleanup(self):
        self.native(RESOURCE + """
class Parent:
    first: Resource
    second: Resource
def take() -> Resource:
    parent = Parent(Resource(1), Resource(2))
    return move parent.first
test:
    taken = take()
""", expected='2\n1\n')

    def test_returned_optional_local_transfers_ownership(self):
        self.native(RESOURCE + """
def make() -> Resource | None:
    value = Resource(1)
    return value
test:
    value ?= make()
""", expected='1\n')

    def test_partial_destruction_rejects_invalid_uses(self):
        for body, message in [
            ('drop(parent.first)\n    drop(parent.first)', 'uninitialized field'),
            ('if true:\n        drop(parent.first)', 'matching branch lifetimes'),
        ]:
            self.rejects(RESOURCE + """
class Parent:
    first: Resource
    second: Resource
def use():
    parent = Parent(Resource(1), Resource(2))
    """ + body, message)

    def test_explicit_throw_cleans_locals(self):
        self.native(RESOURCE + """
class Failure: Error:
    code: int
def fail() -> int | Failure:
    local = Resource(1)
    throw Failure(7)
test:
    result ?= fail()
    assert(result is Failure)
""", expected='1\n')

    def test_consuming_parameter_takes_cleanup_responsibility(self):
        self.native(RESOURCE + """
def consume(value: move Resource):
    print(0)
test:
    value = Resource(1)
    consume(move value)
""", expected='0\n1\n')

    def test_invalid_destructor_signature_and_aliases(self):
        self.rejects('''
class Invalid:
    operator drop(self) -> unit:
        pass
''', 'destructor requires')
        self.rejects('''
class Invalid:
    operator drop(move self) -> int:
        return 1
''', 'destructor must')
        self.rejects(RESOURCE + '''
class Parent:
    child: Resource
test:
    resource = Resource(1)
    parent = Parent(resource)
''', 'owned resource fields require move')

if __name__ == '__main__':
    unittest.main()
