#!/usr/bin/env python3
"""Native resource cleanup and compile-time lifetime rejection regressions."""
import unittest
import owned_records

from migration import COMPILER, MigrationCase, ROOT


RESOURCE = '''
class Resource:
    id: int
    def drop():
        print("released:", id)
'''


class ResourceLifetimes(MigrationCase):
    def test_allocated_resource_fields_are_released(self):
        self.native('''
class Text:
    value: string
    def drop():
        print(value)
test:
    first = Text("first" + "!")
    second = Text("second" + "!")
    drop(first)
''', expected='first!\nsecond!\n')
        owned_records.OwnedRecords.check_allocations(self)

    def test_missing_source_and_import_do_not_compile_as_empty(self):
        missing = self.directory / 'missing.sev'
        result = self.invoke([COMPILER, 'build', missing, '--sysroot', ROOT,
                              '-o', self.directory / 'missing'])
        self.assertGreater(result.returncode, 0)
        self.assertIn('source file does not exist', result.stderr)
        result = self.invoke([COMPILER, 'check', self.write('import * from "missing.sev"'),
                              '--sysroot', ROOT])
        self.assertGreater(result.returncode, 0)
        self.assertIn('source file does not exist', result.stderr)
        self.native('')

    def test_documented_layout(self):
        path = ROOT / 'docs/examples/01-types/04-memory/02-memory-layout.sev'
        self.native(path.read_text(), expected='8B\n4B\n')
        self.native('''
class Padded:
    head: u8
    number: i64
    tail: u8
test:
    assert(bytes[Padded]() == 24B)
    assert(alignment[Padded]() == 8B)
    assert(bytes[bool]() == 1B)
    assert(bytes[list[int]]() == 40B)
    assert(string(1b) == "0.125B")
''')

    def test_documented_lifetimes(self):
        path = ROOT / 'docs/examples/01-types/04-memory/03-resource-lifetime.sev'
        executable = self.directory / 'resource'
        self.succeeds([COMPILER, 'build', path, '--sysroot', ROOT, '-o', executable])
        self.assertEqual(self.succeeds([executable]),
                         'open: database\ndatabase\nclose: database\nfinished\n')
        self.native(path.read_text(), expected='open: temporary\nclose: temporary\n')
        path = ROOT / 'docs/examples/06-ownership/05-drop.sev'
        self.succeeds([COMPILER, 'build', path, '--sysroot', ROOT, '-o', executable])
        self.assertEqual(self.succeeds([executable]), '42\nreleased: 42\n')
        self.native(path.read_text())

    def test_reverse_order_early_return_and_explicit_drop(self):
        self.native(RESOURCE + '''
def use(early: bool):
    first = Resource(1)
    second = Resource(2)
    if early:
        return
test:
    use(true)
    use(false)
    third = Resource(3)
    drop(third)
    fourth = Resource(4)
    fourth.drop()
    reject:
        print(third.id)
    reject:
        drop(fourth)
''', expected='released: 2\nreleased: 1\nreleased: 2\nreleased: 1\nreleased: 3\nreleased: 4\n')

    def test_branch_local_cleanup_does_not_poison_later_bindings(self):
        self.native(RESOURCE + '''
test:
    if true:
        inner = Resource(5)
    later = Resource(6)
    print(later.id)
''', expected='released: 5\n6\nreleased: 6\n')

    def test_invalid_resource_uses_have_lifetime_diagnostics(self):
        self.rejects(RESOURCE + '''
def optional_resource() -> Resource | None:
    return Resource(1)
''', 'nested resource fields require ownership transfer support')
        for body, diagnostic in [
            ('value = Resource(1)\n    drop(value)\n    print(value.id)', 'use after drop'),
            ('value = Resource(1)\n    drop(value)\n    drop(value)', 'use after drop'),
            ('value = Resource(1)\n    value.drop(1)', 'drop takes no arguments'),
            ('value = Resource(1)\n    alias = value', 'resource aliases'),
            ('print(type(Resource(1)))', 'resource temporary requires a local binding'),
            ('value = type(Resource(1))', 'resource temporary requires a local binding'),
        ]:
            with self.subTest(body=body):
                self.rejects(RESOURCE + '\ndef use():\n    ' + body, diagnostic)

    def test_expectations_are_checked_and_do_not_execute(self):
        self.native('''
test:
    value = 42
    accept:
        print(value)
    reject:
        unknown()
    assert(value == 42)
''')
        result = self.invoke([COMPILER, 'test', self.write('''
test:
    reject:
        print(42)
'''), '--emit', 'mlir', '--sysroot', ROOT])
        self.assertGreater(result.returncode, 0)
        self.assertIn('compiler test expectation failed', result.stderr)
        self.rejects('''
def use():
    reject:
        unknown()
''', 'compiler expectations require a test body')


if __name__ == '__main__':
    unittest.main()
