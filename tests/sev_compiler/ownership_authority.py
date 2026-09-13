#!/usr/bin/env python3
"""Source diagnostics and lowering must use the ownership library contract."""
import unittest
import os
import textwrap

from migration import COMPILER, MigrationCase, ROOT


class OwnershipAuthority(MigrationCase):
    def test_check_does_not_require_hosted_c_provider_files(self):
        sysroot = self.directory / 'sysroot'
        (sysroot / 'sev_compiler').mkdir(parents=True)
        (sysroot / 'sev_compiler/universal').symlink_to(ROOT / 'sev_compiler/universal', target_is_directory=True)

        def expose_except(original, destination, excluded):
            destination.mkdir()
            for entry in original.iterdir():
                if entry.name == excluded[0]:
                    if len(excluded) > 1:
                        expose_except(entry, destination / entry.name, excluded[1:])
                else:
                    (destination / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())

        expose_except(ROOT / 'library', sysroot / 'library', ['core', 'memory', 'native'])
        self.assertFalse((sysroot / 'library/core/memory/native/memory.c').exists())
        path = self.write('def main():\n    pass\n')
        self.succeeds([COMPILER, 'check', path, '--sysroot', sysroot])

    def test_incomplete_initialization_is_rejected_at_escape(self):
        path = self.write('''
            class Pair:
                name: string
                count: int
            def make() -> Pair:
                item := Pair()
                item.name = "set"
                return item
        ''', 'initialization.sev')
        result = self.invoke([COMPILER, 'check', path, '--sysroot', ROOT])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('E000303', result.stderr)
        self.assertIn('partially initialized', result.stderr)
        self.assertIn('initialization.sev:7:12', result.stderr)

    def test_views_have_no_release_authority(self):
        cases = [
            ('def consume(value: borrow string):\n    drop value\n', '2:5'),
            ('def consume(value: borrow pointer[int]):\n    unsafe:\n        free(value)\n', '3:9'),
        ]
        for text, location in cases:
            with self.subTest(source=text):
                path = self.write(text, 'view-release.sev')
                result = self.invoke([COMPILER, 'check', path, '--sysroot', ROOT])
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('E000303', result.stderr)
                self.assertIn('borrowed parameter or view', result.stderr)
                self.assertIn('view-release.sev:' + location, result.stderr)

    def test_source_memory_and_storage_use_visible_mlir_buffers(self):
        memory = os.path.relpath(ROOT / 'library/core/memory/src/lib.sev', self.directory)
        storage = os.path.relpath(ROOT / 'library/core/storage/src/lib.sev', self.directory)
        text = f'import * from "{memory}" as memory\nimport * from "{storage}" as storage\n' + textwrap.dedent('''
            test:
                values = memory.zeroed_bytes(2)
                values[0] = u8(42)
                grown = memory.resized_bytes(values, 4)
                duplicate = storage.copy(grown)
                duplicate[0] = u8(7)
                assert(values[0] == u8(42))
                assert(grown[0] == u8(42))
                assert(grown[3] == u8(0))
                assert(duplicate[0] == u8(7))
        ''')
        mlir = self.native(text)
        self.assertIn('memref.alloc', mlir)
        self.assertIn('memref.copy', mlir)
        self.assertNotIn('@__sev_storage_new', mlir)
        self.assertNotRegex(mlir, r'(?m)^.*(?:func\.call|llvm\.call).*__sev_memory_zeroed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
