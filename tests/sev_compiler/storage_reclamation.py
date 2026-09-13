#!/usr/bin/env python3
"""Executable storage ownership contracts, including live-byte accounting."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMPILER = os.environ.get('SEVERIAN_NATIVE_RUST_COMPILER', str(ROOT / 'package.pkg/release/sev'))


class StorageReclamation(unittest.TestCase):
    def native(self, body):
        with tempfile.TemporaryDirectory(prefix='sev-storage-') as temporary:
            source = Path(temporary) / 'main.sev'
            binary = Path(temporary) / 'main'
            source.write_text('''
@c(symbol = "__sev_storage_live_bytes")
def live_bytes() -> int
class Item:
    text: string
def identity(value: string) -> string:
    return value
''' + body + '''
def main():
    baseline = live_bytes()
    work()
    assert(live_bytes() == baseline, "storage survived its owning scope")
''')
            result = subprocess.run([COMPILER, 'build', str(source), '-o', str(binary)], cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_strings_records_and_lists(self):
        self.native('''
def work():
    text = string(42)
    other = identity(text)
    values: list[string] = [text, other]
    alias = values
    values.append(string(43))
    assert(alias[0] == "42")
    item = Item(string(44))
    item_copy = item
    assert(item_copy.text == "44")
''')

    def test_boxed_record_survives_collection_clear(self):
        self.native('''
def work():
    items: list[Item] = [Item(string(42)), Item(string(43))]
    item = items[0]
    assert(item.text == "42")
    items.clear()
    assert(item.text == "42")
''')

    def test_shared_sum_payload_has_one_owner_for_mutable_contents(self):
        self.native('''
enum Wrapped:
    Present(item: Item)
    Empty

def replace(item: Item) -> Item:
    item.text = string(43)
    return item

def rename(value: Wrapped):
    match value:
        case Present:
            renamed = replace(item)
            assert(renamed.text == "43")
        case _:
            pass

def work():
    values = [Wrapped.Present(Item(string(42))), Wrapped.Empty]
    selected = values[0]
    rename(selected)
    values.clear()
    match selected:
        case Present:
            assert(item.text == "43")
        case _:
            assert(false)
''')

    def test_runtime_recursive_storage_and_alias_replacement(self):
        with tempfile.TemporaryDirectory(prefix='sev-storage-asan-') as temporary:
            binary = Path(temporary) / 'storage'
            native = ROOT / 'rust_compiler/runtime/native'
            subprocess.run([os.environ.get('CC', 'cc'), '-std=gnu11', '-Wall', '-Wextra', '-Werror',
                            '-g', '-fsanitize=address,undefined', '-I', str(native),
                            str(ROOT / 'tests/sev_compiler/storage_runtime.c'),
                            *(str(native / name) for name in ['owned.c', 'list.c', 'string.c', 'any.c']),
                            '-lm', '-lquadmath', '-o', str(binary)], check=True)
            # LSan cannot inspect threads under the host tracer. The C fixture
            # independently checks managed live bytes after every iteration.
            subprocess.run([str(binary)], env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'}, check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
