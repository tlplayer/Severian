#!/usr/bin/env python3
"""Execute the prelude box ownership contract with the Rust seed."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RESOURCE = '''
class Resource:
    id: int
    operator drop(move self) -> unit:
        print(id)
'''


class Boxes(unittest.TestCase):
    def check(self, program, expected="", rejection=None):
        compiler = os.environ.get("SEVERIAN_NATIVE_RUST_COMPILER", str(ROOT / "package.pkg/release/sev"))
        with tempfile.TemporaryDirectory(prefix="sev-box-") as temporary:
            directory = Path(temporary)
            (directory / "package.json").write_text(json.dumps({
                "package": {"name": "box-test", "version": "0.1.0", "edition": "2026"},
                "bin": [{"name": "main", "path": "main.sev"}],
            }))
            (directory / "main.sev").write_text(program)
            binary = directory / "main"
            built = subprocess.run([compiler, "build", str(directory), "-o", str(binary)],
                                   cwd=ROOT, capture_output=True, text=True, timeout=60)
            if rejection:
                self.assertNotEqual(built.returncode, 0)
                self.assertIn(rejection, built.stderr)
                return
            self.assertEqual(built.returncode, 0, built.stderr[:6000])
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)

    def test_scalar_clones_have_independent_storage(self):
        self.check('''
def main():
    first = box(42)
    second = clone(first)
    assert(second.replace(9) == 42)
    assert(first.take() == 42)
    assert(second.take() == 9)
    flag = box[bool](true)
    assert(flag.take())
    real = box[f64](2.5)
    assert(real.take() == 2.5)
''')

    def test_payload_dropped_once_after_construction(self):
        self.check(RESOURCE + '''
def main():
    value = box[Resource](Resource(1))
    print("alive")
''', "alive\n1\n")

    def test_replace_and_take_transfer_payloads(self):
        self.check(RESOURCE + '''
def main():
    value = box[Resource](Resource(1))
    previous = value.replace(Resource(2))
    result = value.take()
    print("taken")
''', "taken\n2\n1\n")

    def test_nested_boxes_destroy_inner_payload(self):
        self.check(RESOURCE + '''
def main():
    value = box[box[Resource]](box[Resource](Resource(3)))
    print("nested")
''', "nested\n3\n")

    def test_record_clone_calls_payload_clone(self):
        self.check('''
class Resource:
    id: int
    def clone() -> Resource:
        return Resource(id + 1)
    operator drop(move self) -> unit:
        print(id)
def main():
    first = box[Resource](Resource(4))
    second = first.clone()
    print("cloned")
''', "cloned\n5\n4\n")

    def test_large_record_layout(self):
        self.check('''
class Wide:
    first: i128
    second: i128
    third: int
def main():
    value = box[Wide](Wide(i128(123), i128(456), 789))
    result = value.take()
    assert(result.first == i128(123))
    assert(result.second == i128(456))
    assert(result.third == 789)
''')

    def test_string_payload_clone(self):
        self.check('''
def main():
    first = box[string]("hello")
    second = first.clone()
    assert(first.take() == "hello")
    assert(second.take() == "hello")
''')

    def test_cleanup_when_error_propagates(self):
        self.check(RESOURCE + '''
def fail() -> int | Error:
    value = box[Resource](Resource(7))
    throw Error("failed")
def main():
    try:
        result = fail()
    catch error: Error:
        print(error.message)
''', "7\nfailed\n")

    def test_explicit_drop_and_move(self):
        self.check(RESOURCE + '''
def main():
    first = box[Resource](Resource(6))
    second = move first
    drop(second)
    print("dropped")
''', "6\ndropped\n")

    def test_take_prevents_reuse(self):
        self.check('''
def main():
    value = box[int](1)
    first = value.take()
    second = value.take()
''', rejection="moved")

    def test_noncloneable_owner_is_rejected(self):
        self.check(RESOURCE + '''
def main():
    first = box[Resource](Resource(1))
    second = first.clone()
''', rejection="no overload accepting")

    def test_storage_is_private(self):
        self.check('''
def main():
    value = box[int](1)
    address = value.__storage
''', rejection="is private to class")

    def test_borrowed_box_cannot_be_consumed(self):
        self.check('''
def main():
    value = box[int](1)
    shared = borrow value
    result = shared.take()
''', rejection="view")


if __name__ == "__main__":
    unittest.main()
