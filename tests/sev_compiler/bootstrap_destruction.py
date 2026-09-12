#!/usr/bin/env python3
"""Execute ownership and destruction contracts in the Rust bootstrap backend."""
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
        print(string(id))
'''


class BootstrapDestruction(unittest.TestCase):
    def native(self, program, expected):
        compiler = os.environ.get("SEVERIAN_NATIVE_RUST_COMPILER", str(ROOT / "package.pkg/release/sev"))
        with tempfile.TemporaryDirectory(prefix="sev-bootstrap-destruction-") as temporary:
            source = Path(temporary) / "main.sev"
            binary = Path(temporary) / "main"
            source.write_text(RESOURCE + program)
            built = subprocess.run([compiler, "build", str(source), "-o", str(binary)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr[:4000])
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)

    def test_custom_before_reverse_fields(self):
        self.native('''
class Parent:
    first: Resource
    second: Resource
    operator drop(move self) -> unit:
        print("0")
def main():
    parent = Parent(Resource(1), Resource(2))
''', '0\n2\n1\n')

    def test_move_and_explicit_destruction(self):
        self.native('''
def main():
    first = Resource(1)
    second = move first
    drop(second)
''', '1\n')

    def test_return_transfers_owned_result(self):
        self.native('''
def make() -> Resource:
    other = Resource(1)
    result = Resource(2)
    return result
def main():
    value = make()
    print("0")
''', '1\n0\n2\n')

    def test_reassignment_and_loop_exits(self):
        self.native('''
def main():
    value := Resource(1)
    value = Resource(2)
    index := 0
    while index < 3:
        item = Resource(index + 3)
        index += 1
        if index == 1:
            continue
        break
''', '1\n3\n4\n2\n')

    def test_implicit_error_propagation(self):
        self.native('''
def fail() -> int | Error:
    throw Error("failure")
def make() -> int | Error:
    value = Resource(1)
    result = fail()
    return result
def main():
    try:
        result = make()
    catch error: Error:
        print(error.message)
''', '1\nfailure\n')

    def test_active_optional_payload(self):
        self.native('''
def make(present: bool) -> Resource | None:
    if present:
        return Resource(1)
    return None
def main():
    present ?= make(true)
    absent ?= make(false)
''', '1\n')

    def test_partial_move_keeps_sibling(self):
        self.native('''
class Parent:
    first: Resource
    second: Resource
def main():
    parent = Parent(Resource(1), Resource(2))
    taken = move parent.first
''', '1\n2\n')


if __name__ == "__main__":
    unittest.main()
