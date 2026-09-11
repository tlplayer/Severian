#!/usr/bin/env python3
"""Native and rejection gates for source generic and prelude resolution."""
import os
import tempfile
from pathlib import Path
from bootstrap_mlir import ROOT, SEED, run
from constructors import constructor_gates


def main():
    if not os.environ.get("SEVERIAN_SKIP_BUILD"):
        run([SEED, "build"], cwd=ROOT / "sev_compiler")
    compiler = ROOT / "sev_compiler/package.pkg/host/dev/bin/sev_compiler"
    with tempfile.TemporaryDirectory(prefix="severian-generics-") as temporary:
        directory = Path(temporary)
        # Run outside the checkout: the actual prelude must come from --sysroot.
        hello = directory / "hello"
        run([compiler, "build", ROOT / "docs/examples/00-getting-started/01-hello.sev",
             "--sysroot", ROOT, "-o", hello], cwd=directory)
        assert run([hello], cwd=directory).stdout == "hello, severian\n"
        print("PASS: hello-world with checkout prelude (native)", flush=True)
        layout = directory / "aggregate_layout"
        run([SEED, "build", ROOT / "tests/sev_compiler/fixtures/primitives/aggregate_layout.sev", "-o", layout])
        run([layout], cwd=directory)
        print("PASS: wide aggregate list storage (seed native)", flush=True)
        array_source = ROOT / "sev_compiler/universal/primitive/array.sev"
        constructor = directory / "constructor_dimension.sev"
        constructor.write_text(f'import * from "{array_source}"\n' + """class First[T]:
    value: T
    def First[N: usize](values: array[T, N]):
        value = values[0]

def main():
    values = array[i32, 3]([1, 2, 3])
    first = First[i32](values)
    assert(first.value == 1)
""")
        executable = directory / "constructor_dimension"
        run([SEED, "build", constructor, "-o", executable], cwd=ROOT)
        run([executable], cwd=directory)
        print("PASS: constructor dimension inference (seed native)", flush=True)
        collection = directory / "scalar_list.sev"
        collection.write_text(f'import * from "{ROOT / "sev_compiler/universal/collections/list.sev"}"\n' + """def main():
    values := list[i32]()
    values.append(42)
    values.append(7)
    assert(values.len() == 2)
    assert(values[0] == 42)
    assert(values[1] == 7)
""")
        executable = directory / "scalar_list"
        run([SEED, "build", collection, "-o", executable])
        run([executable], cwd=directory)
        print("PASS: canonical scalar list (seed native)", flush=True)
        constructor_gates(directory)
        accepted = {
            "generic_owned_field": "class Box[T]:\n    value: T\nBox[string](\"owned\")\n",
            "direct_macro": """-> identity[T: int]():
    def identity(value: T) -> T:
        return value
    test:
        assert(identity(T(7)) == T(7))
identity[int]()
""",
            "same_family": """-> pair[S: int, T: int]():
    def first(left: S, right: T) -> S:
        return left
    test:
        assert(first(S(7), T(9)) == S(7))
pair[int, int]()
""",
            "generic_record": """class Box[T]:
    value: T
    def get() -> T:
        return value
    def same() -> Self:
        return self

type IntBox = Box[int]
box = Box[int](42)
other: IntBox = IntBox(7)
assert(box.get() == 42)
assert(other.same().value == 7)
assert(Box[float](2.5).get() == 2.5)
class Widget:
    number: int
assert(Box[Widget](Widget(9)).get().number == 9)
""",
            "multi_parameter_record": """class Pair[Left, Right]:
    left: Left
    right: Right

type Mixed = Pair[int, float]

def first[L, R](value: Pair[L, R]) -> L:
    return value.left

mixed = Mixed(2, 1.5)
assert(first(mixed) == 2)
assert(mixed.right == 1.5)
""",
            "compiler_cache": """class Box[T]:
    value: T

test with compiler:
    reject:
        broken = Box[string]("owned")
    reject:
        broken = Box[string]()
    accept:
        valid = Box[int](42)
""",
            "nested_record": """class Box[T]:
    value: T

def unwrap[T](box: Box[T]) -> T:
    return box.value

def identity[T](value: T) -> T:
    return value

def forward[T](box: Box[T]) -> Box[T]:
    return identity(box)

nested = Box[Box[int]](Box[int](42))
assert(unwrap(unwrap(forward(nested))) == 42)
assert(unwrap(Box[float](2.5)) == 2.5)
""",
        }
        for name, source in accepted.items():
            subject = directory / (name + ".sev")
            subject.write_text(source)
            executable = directory / name
            run([compiler, "check", subject, "--sysroot", ROOT], cwd=directory)
            if name in {"direct_macro", "same_family", "compiler_cache"}:
                run([compiler, "test", subject, "--sysroot", ROOT], cwd=directory)
            else:
                run([compiler, "build", subject, "--sysroot", ROOT, "-o", executable], cwd=directory)
                run([executable], cwd=directory)
            print(f"PASS: {name} (native)", flush=True)
        rejected = {
            "record_implementation": ("trait Counted:\n    def size() -> int\nclass Box[T]: Counted\n    value: T\nBox[int](42)\n", "class does not satisfy trait Counted"),
            "record_constraint": ("trait Measured:\n    property size: int\nclass Box[T: Measured]:\n    value: T\nBox[int](42)\n", "type does not satisfy trait Measured"),
            "duplicate_binding": ("-> bad[T: int, T: float]():\n    def value(input: T) -> T:\n        return input\n", "duplicate macro type binding"),
            "unknown_family": ("-> bad[T: Missing]():\n    def value(input: T) -> T:\n        return input\n", "unsupported scalar macro family Missing"),
            "generic_conflict": ("class Box[T]:\n    value: T\ndef same[T](left: Box[T], right: Box[T]) -> T:\n    return left.value\nsame(Box[int](1), Box[float](2.0))\n", "conflicting generic type arguments"),
            "record_arity": ("class Box[T]:\n    value: T\nvalue: Box = 1\n", "missing type argument T"),
            "generic_recursion": ("class Box[T]:\n    value: Box[T]\nvalue: Box[int] = Box[int]()\n", "recursive value record requires indirection"),
        }
        for name, (source, diagnostic) in rejected.items():
            subject = directory / (name + ".sev")
            subject.write_text(source)
            result = run([compiler, "build", "--emit", "mlir", subject, "--sysroot", ROOT], succeeds=False, cwd=directory)
            assert diagnostic in result.stderr, result.stderr
            checked = run([compiler, "check", subject, "--sysroot", ROOT], succeeds=False, cwd=directory)
            assert diagnostic in checked.stderr, checked.stderr
            print(f"PASS: {name} diagnostic", flush=True)


if __name__ == "__main__":
    main()
