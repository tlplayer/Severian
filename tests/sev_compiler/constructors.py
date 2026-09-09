#!/usr/bin/env python3
"""Seed constructor execution, overload selection, and initialization gates."""
import tempfile
from pathlib import Path
from bootstrap_mlir import ROOT, SEED, run


def constructor_gates(directory):
    accepted = {
        "constructor_body": ('''class Counter:
    value: i32
    calls: i32 = 0
    def Counter(input: i32):
        local := input
        value := 0
        while local > 0:
            value += local
            local -= 1
        if input == 4:
            value = value + 1
        else:
            value = value + 2
        bump()
        assert(calls == 1)
    def bump():
        calls += 1

def argument() -> i32:
    print("argument")
    return 4

def main():
    counter = Counter(argument())
    assert(counter.value == 11)
    assert(counter.calls == 1)
''', "argument\n"),
        "constructor_branches": ('''class Branch:
    value: i32
    def Branch(input: bool):
        if input:
            self.value = 7
            return
        else:
            self.value = 9
        value += 1

def main():
    assert(Branch(true).value == 7)
    assert(Branch(false).value == 10)
''', ""),
        "constructor_overloads": ('''class Choice:
    value: i32
    def Choice(input: f64):
        value = 2
    def Choice(input: i32):
        value = 1
    def Choice(input: bool):
        value = 3

def main():
    assert(Choice(i32(7)).value == 1)
    assert(Choice(7.5).value == 2)
    assert(Choice(true).value == 3)
''', ""),
        "constructor_named": ('''class Pair:
    left: i32
    right: i32
    def Pair(a: i32, b: i32 = 9):
        left = a
        right = b

def first() -> i32:
    print("first")
    return 1

def second() -> i32:
    print("second")
    return 2

def main():
    pair = Pair(b=second(), a=first())
    assert(pair.left == 1)
    assert(pair.right == 2)
    assert(Pair(3).right == 9)
''', "second\nfirst\n"),
        "constructor_generic": ('''class Select[T]:
    value: T
    def Select[U](input: U, output: T):
        ignored = input
        value = output

class Specific:
    value: i32
    def Specific[T](input: T):
        value = 1
    def Specific(input: i32):
        value = 2

def main():
    assert(Select[i32](output=i32(8), input=3.5).value == 8)
    assert(Select[i32](true, i32(4)).value == 4)
    assert(Specific(i32(0)).value == 2)
    assert(Specific(true).value == 1)
''', ""),
        "constructor_recursion": ('''class Factorial:
    value: i32
    def Factorial(input: i32):
        if input == 0:
            value = 1
        else:
            previous = Factorial(input - 1)
            value = input * previous.value

def main():
    assert(Factorial(5).value == 120)
''', ""),
        "constructor_integer_bounds": ('''def bound(value: usize) -> usize:
    print("bound")
    return value

class Bounds:
    low: usize
    high: usize
    def Bounds():
        low = min(bound(usize(3)), bound(usize(4)))
        high = max(usize(-1, lossy), usize(4))

def main():
    result = Bounds()
    assert(result.low == 3)
    assert(result.high == usize(-1, lossy))
''', "bound\nbound\n"),
        "constructor_list_inputs": (f'import "{ROOT / "sev_compiler/universal/collections/list.sev"}"\n' + '''def main():
    input = array[i32, 3]([1, 2, 3])
    values := list[i32](input)
    assert(values.len() == 3)
    assert(values.cap() == 4)
    assert(values[0] == 1)
    assert(values[2] == 3)
    values.append(4)
    values.append(5)
    assert(values.len() == 5)
    assert(values[0] == 1)
    assert(values[4] == 5)
    view = input[1:3]
    copied := list[i32](view)
    assert(copied.len() == 2)
    assert(copied[0] == 2)
    assert(copied[1] == 3)
    empty = list[i32](array[i32, 0]())
    assert(empty.len() == 0)
    bigger = list[i32](array[i32, 6]([1, 2, 3, 4, 5, 6]))
    assert(bigger.len() == 6)
    assert(bigger[5] == 6)
''', ""),
    }
    for name, (source, expected) in accepted.items():
        subject = directory / (name + ".sev")
        subject.write_text(source)
        executable = directory / name
        run([SEED, "build", subject, "-o", executable])
        assert run([executable], cwd=directory).stdout == expected, name
        print(f"PASS: {name} (seed native)", flush=True)

    assertion = directory / "constructor_assertion.sev"
    assertion.write_text('''class Checked:
    value: i32
    def Checked(input: i32):
        assert(input != 0)
        value = input

def main():
    checked = Checked(0)
''')
    executable = directory / "constructor_assertion"
    run([SEED, "build", assertion, "-o", executable])
    result = run([executable], succeeds=False, cwd=directory)
    assert "assertion failed" in result.stderr
    print("PASS: constructor assertion executes (seed native)", flush=True)

    rejected = {
        "missing": ("value = 1", "other: i32", "uninitialized field `other`"),
        "read_before_write": ("value = value + 1", "", "uninitialized field `value`"),
        "branch_missing": ("if input:\n            value = 1", "", "uninitialized field `value`"),
        "loop_missing": ("while input:\n            value = 1\n            break", "", "uninitialized field `value`"),
        "early_return": ("if input:\n            return\n        value = 1", "", "uninitialized field `value`"),
        "escape": ("copy = self\n        value = 1", "", "uninitialized field `value`"),
    }
    for name, (body, extra, diagnostic) in rejected.items():
        subject = directory / ("constructor_reject_" + name + ".sev")
        subject.write_text(f"class Checked:\n    value: i32\n    {extra}\n    def Checked(input: bool):\n        {body}\n\ndef main():\n    checked = Checked(true)\n")
        result = run([SEED, "check", subject], succeeds=False)
        assert diagnostic in result.stderr, result.stderr
        print(f"PASS: constructor {name} diagnostic", flush=True)

    for name, source, diagnostic in [
        ("ambiguous", "class Ambiguous:\n    value: i32\n    def Ambiguous(a: i32):\n        value = 1\n    def Ambiguous(b: i32):\n        value = 2\ndef main():\n    x = Ambiguous(i32(3))\n", "ambiguous constructor call"),
        ("wrong_result", "class Checked:\n    value: i32\n    def Checked() -> i32:\n        value = 1\ndef main():\n    x = Checked()\n", "field-initializing constructor must return its class"),
        ("no_match", "class Checked:\n    value: i32\n    def Checked(input: i32):\n        value = input\ndef main():\n    x = Checked(\"wrong\")\n", "no overload accepting"),
    ]:
        subject = directory / ("constructor_" + name + ".sev")
        subject.write_text(source)
        result = run([SEED, "check", subject], succeeds=False)
        assert diagnostic in result.stderr, result.stderr
        print(f"PASS: constructor {name} diagnostic", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="severian-constructors-") as temporary:
        constructor_gates(Path(temporary))


if __name__ == "__main__":
    main()
