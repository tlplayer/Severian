#!/usr/bin/env python3
"""Build the source compiler, then verify and execute its emitted MLIR."""
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "sev_compiler/bootstrap/target/acceptance"
SEED = Path(os.environ.get("SEVERIAN_BIN", ROOT / "target/debug/sev"))


def tool(variable, default):
    selected = os.environ.get(variable, default)
    resolved = shutil.which(selected)
    if not resolved:
        raise RuntimeError(f"required tool is unavailable: {selected}")
    return resolved


def run(arguments, *, output=None, succeeds=True):
    result = subprocess.run(
        [str(argument) for argument in arguments], cwd=ROOT,
        capture_output=True, text=True, timeout=180,
    )
    if output:
        output.write_text(result.stdout)
    if (result.returncode == 0) != succeeds:
        raise AssertionError(
            f"unexpected exit {result.returncode}: {' '.join(map(str, arguments))}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    opt = tool("SEVERIAN_MLIR_OPT", "mlir-opt-21")
    translate = tool("SEVERIAN_MLIR_TRANSLATE", "mlir-translate-21")
    clang = tool("SEVERIAN_CLANG", "clang-21")
    compiler = ARTIFACTS / "sev-bootstrap-driver"
    run([SEED, "build", "sev_compiler/bootstrap", "--bin", "sev-bootstrap-driver", "-o", compiler])
    cases = {
        "int_add": "left: i32 = 1\nright: i32 = 2\nsum: i32 = left + right\nassert(sum == 3)\n",
        "arithmetic": "left: i32 = 7\nright: i32 = 5\nanswer: i32 = (left + right) * 2 - 4\nassert(answer == 20)\n",
        "negative": "value: i32 = -7\nanswer: i32 = value + 9\nassert(answer == 2)\n",
        "false_assertion": "left: i32 = 1\nright: i32 = 2\nassert(left + right == 4)\n",
        "main": "def add(value: i32) -> i32:\n    return value + 1\ndef main() -> i32:\n    assert(add(41) == 42)\n    return 0\n",
        "main_status": "def main() -> i32:\n    return 7\n",
        "unit_main": "def main():\n    assert(1 == 1)\n",
    }
    inputs = {}
    for name, source in cases.items():
        source_path = ARTIFACTS / f"{name}.sev"
        source_path.write_text(source)
        inputs[name] = (source_path, "build")
    inputs.update({
        "example_math": (ROOT / "docs/examples/05-building/src/math.sev", "build"),
        "example_clamp": (ROOT / "docs/examples/03-testing/01-basics/01-ordinary-and-named.sev", "test"),
        "scalar_functions": (ROOT / "tests/sev_compiler/fixtures/scalar_functions.sev", "test"),
    })
    test_selection = ARTIFACTS / "test_selection.sev"
    test_selection.write_text("def main():\n    return\ntest:\n    assert(false)\n")
    inputs["build_excludes_tests"] = (test_selection, "build")
    inputs["false_test"] = (test_selection, "test")
    test_main = ARTIFACTS / "test_main.sev"
    test_main.write_text("def main():\n    assert(false)\ntest:\n    assert(true)\n")
    inputs["test_excludes_main"] = (test_main, "test")
    for name, (source_path, command) in inputs.items():
        emitted = ARTIFACTS / f"{name}.mlir"
        run([compiler, command, "--emit", "mlir", source_path], output=emitted)
        text = emitted.read_text()
        assert "module {" in text and '"func.return"' in text, "expected executable MLIR"
        if name in {"example_math", "example_clamp", "scalar_functions"}:
            assert "__sev_scalar_" in text, "source functions must survive lowering"
        if name in {"example_clamp", "scalar_functions"}:
            assert '"func.call"' in text and '"scf.if"' in text
        verified = ARTIFACTS / f"{name}.verified.mlir"
        run([opt, "--verify-each", emitted, "-o", verified])
        llvm_mlir = ARTIFACTS / f"{name}.llvm.mlir"
        run([opt, emitted, "--convert-scf-to-cf", "--convert-arith-to-llvm",
             "--convert-cf-to-llvm", "--convert-func-to-llvm",
             "--reconcile-unrealized-casts", "-o", llvm_mlir])
        llvm_ir = ARTIFACTS / f"{name}.ll"
        run([translate, "--mlir-to-llvmir", llvm_mlir], output=llvm_ir)
        executable = ARTIFACTS / name
        run([clang, llvm_ir, "-o", executable])
        result = run([executable], succeeds=name not in {"false_assertion", "false_test", "main_status"})
        if name == "main_status":
            assert result.returncode == 7, "preserve the source main's exit status"
        print(f"PASS: {name} (source -> MLIR -> native)", flush=True)
    rejected = {
        "unknown_name": ("answer: i32 = missing + 1\n", "unknown name missing"),
        "wrong_type": ("answer: i32 = true\n", "boolean cannot initialize an integer"),
        "overflow": ("answer: i8 = 128\n", "integer literal is outside"),
        "unsupported": ("def answer[T](value: T) -> T:\n    return value\n", "outside the scalar bootstrap subset"),
        "malformed": ("answer: i32 =\n", "expected an expression"),
        "missing_return": ("def answer() -> i32:\n    value = 3\n", "without returning a value"),
        "branch_missing_return": ("def answer(value: bool) -> i32:\n    if value:\n        return 1\n", "without returning a value"),
        "wrong_return": ("def answer() -> i32:\n    return true\n", "boolean cannot initialize an integer"),
        "wrong_argument": ("def answer(value: i32) -> i32:\n    return value\nanswer(true)\n", "boolean cannot initialize an integer"),
        "wrong_arity": ("def answer(value: i32) -> i32:\n    return value\nanswer()\n", "wrong argument count"),
        "duplicate_function": ("def answer():\n    return\ndef answer():\n    return\n", "duplicate or reserved function"),
        "parameter_default": ("def answer(value: i32 = 3) -> i32:\n    return value\n", "parameter defaults"),
        "named_argument": ("def answer(value: i32) -> i32:\n    return value\nanswer(value=3)\n", "named arguments"),
        "global_capture": ("value: i32 = 3\ndef answer() -> i32:\n    return value\n", "unknown name value"),
        "unreachable": ("def answer() -> i32:\n    return 1\n    return 2\n", "statement after unconditional return"),
        "wrong_condition": ("def answer() -> i32:\n    if 1:\n        return 1\n    return 0\n", "integer cannot initialize bool"),
        "unit_binding": ("def nothing():\n    return\nvalue = nothing()\n", "unit calls cannot initialize"),
    }
    for name, (source, diagnostic) in rejected.items():
        path = ARTIFACTS / f"{name}.sev"
        path.write_text(source)
        result = run([compiler, "build", "--emit", "mlir", path], succeeds=False)
        assert result.returncode > 0, "a crash is not a diagnostic"
        assert "error:" in result.stderr and "module {" not in result.stdout
        assert diagnostic in result.stderr, result.stderr
        print(f"PASS: {name} rejected with a diagnostic", flush=True)
    print(f"PASS: {len(inputs) + len(rejected)} bootstrap acceptance checks")
    print(f"Bootstrap acceptance artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
