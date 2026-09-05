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
    }
    for name, source in cases.items():
        source_path = ARTIFACTS / f"{name}.sev"
        source_path.write_text(source)
        emitted = ARTIFACTS / f"{name}.mlir"
        run([compiler, "build", "--emit", "mlir", source_path], output=emitted)
        text = emitted.read_text()
        assert "module {" in text and '"arith.addi"' in text, "expected actual arithmetic MLIR"
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
        run([executable], succeeds=name != "false_assertion")
        print(f"PASS: {name} (source -> MLIR -> native)", flush=True)
    rejected = {
        "unknown_name": ("answer: i32 = missing + 1\n", "unknown name missing"),
        "wrong_type": ("answer: i32 = true\n", "boolean cannot initialize an integer"),
        "overflow": ("answer: i8 = 128\n", "integer literal is outside"),
        "unsupported": ("def answer() -> i32:\n    return 3\n", "outside the scalar bootstrap subset"),
        "malformed": ("answer: i32 =\n", "expected an expression"),
    }
    for name, (source, diagnostic) in rejected.items():
        path = ARTIFACTS / f"{name}.sev"
        path.write_text(source)
        result = run([compiler, "build", "--emit", "mlir", path], succeeds=False)
        assert result.returncode > 0, "a crash is not a diagnostic"
        assert "error:" in result.stderr and "module {" not in result.stdout
        assert diagnostic in result.stderr, result.stderr
        print(f"PASS: {name} rejected with a diagnostic", flush=True)
    print(f"Bootstrap acceptance artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
