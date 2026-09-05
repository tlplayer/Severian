#!/usr/bin/env python3
"""Exercise callable bodies through the compiler written in Severian."""
import os
from bootstrap_mlir import ROOT, ARTIFACTS, SEED, run, tool


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("SEVERIAN_SKIP_BUILD"):
        run([SEED, "build"], cwd=ROOT / "sev_compiler")
    compiler = ROOT / "sev_compiler/target/host/dev/bin/sev_compiler"
    subjects = ROOT / "sev_compiler/frontend/semantic/src/callable/tests"
    for subject in sorted(subjects.glob("*.sev")):
        if os.environ.get("SEVERIAN_CASE") and subject.stem != os.environ["SEVERIAN_CASE"]:
            continue
        output = ARTIFACTS / ("callable_" + subject.stem)
        emitted = output.with_suffix(".mlir")
        run([compiler, "test", "--emit", "mlir", subject, "--sysroot", ROOT], output=emitted)
        run([tool("SEVERIAN_MLIR_OPT", "mlir-opt-21"), "--verify-each", emitted, "-o", output.with_suffix(".verified.mlir")])
        lowered = output.with_suffix(".llvm.mlir")
        run([tool("SEVERIAN_MLIR_OPT", "mlir-opt-21"), emitted,
             "--buffer-deallocation-pipeline=private-function-dynamic-ownership",
             "--convert-bufferization-to-memref", "--convert-scf-to-cf", "--convert-arith-to-llvm",
             "--convert-cf-to-llvm", "--finalize-memref-to-llvm", "--convert-func-to-llvm",
             "--reconcile-unrealized-casts", "-o", lowered])
        llvm = output.with_suffix(".ll")
        run([tool("SEVERIAN_MLIR_TRANSLATE", "mlir-translate-21"), "--mlir-to-llvmir", lowered], output=llvm)
        run([tool("SEVERIAN_CLANG", "clang-21"), llvm, "-o", output])
        actual = run([output])
        expected = {"receivers": "r21r3receivers complete\n"}.get(subject.stem, subject.stem + " complete\n")
        assert actual.stdout == expected, repr(actual.stdout)
        if os.environ.get("SEVERIAN_SANITIZE") == "1":
            sanitized = output.with_name(output.name + "_asan")
            run([tool("SEVERIAN_CLANG", "clang-21"), "-fsanitize=address", llvm, "-o", sanitized])
            checked = run([sanitized])
            assert checked.stdout == expected, repr(checked.stdout)
        print(f"PASS: {subject.stem} (Severian compiler -> MLIR -> native)", flush=True)

    if not os.environ.get("SEVERIAN_CASE"):
        for name, diagnostic in {
            "owned_fields": "record string fields require aggregate ownership lowering",
            "recursive_records": "recursive value record requires indirection",
        }.items():
            rejected = run([compiler, "build", "--emit", "mlir", subjects / "reject" / (name + ".sev"), "--sysroot", ROOT], succeeds=False)
            assert diagnostic in rejected.stderr, rejected.stderr
            print(f"PASS: {name} diagnostic", flush=True)


if __name__ == "__main__":
    main()
