#!/usr/bin/env python3
"""Build, validate, and benchmark the 00-07 example corpus."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
WORK = BENCH / ".work"
DOCS = ROOT / "docs" / "examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="correctness check with one execution and no timing warmup",
    )
    return parser.parse_args()


def inventory() -> list[Path]:
    sources: list[Path] = []
    for number in range(8):
        directories = sorted(DOCS.glob(f"{number:02d}-*"))
        if len(directories) != 1:
            raise RuntimeError(
                f"expected one docs example directory for {number:02d}, got {directories}"
            )
        sources.extend(sorted(directories[0].glob("*.sev")))
    return sources


def run(command: list[str], *, timeout: float = 30) -> tuple[subprocess.CompletedProcess[bytes], float]:
    started = time.perf_counter_ns()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return result, elapsed_ms


def explain_failure(label: str, result: subprocess.CompletedProcess[bytes]) -> str:
    stderr = result.stderr.decode(errors="replace").strip()
    stdout = result.stdout.decode(errors="replace").strip()
    detail = stderr or stdout or f"exit status {result.returncode}"
    return f"{label}: {detail.replace(os.linesep, ' | ')}"


def validate_run(command: list[str], expected: bytes) -> tuple[bool, str]:
    try:
        result, _ = run(command, timeout=5)
    except subprocess.TimeoutExpired:
        return False, "execution timed out"
    if result.returncode != 0:
        return False, explain_failure("execution failed", result)
    if result.stderr:
        return False, f"unexpected stderr: {result.stderr.decode(errors='replace').strip()}"
    if result.stdout != expected:
        return False, f"stdout mismatch: expected {expected!r}, got {result.stdout!r}"
    return True, ""


def measure(command: list[str], expected: bytes, warmup: int, samples: int) -> tuple[float, float]:
    for _ in range(warmup):
        valid, message = validate_run(command, expected)
        if not valid:
            raise RuntimeError(message)
    timings: list[float] = []
    for _ in range(samples):
        result, elapsed_ms = run(command, timeout=5)
        if result.returncode != 0 or result.stderr or result.stdout != expected:
            raise RuntimeError(explain_failure("timed execution failed", result))
        timings.append(elapsed_ms)
    timings.sort()
    percentile_index = min(len(timings) - 1, int(0.95 * len(timings)))
    return statistics.median(timings), timings[percentile_index]


def main() -> int:
    args = parse_args()
    if args.check:
        args.samples = 1
        args.warmup = 0
    if args.samples < 1 or args.warmup < 0:
        raise SystemExit("samples must be positive and warmup must be non-negative")

    WORK.mkdir(parents=True, exist_ok=True)
    build, _ = run(["cargo", "build", "-p", "severian-driver", "--bin", "sev"], timeout=300)
    if build.returncode != 0:
        print(explain_failure("could not build sev", build), file=sys.stderr)
        return 1
    sev = ROOT / "target" / "debug" / "sev"

    rows: list[dict[str, object]] = []
    failures = 0
    print(f"{'example':54} {'language':9} {'compile ms':>11} {'median ms':>11} {'p95 ms':>11} status")
    print("-" * 112)

    for source in inventory():
        relative = source.relative_to(DOCS)
        case = relative.with_suffix("")
        expected_path = source.with_suffix(".stdout")
        rust_source = (BENCH / "rust" / relative).with_suffix(".rs")
        python_source = (BENCH / "python" / relative).with_suffix(".py")
        case_work = WORK / case
        case_work.mkdir(parents=True, exist_ok=True)
        rust_binary = case_work / "rust"
        sev_binary = case_work / "severian"
        python_bytecode = case_work / "python.pyc"

        setup_errors: list[str] = []
        has_main = any(line.startswith("def main(") for line in source.read_text().splitlines())
        if not has_main:
            setup_errors.append(f"missing source main() in {source.relative_to(ROOT)}")
        if not expected_path.is_file():
            setup_errors.append(f"missing {expected_path.relative_to(ROOT)}")
        if not rust_source.is_file():
            setup_errors.append(f"missing {rust_source.relative_to(ROOT)}")
        if not python_source.is_file():
            setup_errors.append(f"missing {python_source.relative_to(ROOT)}")
        expected = expected_path.read_bytes() if expected_path.is_file() else b""

        implementations = [
            ("severian", [str(sev), "compile", str(source), "-o", str(sev_binary)], [str(sev_binary)]),
            ("rust", ["rustc", "-O", str(rust_source), "-o", str(rust_binary)], [str(rust_binary)]),
            (
                "python",
                [
                    sys.executable,
                    "-c",
                    "import py_compile,sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)",
                    str(python_source),
                    str(python_bytecode),
                ],
                [sys.executable, str(python_source)],
            ),
        ]

        for language, compile_command, execute_command in implementations:
            error = "; ".join(setup_errors)
            compile_ms: float | None = None
            median_ms: float | None = None
            p95_ms: float | None = None
            if not error:
                try:
                    compiled, compile_ms = run(compile_command, timeout=60)
                    if compiled.returncode != 0:
                        error = explain_failure("compile failed", compiled)
                    else:
                        valid, error = validate_run(execute_command, expected)
                        if valid:
                            median_ms, p95_ms = measure(
                                execute_command, expected, args.warmup, args.samples
                            )
                except (OSError, subprocess.TimeoutExpired, RuntimeError) as exception:
                    error = str(exception)

            status = "PASS" if not error else f"FAIL {error}"
            if error:
                failures += 1
            def display(value: float | None) -> str:
                return "-" if value is None else f"{value:.3f}"
            print(
                f"{str(case):54} {language:9} {display(compile_ms):>11} "
                f"{display(median_ms):>11} {display(p95_ms):>11} {status}"
            )
            rows.append(
                {
                    "example": str(case),
                    "language": language,
                    "compile_ms": compile_ms,
                    "median_ms": median_ms,
                    "p95_ms": p95_ms,
                    "status": "pass" if not error else "fail",
                    "error": error,
                }
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{len(rows)} implementations checked; {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
