#!/usr/bin/env python3
"""Compare the large Severian and Python forward/backward programs."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SEVERIAN_SOURCE = ROOT / "docs/examples/19-distributed-learning/main.sev"
PYTHON_SOURCE = Path(__file__).with_name("python.py")
EXPECTED = SEVERIAN_SOURCE.with_suffix(".stdout").read_bytes()
WORK = ROOT / "bench/.work/distributed-learning"


def timed(command: list[str], timeout: int = 60):
    started = time.perf_counter_ns()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout)
    return result, (time.perf_counter_ns() - started) / 1_000_000


def validated(command: list[str]):
    result, elapsed = timed(command)
    if result.returncode != 0 or result.stderr or result.stdout != EXPECTED:
        raise RuntimeError(
            f"{command[0]} failed: status={result.returncode}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    return elapsed


def source_lines(path: Path) -> int:
    return sum(
        1
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup must be non-negative")

    WORK.mkdir(parents=True, exist_ok=True)
    sev = ROOT / "target/debug/sev"
    binary = WORK / "severian"
    build, _ = timed(["cargo", "build", "-p", "severian-driver", "--bin", "sev"])
    if build.returncode != 0:
        sys.stderr.buffer.write(build.stderr)
        return 1

    severian_compile, severian_compile_ms = timed(
        [str(sev), "compile", str(SEVERIAN_SOURCE), "-o", str(binary)]
    )
    if severian_compile.returncode != 0:
        sys.stderr.buffer.write(severian_compile.stderr)
        return 1
    python_compile, python_compile_ms = timed(
        [
            sys.executable,
            "-c",
            "import py_compile,sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)",
            str(PYTHON_SOURCE),
            str(WORK / "python.pyc"),
        ]
    )
    if python_compile.returncode != 0:
        sys.stderr.buffer.write(python_compile.stderr)
        return 1

    commands = {
        "Severian": [str(binary)],
        "Python": [sys.executable, str(PYTHON_SOURCE)],
    }
    results = {}
    for language, command in commands.items():
        for _ in range(args.warmup):
            validated(command)
        samples = [validated(command) for _ in range(args.samples)]
        samples.sort()
        results[language] = (
            statistics.median(samples),
            samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        )

    print("65,536-value, four-worker ReLU forward/backward")
    print("language  compile ms  median run ms  p95 run ms  entry LOC")
    print(
        f"Severian  {severian_compile_ms:10.3f}  {results['Severian'][0]:13.3f}  "
        f"{results['Severian'][1]:10.3f}  {source_lines(SEVERIAN_SOURCE):9d}"
    )
    print(
        f"Python    {python_compile_ms:10.3f}  {results['Python'][0]:13.3f}  "
        f"{results['Python'][1]:10.3f}  {source_lines(PYTHON_SOURCE):9d}"
    )
    print("Both implementations produced the exact checked stdout fixture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
