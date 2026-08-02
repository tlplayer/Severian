#!/usr/bin/env python3
"""Compare automatic activation-chain fusion with materialized intermediates."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
WORK = ROOT / "bench/.work/activation-fusion"
EXPECTED = (HERE / "expected.stdout").read_bytes()


def timed(command):
    started = time.perf_counter_ns()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
    return result, (time.perf_counter_ns() - started) / 1_000_000


def checked(command):
    result, elapsed = timed(command)
    if result.returncode != 0 or result.stderr or result.stdout != EXPECTED:
        raise RuntimeError(
            f"{command[0]} failed: status={result.returncode}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    return elapsed


def p95(samples):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup non-negative")

    WORK.mkdir(parents=True, exist_ok=True)
    sev = ROOT / "target/debug/sev"
    subprocess.run(
        ["cargo", "build", "-p", "severian-driver", "--bin", "sev"],
        cwd=ROOT,
        check=True,
    )
    commands = {}
    compile_times = {}
    for name, source in (
        ("automatic", HERE / "fused.sev"),
        ("materialized", HERE / "materialized.sev"),
    ):
        binary = WORK / name
        result, elapsed = timed([str(sev), "compile", str(source), "-o", str(binary)])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode())
        commands[name] = [str(binary)]
        compile_times[name] = elapsed

    timings = {}
    for name, command in commands.items():
        for _ in range(args.warmup):
            checked(command)
        timings[name] = [checked(command) for _ in range(args.samples)]

    print("262,144-value Relu -> FastTanh -> Swish pipeline")
    print("form          compile ms  median process ms  p95 process ms")
    for name in commands:
        print(
            f"{name:12}  {compile_times[name]:10.3f}  "
            f"{statistics.median(timings[name]):17.3f}  {p95(timings[name]):14.3f}"
        )
    ratio = statistics.median(timings["materialized"]) / statistics.median(
        timings["automatic"]
    )
    print(f"automatic fusion speedup: {ratio:.3f}x")
    print("Both executables produced the exact checked stdout fixture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
