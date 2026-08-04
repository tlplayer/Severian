#!/usr/bin/env python3
"""Validate ROCm lowering and compare the current CPU baseline with PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/examples/25-transformer-rocm/main.sev"
EXPECTED = b"[1.75, 0]\n"


def invoke(command: list[str], timeout: int = 120) -> tuple[subprocess.CompletedProcess[bytes], float]:
    started = time.perf_counter_ns()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout)
    return result, (time.perf_counter_ns() - started) / 1_000_000


def checked(command: list[str], expected: bytes | None = None) -> float:
    result, elapsed = invoke(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or "command failed")
    if expected is not None and result.stdout != expected:
        raise RuntimeError(f"expected {expected!r}, got {result.stdout!r}")
    return elapsed


def median(command: list[str], samples: int) -> float:
    checked(command, EXPECTED)
    return statistics.median(checked(command, EXPECTED) for _ in range(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--chip", default="gfx1100")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    work = ROOT / "bench/.work/transformer-rocm"
    work.mkdir(parents=True, exist_ok=True)
    sev = ROOT / "target/debug/sev"
    native = work / "severian"
    rocdl = work / f"transformer-{args.chip}.mlir"

    checked(["cargo", "build", "-q", "-p", "severian-driver", "--bin", "sev"])
    checked([str(sev), "compile", str(SOURCE), "-o", str(native)])
    target, _ = invoke(
        [str(sev), "emit-mlir", str(SOURCE), "--target", "rocm", "--chip", args.chip]
    )
    if target.returncode != 0:
        raise RuntimeError(target.stderr.decode(errors="replace").strip())
    if b"rocdl.target" not in target.stdout or b"gpu.launch_func" not in target.stdout:
        raise RuntimeError("target output contains no outlined ROCDL kernels")
    rocdl.write_bytes(target.stdout)

    torch_command = [sys.executable, str(HERE / "torch_baseline.py")]
    severian_ms = median([str(native)], args.samples)
    try:
        pytorch_ms = median(torch_command, args.samples)
    except RuntimeError as error:
        if "No module named 'torch'" in str(error):
            raise RuntimeError(
                "PyTorch is not installed in this Python environment; install a ROCm/CPU "
                "PyTorch build to run the comparison"
            ) from error
        raise
    ratio = severian_ms / pytorch_ms

    print(f"ROCDL validation: PASS ({args.chip}, {rocdl.relative_to(ROOT)})")
    print("Fresh-process CPU baseline (float64; lower is better)")
    print(f"  Severian native: {severian_ms:.3f} ms")
    print(f"  PyTorch CPU:     {pytorch_ms:.3f} ms")
    print(f"  Severian/PyTorch: {ratio:.3f}x")
    print("GPU execution is not timed: the ROCm path currently emits target MLIR but does not link/transfer buffers.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
