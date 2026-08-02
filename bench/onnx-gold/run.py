#!/usr/bin/env python3
"""Validate and time native Severian against PyTorch on an exported ONNX model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
WORK = ROOT / "bench/.work/onnx-gold"


def timed(command, timeout=120):
    started = time.perf_counter_ns()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout)
    return result, (time.perf_counter_ns() - started) / 1_000_000


def parse_summary(output: bytes):
    lines = output.decode().splitlines()
    if len(lines) != 7:
        raise RuntimeError(f"expected seven output lines, found {output!r}")
    return (
        int(lines[0]),
        [float(value) for value in lines[1:4]],
        [int(value) for value in lines[4:7]],
    )


def validate(candidate, reference, name):
    if candidate[0] != reference[0] or candidate[2] != reference[2]:
        raise RuntimeError(f"{name} shape/class counts differ: {candidate} != {reference}")
    for actual, expected in zip(candidate[1], reference[1]):
        if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=0.05):
            raise RuntimeError(f"{name} checksum differs: {actual} != {expected}")


def checked(command):
    result, elapsed = timed(command)
    if result.returncode != 0:
        raise RuntimeError(f"{command[0]} failed:\n{result.stderr.decode()}")
    return parse_summary(result.stdout), elapsed


def percentile_95(samples):
    return sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--torch-python",
        type=Path,
        default=Path("/tmp/severian-onnx-venv/bin/python"),
    )
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup must be non-negative")
    if not args.torch_python.is_file():
        parser.error(f"PyTorch interpreter not found: {args.torch_python}")

    required = [
        GENERATED / "model.sev",
        GENERATED / "model-unfused.sev",
        GENERATED / "model-sequential.sev",
        GENERATED / "iris-mlp.onnx",
        GENERATED / "features.npy",
    ]
    if args.prepare or not all(path.exists() for path in required):
        prepared = subprocess.run(
            [str(args.torch_python), str(HERE / "prepare.py")], cwd=ROOT
        )
        if prepared.returncode != 0:
            return prepared.returncode

    WORK.mkdir(parents=True, exist_ok=True)
    sev = ROOT / "target/debug/sev"
    build = subprocess.run(
        ["cargo", "build", "-p", "severian-driver", "--bin", "sev"], cwd=ROOT
    )
    if build.returncode != 0:
        return build.returncode
    binary = WORK / "iris-severian"
    compiled, compile_ms = timed(
        [str(sev), "compile", str(GENERATED / "model.sev"), "-o", str(binary)]
    )
    if compiled.returncode != 0:
        raise RuntimeError(compiled.stderr.decode())
    unfused_binary = WORK / "iris-severian-unfused"
    unfused_compiled, _ = timed(
        [
            str(sev),
            "compile",
            str(GENERATED / "model-unfused.sev"),
            "-o",
            str(unfused_binary),
        ]
    )
    if unfused_compiled.returncode != 0:
        raise RuntimeError(unfused_compiled.stderr.decode())
    sequential_binary = WORK / "iris-severian-sequential"
    sequential_compiled, _ = timed(
        [
            str(sev),
            "compile",
            str(GENERATED / "model-sequential.sev"),
            "-o",
            str(sequential_binary),
        ]
    )
    if sequential_compiled.returncode != 0:
        raise RuntimeError(sequential_compiled.stderr.decode())

    commands = {
        "Severian fused 4x": [str(binary)],
        "Severian unfused 4x": [str(unfused_binary)],
        "Severian fused 1x": [str(sequential_binary)],
        "PyTorch": [str(args.torch_python), str(HERE / "reference.py"), "pytorch"],
        "ONNX Runtime": [
            str(args.torch_python),
            str(HERE / "reference.py"),
            "onnxruntime",
        ],
    }
    summaries = {}
    timings = {}
    for name, command in commands.items():
        for _ in range(args.warmup):
            checked(command)
        runs = [checked(command) for _ in range(args.samples)]
        summaries[name] = runs[0][0]
        timings[name] = [run[1] for run in runs]

    validate(summaries["ONNX Runtime"], summaries["PyTorch"], "ONNX Runtime")
    for name in ("Severian fused 4x", "Severian unfused 4x", "Severian fused 1x"):
        validate(summaries[name], summaries["PyTorch"], name)
    metadata = json.loads((GENERATED / "metadata.json").read_text())
    steady = {}
    for engine in ("pytorch", "onnxruntime"):
        result = subprocess.run(
            [
                str(args.torch_python),
                str(HERE / "reference.py"),
                engine,
                "--benchmark",
                "20",
            ],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        steady[engine] = json.loads(result.stdout)
    print("Iris MLP ONNX gold test: 60,000 samples, 4 -> 12 -> 3")
    print(
        f"training accuracy: {metadata['accuracy']:.3%}; "
        f"ONNX graph: {' -> '.join(metadata['operators'])}"
    )
    print(f"Severian native compile: {compile_ms:.3f} ms")
    print("engine                 median process ms  p95 process ms")
    for name in commands:
        print(
            f"{name:22}  {statistics.median(timings[name]):17.3f}  "
            f"{percentile_95(timings[name]):14.3f}"
        )
    print("warm model call (framework already loaded)")
    print("engine        median kernel ms  p95 kernel ms")
    for engine, name in (("pytorch", "PyTorch"), ("onnxruntime", "ONNX Runtime")):
        print(
            f"{name:12}  {steady[engine]['median_ms']:16.3f}  "
            f"{steady[engine]['p95_ms']:13.3f}"
        )
    print(f"class counts: {summaries['PyTorch'][2]}")
    print("All engines matched output shape, class counts, and logit checksums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
