#!/usr/bin/env python3
"""Benchmark the same complete transformer encoder in Severian and PyTorch."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXAMPLE = ROOT / "docs/examples/25-transformer-rocm/main.sev"


def require(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode:
        raise RuntimeError(f"{label}: {result.stderr.strip() or result.stdout.strip()}")


def generate_benchmark(path: Path, iterations: int, warmup: int) -> None:
    source = EXAMPLE.read_text()
    source = source[: source.index("def main():")]
    source += f'''native("__sev_monotonic_ns") def monotonicNs() -> int

def main():
    inference := transformerEncoder()
    for iteration in range(0, {warmup}):
        inference = transformerEncoder()
    inferenceStart = monotonicNs()
    for iteration in range(0, {iterations}):
        inference = transformerEncoder()
    inferenceElapsed = monotonicNs() - inferenceStart

    training := transformerTrainStep()
    for iteration in range(0, {warmup}):
        training = transformerTrainStep()
    trainingStart = monotonicNs()
    for iteration in range(0, {iterations}):
        training = transformerTrainStep()
    trainingElapsed = monotonicNs() - trainingStart

    print(inferenceElapsed)
    print(inference)
    print(trainingElapsed)
    print(training)
'''
    path.write_text(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--chip", default="gfx1101")
    parser.add_argument(
        "--torch-python",
        type=Path,
        default=Path("/home/tplayer/.pyenv/versions/betterquest-rocm/bin/python"),
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")

    work = ROOT / "bench/.work/transformer-rocm"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "benchmark.sev"
    executable = work / "benchmark"
    generate_benchmark(source, args.iterations, args.warmup)
    (work / "Severian.toml").write_text(
        '''[package]
name = "transformer-rocm-benchmark"
version = "0.1.0"
edition = "2026"

[[bin]]
name = "transformer-rocm-benchmark"
path = "benchmark.sev"

[dependencies]
parallel = { path = "../../../library/parallel", version = "0.1.0" }
tensor = { path = "../../../library/tensor", version = "0.1.0" }
'''
    )

    require(
        subprocess.run(
            ["cargo", "build", "-q", "-p", "severian-driver", "--bin", "sev"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ),
        "building sev",
    )
    require(
        subprocess.run(
            [
                str(ROOT / "target/debug/sev"),
                "compile",
                str(source),
                "--target",
                "rocm",
                "--chip",
                args.chip,
                "-o",
                str(executable),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ),
        "compiling Severian ROCm benchmark",
    )
    severian = subprocess.run(
        [str(executable)], cwd=ROOT, text=True, capture_output=True, timeout=300
    )
    require(severian, "Severian GPU benchmark")
    lines = severian.stdout.splitlines()
    if len(lines) != 4:
        raise RuntimeError(f"unexpected Severian output: {severian.stdout!r}")
    severian_inference_ms = int(lines[0]) / args.iterations / 1_000_000
    severian_output = ast.literal_eval(lines[1])
    severian_training_ms = int(lines[2]) / args.iterations / 1_000_000
    severian_update = ast.literal_eval(lines[3])

    torch_env = os.environ.copy()
    torch_result = subprocess.run(
        [
            str(args.torch_python),
            str(HERE / "pytorch_workload.py"),
            "--iterations",
            str(args.iterations),
            "--warmup",
            str(args.warmup),
        ],
        cwd=ROOT,
        env=torch_env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    require(torch_result, "PyTorch ROCm benchmark")
    torch = json.loads(torch_result.stdout)

    output_error = max(abs(a - b) for a, b in zip(severian_output, torch["output"]))
    update_error = max(abs(a - b) for a, b in zip(severian_update, torch["update"]))
    if len(severian_output) != len(torch["output"]) or output_error > 2e-4:
        raise RuntimeError(f"inference mismatch (max absolute error {output_error})")
    if len(severian_update) != len(torch["update"]) or update_error > 2e-4:
        raise RuntimeError(f"training mismatch (max absolute error {update_error})")

    print(f"device: {torch['device']}")
    print(f"software: Severian MLIR/ROCDL gfx1101; PyTorch {torch['torch_version']} HIP {torch['torch_hip']}")
    print(f"dataset: 3 tokens, hidden=2, one attention head, FFN=4, float64")
    print(f"correctness: PASS (forward max abs {output_error:.3g}; SGD update max abs {update_error:.3g})")
    print("warm latency per step (Severian mean; PyTorch median)")
    print(f"  inference  Severian {severian_inference_ms:.6f} ms | PyTorch {torch['inference_ms']:.6f} ms")
    print(f"  training   Severian {severian_training_ms:.6f} ms | PyTorch {torch['training_ms']:.6f} ms")
    print(f"  inference ratio Severian/PyTorch: {severian_inference_ms / torch['inference_ms']:.2f}x")
    print(f"  training ratio Severian/PyTorch: {severian_training_ms / torch['training_ms']:.2f}x")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
