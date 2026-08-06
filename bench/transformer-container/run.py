#!/usr/bin/env python3
"""Benchmark the same compiled Severian transformer on the host and in OCI."""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEV = ROOT / "target/debug/sev"
EXECUTABLE = HERE / "target/debug/transformer-container-benchmark"
ITERATIONS = 500


def checked(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=300)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result


def parse_workload(output: str) -> tuple[float, str, str]:
    lines = output.splitlines()
    if len(lines) != 3:
        raise RuntimeError(f"unexpected benchmark output: {output!r}")
    return int(lines[0]) / ITERATIONS / 1_000_000, lines[1], lines[2]


def sample(command: list[str], count: int) -> tuple[float, float, str, str]:
    inference_samples = []
    process_samples = []
    expected = None
    for _ in range(count):
        started = time.perf_counter_ns()
        result = checked(command, cwd=HERE)
        process_samples.append((time.perf_counter_ns() - started) / 1_000_000)
        inference_ms, checksum, shape = parse_workload(result.stdout)
        inference_samples.append(inference_ms)
        current = (checksum, shape)
        if expected is not None and current != expected:
            raise RuntimeError(f"non-deterministic output: expected {expected}, got {current}")
        expected = current
    assert expected is not None
    return (
        statistics.median(inference_samples),
        statistics.median(process_samples),
        expected[0],
        expected[1],
    )


def find_runtime(requested: str) -> str | None:
    if requested != "auto":
        return shutil.which(requested)
    return shutil.which("podman") or shutil.which("docker")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--runtime", choices=["auto", "podman", "docker"], default="auto")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--tag", default="severian/transformer-benchmark:local")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")

    checked(["cargo", "build", "-q", "-p", "severian-driver", "--bin", "sev"])
    checked([str(SEV), "build"], cwd=HERE)
    host = sample([str(EXECUTABLE)], args.samples)

    print("workload: encoder inference, 3 tokens, hidden=2, one head, FFN=4, float64")
    print(f"correctness: checksum={host[2]} shape={host[3]}")
    print(f"host:      inference median {host[0]:.6f} ms; process median {host[1]:.3f} ms")

    if args.host_only:
        return 0
    runtime = find_runtime(args.runtime)
    if runtime is None:
        raise RuntimeError("Podman or Docker is required; pass --host-only to measure only the executable")
    checked([runtime, "build", "-t", args.tag, "-f", "Containerfile", "."], cwd=HERE)
    container = sample(
        [runtime, "run", "--rm", "--network", "none", "--memory", "256m", "--cpus", "1", args.tag],
        args.samples,
    )
    if container[2:] != host[2:]:
        raise RuntimeError(f"container output {container[2:]} does not match host {host[2:]}")

    print(f"container: inference median {container[0]:.6f} ms; cold-run median {container[1]:.3f} ms")
    print(f"container/host inference ratio: {container[0] / host[0]:.3f}x")
    print(f"container startup overhead: {container[1] - host[1]:.3f} ms")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}")
        raise SystemExit(1) from None
