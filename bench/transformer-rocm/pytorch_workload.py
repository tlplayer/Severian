#!/usr/bin/env python3
"""PyTorch ROCm reference for the Severian transformer encoder benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as functional


def synchronize() -> None:
    torch.cuda.synchronize()


def tensors(requires_grad: bool = False):
    make = lambda values, shape: torch.tensor(values, device="cuda", dtype=torch.float64).reshape(shape).requires_grad_(requires_grad)
    return (
        make([0.2, -0.4, 0.7, 0.1, -0.3, 0.8], (3, 2)),
        make([1.0, 0.0, 0.0, 1.0], (2, 2)),
        make([1.0, 0.0, 0.0, 1.0], (2, 2)),
        make([1.0, 0.0, 0.0, 1.0], (2, 2)),
        make([1.0, 0.0, 0.0, 1.0], (2, 2)),
        make([0.5, -0.25, 0.75, 0.1, -0.4, 0.6, 0.2, 0.3], (2, 4)),
        make([0.4, -0.2, 0.1, 0.5, -0.3, 0.7, 0.6, 0.2], (4, 2)),
    )


def forward(values):
    tokens, query, key, value, output, feed_in, feed_out = values
    bias_in = torch.tensor([0.01, -0.02, 0.03, 0.04] * 3, device="cuda", dtype=torch.float64).reshape(3, 4)
    bias_out = torch.zeros((3, 2), device="cuda", dtype=torch.float64)
    queries = tokens @ query
    keys = tokens @ key
    projected_values = tokens @ value
    attention = torch.softmax((queries @ keys.T) * (2.0 ** -0.5), dim=1)
    context = attention @ projected_values
    residual = functional.layer_norm(tokens + context @ output, (2,), eps=1e-5)
    hidden = torch.relu(residual @ feed_in + bias_in)
    return functional.layer_norm(residual + hidden @ feed_out + bias_out, (2,), eps=1e-5)


def inference_step():
    with torch.no_grad():
        return forward(tensors(False))


def training_step():
    values = tensors(True)
    result = forward(values)
    result.square().mean().backward()
    with torch.no_grad():
        return values[5] - 0.01 * values[5].grad


def time_steps(function, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        function()
    synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    args = parser.parse_args()
    output = inference_step()
    update = training_step()
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_hip": torch.version.hip,
        "output": output.flatten().tolist(),
        "update": update.flatten().tolist(),
        "inference_ms": time_steps(inference_step, args.iterations, args.warmup),
        "training_ms": time_steps(training_step, args.iterations, args.warmup),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
