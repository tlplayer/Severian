#!/usr/bin/env python3
"""Run the exported Iris ONNX weights with PyTorch or ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import onnx
from onnx import numpy_helper


GENERATED = Path(__file__).resolve().parent / "generated"


def output_summary(logits):
    sums = logits.astype(np.float64).sum(axis=0)
    counts = np.bincount(logits.argmax(axis=1), minlength=3)
    print(logits.size)
    for value in sums:
        print(format(float(value), ".15g"))
    for value in counts:
        print(int(value))


def pytorch_runner(features):
    import torch

    torch.set_num_threads(4)
    initializers = {
        item.name: numpy_helper.to_array(item).copy()
        for item in onnx.load(GENERATED / "iris-mlp.onnx").graph.initializer
    }
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 12),
        torch.nn.ReLU(),
        torch.nn.Linear(12, 3),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(initializers["hidden.weight"]))
        model[0].bias.copy_(torch.from_numpy(initializers["hidden.bias"]))
        model[2].weight.copy_(torch.from_numpy(initializers["output.weight"]))
        model[2].bias.copy_(torch.from_numpy(initializers["output.bias"]))
    values = torch.from_numpy(features)

    def run():
        with torch.no_grad():
            return model(values).numpy()

    return run


def onnxruntime_runner(features):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    session = onnxruntime.InferenceSession(
        GENERATED / "iris-mlp.onnx",
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return lambda: session.run(["logits"], {"features": features})[0]


def benchmark(runner, samples):
    runner()
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        runner()
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    timings.sort()
    return {
        "median_ms": statistics.median(timings),
        "p95_ms": timings[min(len(timings) - 1, int(0.95 * len(timings)))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("pytorch", "onnxruntime"))
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--benchmark", type=int, default=0, metavar="SAMPLES")
    args = parser.parse_args()
    features = np.tile(np.load(GENERATED / "features.npy"), (args.repeats, 1))
    if args.engine == "pytorch":
        runner = pytorch_runner(features)
    else:
        runner = onnxruntime_runner(features)
    if args.benchmark:
        print(json.dumps(benchmark(runner, args.benchmark)))
        return
    logits = runner()
    output_summary(logits)


if __name__ == "__main__":
    main()
