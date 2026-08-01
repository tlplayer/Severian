import argparse
import json
import statistics
import time

import torch


COUNT = 65_536
WORKERS = 4


def workload(base):
    values = base.clone().requires_grad_(True)
    activations = torch.relu(values)
    activations.sum().backward()
    return activations, values.grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=int, default=0, metavar="SAMPLES")
    args = parser.parse_args()
    torch.set_num_threads(WORKERS)
    base = torch.arange(-COUNT // 2, COUNT // 2, dtype=torch.float64)
    if args.benchmark:
        workload(base)
        timings = []
        for _ in range(args.benchmark):
            started = time.perf_counter_ns()
            workload(base)
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
        timings.sort()
        print(
            json.dumps(
                {
                    "median_ms": statistics.median(timings),
                    "p95_ms": timings[
                        min(len(timings) - 1, int(0.95 * len(timings)))
                    ],
                }
            )
        )
        return

    activations, gradients = workload(base)

    print(activations.numel())
    print(int(activations.sum().item()))
    print(int(gradients.sum().item()))


if __name__ == "__main__":
    main()
