#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from server_common import _VramMonitor, _gpu_snapshot

CHECKPOINT_REVISION = "14d7620ba47cf51be0b176e14e27e38a34d4ff88"
REQUEST_PATTERN = re.compile(
    r"^request_measurement (\d+) (\d+) (\d+) (\d+) (\[.*?\]) (\[.*\])$"
)


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("benchmarks/inference/severian/serving_benchmark.sev"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/inference/results/severian_serving_256x32_c16.json"),
    )
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    fixture = json.loads((ROOT / "inputs.json").read_text())["256"]
    if len(fixture["input_ids"]) != 256 or fixture["output_tokens"] != 32:
        raise RuntimeError("the canonical serving fixture must be exactly 256 input / 32 output")

    before = _gpu_snapshot()
    harness_started_ns = time.monotonic_ns()
    with _VramMonitor(before["vram_used_bytes"]) as monitor:
        process = subprocess.run(
            ["target/debug/sev", "run", str(args.source)],
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
    harness_ended_ns = time.monotonic_ns()
    if process.returncode != 0:
        raise RuntimeError(process.stderr or process.stdout)
    if "platform ROCM" not in process.stderr or "gfx1101" not in process.stderr:
        raise RuntimeError("XLA did not report the expected ROCm gfx1101 device")

    fields: dict[str, str] = {}
    requests = []
    for line in process.stdout.splitlines():
        match = REQUEST_PATTERN.match(line)
        if match:
            request_id, started, first, completed = map(int, match.groups()[:4])
            token_ids = json.loads(match.group(5))
            token_times = json.loads(match.group(6))
            if len(token_ids) != 32 or len(token_times) != 32:
                raise RuntimeError(f"request {request_id} did not generate exactly 32 tokens")
            tpot_ms = [
                (right - left) / 1e6
                for left, right in zip(token_times, token_times[1:])
            ]
            requests.append(
                {
                    "request_id": request_id,
                    "wave": request_id // 16,
                    "started_ns": started,
                    "first_token_ns": first,
                    "completed_ns": completed,
                    "ttft_ms": (first - started) / 1e6,
                    "e2e_latency_ms": (completed - started) / 1e6,
                    "mean_tpot_ms": statistics.fmean(tpot_ms),
                    "generated_token_ids": token_ids,
                    "token_timestamps_ns": token_times,
                    "tpot_intervals_ms": tpot_ms,
                }
            )
            continue
        field = re.match(r"^(\w+)\s+(.*)$", line)
        if field:
            fields[field.group(1)] = field.group(2)

    requests.sort(key=lambda item: item["request_id"])
    if [item["request_id"] for item in requests] != list(range(80)):
        raise RuntimeError(f"expected request IDs 0..79, got {len(requests)} records")
    required = {
        "process_start_ns",
        "weights_mapped_ns",
        "warmup_start_ns",
        "model_ready_ns",
        "benchmark_start_ns",
        "benchmark_end_ns",
        "request_count",
        "concurrency",
    }
    missing = required - fields.keys()
    if missing:
        raise RuntimeError(f"benchmark output is missing {sorted(missing)}")
    if int(fields["request_count"]) != 80 or int(fields["concurrency"]) != 16:
        raise RuntimeError("benchmark workload metadata changed unexpectedly")

    elapsed_s = (
        int(fields["benchmark_end_ns"]) - int(fields["benchmark_start_ns"])
    ) / 1e9
    ttft_values = [item["ttft_ms"] for item in requests]
    tpot_values = [
        interval
        for item in requests
        for interval in item["tpot_intervals_ms"]
    ]
    generated_sequences = {tuple(item["generated_token_ids"]) for item in requests}
    result = {
        "framework": "severian_xla_pjrt",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "dtype": "bfloat16",
        "gpu_model": "AMD Radeon RX 7700",
        "gpu_arch": "gfx1101",
        "gpu_index": 0,
        "gpu_vram_bytes": 17_163_091_968,
        "gpu_memory_bandwidth_gbps": 624.0,
        "gpu_fp16_matrix_tflops": 50.4,
        "gpu_bf16_advertised_tflops": None,
        "input_tokens": 256,
        "output_tokens_per_request": 32,
        "request_count": 80,
        "concurrency": 16,
        "scheduler": fields.get("scheduler", "round_robin_batch_one"),
        "input_token_ids_sha256": hashlib.sha256(
            json.dumps(fixture["input_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "warmup_requests": 1,
        "process_start_ns": int(fields["process_start_ns"]),
        "weights_mapped_ns": int(fields["weights_mapped_ns"]),
        "model_ready_ns": int(fields["model_ready_ns"]),
        "process_to_weights_mapped_ms": (
            int(fields["weights_mapped_ns"]) - int(fields["process_start_ns"])
        ) / 1e6,
        "warmup_ms": (
            int(fields["model_ready_ns"]) - int(fields["warmup_start_ns"])
        ) / 1e6,
        "process_to_ready_ms": (
            int(fields["model_ready_ns"]) - int(fields["process_start_ns"])
        ) / 1e6,
        "benchmark_elapsed_s": elapsed_s,
        "output_tokens_per_second": 80 * 32 / elapsed_s,
        "total_tokens_per_second": 80 * (256 + 32) / elapsed_s,
        "requests_per_second": 80 / elapsed_s,
        "ttft_p50_ms": percentile(ttft_values, 50),
        "ttft_p95_ms": percentile(ttft_values, 95),
        "tpot_p50_ms": percentile(tpot_values, 50),
        "tpot_p95_ms": percentile(tpot_values, 95),
        "gpu_memory_before_bytes": before["vram_used_bytes"],
        "peak_vram_bytes": monitor.peak,
        "all_outputs_identical": len(generated_sequences) == 1,
        "generated_token_ids": requests[0]["generated_token_ids"],
        "harness_process_ns": harness_ended_ns - harness_started_ns,
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
