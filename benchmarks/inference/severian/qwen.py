#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_REVISION = "14d7620ba47cf51be0b176e14e27e38a34d4ff88"
sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))
from server_common import _VramMonitor, _gpu_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-token", type=int, default=348)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    process_start = time.monotonic_ns()
    before = _gpu_snapshot()
    with _VramMonitor(before["vram_used_bytes"]) as monitor:
        process = subprocess.run(
            ["sev", "run", str(args.source)],
            text=True,
            capture_output=True,
            timeout=args.timeout,
        )
    process_end = time.monotonic_ns()
    if process.returncode != 0:
        raise RuntimeError(process.stderr or process.stdout)
    fields: dict[str, str] = {}
    for line in process.stdout.splitlines():
        match = re.match(r"^(\w+)\s+(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2)
    required = ("process_start_ns", "model_ready_ns", "prefill_ns", "next_token")
    missing = [name for name in required if name not in fields]
    if missing:
        raise RuntimeError(f"Severian output is missing {missing}:\n{process.stdout}")
    token = int(fields["next_token"])
    if token != args.expected_token:
        raise RuntimeError(f"Severian returned token {token}, expected {args.expected_token}")
    gpu = "unknown"
    match = re.search(r"StreamExecutor \[0\]: ([^,]+)", process.stderr)
    if match:
        gpu = match.group(1)
    warmup_ns = int(fields["warmup_ns"])
    process_to_ready_ns = int(fields["load_ns"])
    result = {
        "framework": "severian_xla",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "gpu_model": gpu,
        "gpu_index": 0,
        "input_tokens": int(fields["prompt_token_count"]),
        "output_tokens": 1,
        "process_start_ns": int(fields["process_start_ns"]),
        "model_ready_ns": int(fields["model_ready_ns"]),
        "first_token_ns": int(fields["model_ready_ns"]) + int(fields["prefill_ns"]),
        "load_ns": process_to_ready_ns - warmup_ns,
        "warmup_ns": warmup_ns,
        "process_to_ready_ns": process_to_ready_ns,
        "prefill_ns": int(fields["prefill_ns"]),
        "ttft_ns": int(fields["ttft_ns"]),
        "gpu_memory_before_bytes": before["vram_used_bytes"],
        "peak_vram_bytes": monitor.peak,
        "generated_token_ids": [token],
        "harness_process_ns": process_end - process_start,
    }
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
