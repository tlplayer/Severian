#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

REQUIRED = {
    "framework",
    "gpu_model",
    "gpu_index",
    "model_ready_ns",
    "first_token_ns",
    "peak_vram_bytes",
    "generated_token_ids",
    "checkpoint_revision",
}


def run_once(command: list[str], timeout: float) -> dict:
    started = time.monotonic_ns()
    process = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    ended = time.monotonic_ns()
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}):\n{process.stderr}")
    lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError("framework emitted no JSON measurement")
    result = json.loads(lines[-1])
    missing = REQUIRED - result.keys()
    if missing:
        raise RuntimeError(f"measurement missing proof fields: {sorted(missing)}")
    if not result["generated_token_ids"]:
        raise RuntimeError("framework generated no tokens")
    result["harness_process_ns"] = ended - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("a framework command is required")
    samples = [run_once(args.command, args.timeout) for _ in range(args.repetitions)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"samples": samples}, indent=2) + "\n")


if __name__ == "__main__":
    main()
