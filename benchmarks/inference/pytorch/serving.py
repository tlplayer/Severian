#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def synchronize() -> None:
    torch.cuda.synchronize()


def run_request_prefill(model, input_ids, request_id: int, started_ns: int) -> dict:
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(256, device=input_ids.device).unsqueeze(0)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        token = int(output.logits[:, -1].argmax(dim=-1).item())
    synchronize()
    first_ns = time.monotonic_ns()
    return {
        "request_id": request_id,
        "started_ns": started_ns,
        "first_token_ns": first_ns,
        "completed_ns": 0,
        "token": token,
        "tokens": [token],
        "token_times_ns": [first_ns],
        "cache": output.past_key_values,
    }


def run_request_decode(model, state: dict, position: int, device) -> None:
    input_ids = torch.tensor([[state["token"]]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, position + 1), dtype=torch.long, device=device)
    position_ids = torch.tensor([[position]], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=state["cache"],
            use_cache=True,
        )
        token = int(output.logits[:, -1].argmax(dim=-1).item())
    synchronize()
    token_ns = time.monotonic_ns()
    state["token"] = token
    state["tokens"].append(token)
    state["token_times_ns"].append(token_ns)
    state["completed_ns"] = token_ns
    state["cache"] = output.past_key_values


def run_wave(model, fixture_ids, first_request_id: int, count: int, device) -> list[dict]:
    wave_start_ns = time.monotonic_ns()
    input_ids = torch.tensor([fixture_ids], dtype=torch.long, device=device)
    states = [
        run_request_prefill(model, input_ids, first_request_id + offset, wave_start_ns)
        for offset in range(count)
    ]
    for position in range(256, 287):
        for state in states:
            run_request_decode(model, state, position, device)
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm GPU is unavailable")
    device = torch.device("cuda:0")
    fixture = json.loads(args.inputs.read_text())["256"]
    fixture_ids = fixture["input_ids"]
    if len(fixture_ids) != 256 or fixture["output_tokens"] != 32:
        raise RuntimeError("serving fixture must be exactly 256 input / 32 output tokens")

    process_start_ns = time.monotonic_ns()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    if args.compile:
        model = torch.compile(model, dynamic=True)
    synchronize()
    model_loaded_ns = time.monotonic_ns()

    warmup_start_ns = time.monotonic_ns()
    warmup = run_wave(model, fixture_ids, -1, 1, device)[0]
    if len(warmup["tokens"]) != 32:
        raise RuntimeError("warmup did not generate 32 tokens")
    synchronize()
    model_ready_ns = time.monotonic_ns()
    del warmup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    benchmark_start_ns = time.monotonic_ns()
    requests = []
    for wave in range(5):
        states = run_wave(model, fixture_ids, wave * 16, 16, device)
        for state in states:
            token_times = state.pop("token_times_ns")
            state.pop("cache")
            state.pop("token")
            intervals_ms = [
                (right - left) / 1e6
                for left, right in zip(token_times, token_times[1:])
            ]
            state["token_timestamps_ns"] = token_times
            state["ttft_ms"] = (state["first_token_ns"] - state["started_ns"]) / 1e6
            state["e2e_latency_ms"] = (state["completed_ns"] - state["started_ns"]) / 1e6
            state["mean_tpot_ms"] = statistics.fmean(intervals_ms)
            state["tpot_intervals_ms"] = intervals_ms
            state["generated_token_ids"] = state.pop("tokens")
            state["wave"] = wave
            requests.append(state)
        del states
    synchronize()
    benchmark_end_ns = time.monotonic_ns()

    if len(requests) != 80 or any(len(item["generated_token_ids"]) != 32 for item in requests):
        raise RuntimeError("measured workload did not complete 80 requests with 32 tokens each")
    elapsed_s = (benchmark_end_ns - benchmark_start_ns) / 1e9
    ttfts = [item["ttft_ms"] for item in requests]
    tpots = [value for item in requests for value in item["tpot_intervals_ms"]]
    properties = torch.cuda.get_device_properties(device)
    result = {
        "framework": "pytorch_compile_warm" if args.compile else "pytorch_eager",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "dtype": "bfloat16",
        "gpu_model": properties.name,
        "gpu_index": 0,
        "input_tokens": 256,
        "output_tokens_per_request": 32,
        "request_count": 80,
        "concurrency": 16,
        "scheduler": "round_robin_batch_one",
        "input_token_ids_sha256": hashlib.sha256(
            json.dumps(fixture_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "process_to_load_ms": (model_loaded_ns - process_start_ns) / 1e6,
        "warmup_ms": (model_ready_ns - warmup_start_ns) / 1e6,
        "process_to_ready_ms": (model_ready_ns - process_start_ns) / 1e6,
        "benchmark_elapsed_s": elapsed_s,
        "output_tokens_per_second": 80 * 32 / elapsed_s,
        "total_tokens_per_second": 80 * (256 + 32) / elapsed_s,
        "requests_per_second": 80 / elapsed_s,
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "tpot_p50_ms": percentile(tpots, 50),
        "tpot_p95_ms": percentile(tpots, 95),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "all_outputs_identical": len({tuple(item["generated_token_ids"]) for item in requests}) == 1,
        "generated_token_ids": requests[0]["generated_token_ids"],
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "requests"}))


if __name__ == "__main__":
    main()
