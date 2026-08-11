#!/usr/bin/env python3
import argparse
import json
import resource
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_REVISION = "14d7620ba47cf51be0b176e14e27e38a34d4ff88"


def synchronize() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--length", required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--end-to-end", action="store_true")
    args = parser.parse_args()
    process_start = time.monotonic_ns()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm GPU is unavailable")
    device = torch.device("cuda:0")
    before = torch.cuda.mem_get_info(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    if args.compile:
        model = torch.compile(model)
    synchronize()
    model_ready = time.monotonic_ns()
    after = torch.cuda.mem_get_info(device)
    record = json.loads(args.inputs.read_text())[args.length]
    tokenize_start = time.monotonic_ns()
    if args.end_to_end:
        token_ids = tokenizer.encode(record["prompt_text"], add_special_tokens=False)
        if token_ids != record["input_ids"]:
            raise RuntimeError("end-to-end tokenizer IDs differ from the fixed compute input")
    else:
        token_ids = record["input_ids"]
    tokenize_end = time.monotonic_ns()
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    warmup_start = time.monotonic_ns()
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    synchronize()
    warmup_end = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        prefill_start = time.monotonic_ns()
        output = model(input_ids=input_ids, use_cache=True)
        synchronize()
        prefill_end = time.monotonic_ns()
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        past = output.past_key_values
        generated = [int(next_token.item())]
        first_token = prefill_end
        decode_start = time.monotonic_ns()
        for _ in range(record["output_tokens"] - 1):
            output = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
        synchronize()
        decode_end = time.monotonic_ns()
    properties = torch.cuda.get_device_properties(device)
    result = {
        "framework": "pytorch_compile" if args.compile else "pytorch_eager",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "gpu_model": properties.name,
        "gpu_index": 0,
        "input_tokens": len(record["input_ids"]),
        "output_tokens": len(generated),
        "process_start_ns": process_start,
        "model_ready_ns": model_ready,
        "first_token_ns": first_token,
        "prefill_ns": prefill_end - prefill_start,
        "decode_ns": decode_end - decode_start,
        "load_ns": model_ready - process_start,
        "tokenize_ns": tokenize_end - tokenize_start,
        "warmup_ns": warmup_end - warmup_start,
        "process_to_ready_ns": warmup_end - process_start,
        "measurement_mode": "end_to_end" if args.end_to_end else "compute",
        "ttft_ns": first_token - prefill_start,
        "decode_tokens_per_second": (
            (len(generated) - 1) * 1e9 / (decode_end - decode_start)
            if len(generated) > 1 and decode_end > decode_start
            else None
        ),
        "gpu_free_total_before": before,
        "gpu_free_total_after": after,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "generated_token_ids": generated,
    }
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
