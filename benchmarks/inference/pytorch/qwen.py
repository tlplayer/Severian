#!/usr/bin/env python3
import argparse
import hashlib
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
    record = json.loads(args.inputs.read_text())[args.length]
    token_ids = record["input_ids"]
    attention_mask_values = record["attention_mask"]
    position_id_values = record["position_ids"]
    logits_position = record["logits_position"]
    if len(token_ids) != 32 or len(attention_mask_values) != 32 or len(position_id_values) != 32:
        raise RuntimeError("the full-pass fixture must contain exactly 32 tokens")
    if attention_mask_values != [1] * 32 or position_id_values != list(range(32)):
        raise RuntimeError("the full-pass fixture mask or position IDs are invalid")
    if logits_position != 31:
        raise RuntimeError("the full-pass fixture must select logits position 31")
    tokenizer = None
    if args.end_to_end:
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
    tokenize_start = time.monotonic_ns()
    if args.end_to_end:
        assert tokenizer is not None
        token_ids = tokenizer.encode(record["prompt_text"], add_special_tokens=False)
        if token_ids != record["input_ids"]:
            raise RuntimeError("end-to-end tokenizer IDs differ from the fixed compute input")
    else:
        token_ids = record["input_ids"]
    tokenize_end = time.monotonic_ns()
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [attention_mask_values], dtype=torch.long, device=device
    )
    position_ids = torch.tensor([position_id_values], dtype=torch.long, device=device)
    warmup_start = time.monotonic_ns()
    with torch.inference_mode():
        warmup_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    synchronize()
    warmup_token = int(warmup_output.logits[:, logits_position].argmax(dim=-1).item())
    warmup_end = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        synchronize()
        prefill_start = time.monotonic_ns()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        synchronize()
        prefill_end = time.monotonic_ns()
        next_token = output.logits[:, logits_position].argmax(dim=-1, keepdim=True)
        generated = [int(next_token.item())]
        first_token = time.monotonic_ns()
        decode_start = time.monotonic_ns()
        synchronize()
        decode_end = time.monotonic_ns()
    if generated[0] != warmup_token:
        raise RuntimeError("warmup and measured next-token argmax differ")
    expected = record.get("expected_output_ids", [])
    if expected and generated != expected:
        raise RuntimeError(f"generated {generated}, expected {expected}")
    properties = torch.cuda.get_device_properties(device)
    result = {
        "framework": "pytorch_compile" if args.compile else "pytorch_eager",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "gpu_model": properties.name,
        "gpu_index": 0,
        "input_tokens": len(record["input_ids"]),
        "input_token_ids_sha256": hashlib.sha256(
            json.dumps(record["input_ids"], separators=(",", ":")).encode()
        ).hexdigest(),
        "logits_position": logits_position,
        "warmup_token": warmup_token,
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
