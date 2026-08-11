#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--token", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm GPU is unavailable")
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to(device)
    input_ids = torch.tensor([[args.token]], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=True).logits[:, -1]
        torch.cuda.synchronize()
    token = int(logits.argmax(dim=-1).item())
    value = float(logits[0, token].float().item())
    properties = torch.cuda.get_device_properties(device)
    print(json.dumps({
        "framework": "pytorch_eager_reference",
        "gpu_model": properties.name,
        "gpu_index": 0,
        "input_token": args.token,
        "next_token": token,
        "max_logit": value,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint_revision": "14d7620ba47cf51be0b176e14e27e38a34d4ff88",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
