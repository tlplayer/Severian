#!/usr/bin/env python3
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "Qwen2.5-3B-Instruct"
OUTPUT = ROOT / "inputs.json"
LENGTHS = (128, 256, 512, 2048)


def exact_length(tokenizer, target: int) -> tuple[list[int], str]:
    seed = "Explain why deterministic compiler benchmarks require identical token IDs. "
    ids = tokenizer.encode(seed * (target // 8 + 2), add_special_tokens=False)
    if len(ids) < target:
        raise RuntimeError(f"could not construct {target} input tokens")
    ids = ids[:target]
    prompt = tokenizer.decode(ids, skip_special_tokens=False)
    if tokenizer.encode(prompt, add_special_tokens=False) != ids:
        raise RuntimeError(f"tokenizer did not round-trip the {target}-token prompt")
    return ids, prompt


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    records = {}
    for length in LENGTHS:
        input_ids, prompt = exact_length(tokenizer, length)
        records[str(length)] = {
            "input_ids": input_ids,
            "prompt_text": prompt,
            "output_tokens": 32 if length == 256 else 128,
        }
    OUTPUT.write_text(json.dumps(records, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
