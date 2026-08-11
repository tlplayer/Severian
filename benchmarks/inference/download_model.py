#!/usr/bin/env python3
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "14d7620ba47cf51be0b176e14e27e38a34d4ff88"
DESTINATION = Path(__file__).parent / "models" / "Qwen2.5-3B-Instruct"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        local_dir=DESTINATION,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ],
    )


if __name__ == "__main__":
    main()
