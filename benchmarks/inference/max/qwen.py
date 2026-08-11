#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))
from server_common import ServerSpec, benchmark_server, load_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--length", choices=("128", "512", "2048"), required=True)
    parser.add_argument("--port", type=int, default=32000)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    record = load_record(args.inputs, args.length)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    root = f"http://127.0.0.1:{args.port}"
    spec = ServerSpec(
        framework="max_mojo",
        command=[
            "max",
            "serve",
            "--model",
            str(args.model),
            "--devices",
            "gpu:0",
            "--port",
            str(args.port),
        ],
        health_url=root + "/health",
        generate_url=root + "/v1/completions",
        request={
            "model": str(args.model),
            "prompt": record["prompt_text"],
            "temperature": 0,
            "max_tokens": record["output_tokens"],
            "ignore_eos": True,
            "stream": True,
        },
        response_kind="openai",
    )
    print(json.dumps(benchmark_server(spec, tokenizer, record["output_tokens"], args.timeout)))


if __name__ == "__main__":
    main()
