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
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    record = load_record(args.inputs, args.length)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    root = f"http://127.0.0.1:{args.port}"
    spec = ServerSpec(
        framework="sglang",
        command=[
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(args.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--mem-fraction-static",
            str(args.memory_fraction),
        ],
        health_url=root + "/health",
        generate_url=root + "/generate",
        request={
            "input_ids": record["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": record["output_tokens"],
                "ignore_eos": True,
            },
            "stream": True,
        },
        response_kind="sglang",
    )
    print(json.dumps(benchmark_server(spec, tokenizer, record["output_tokens"], args.timeout)))


if __name__ == "__main__":
    main()
