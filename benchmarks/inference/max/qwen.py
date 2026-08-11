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
    parser.add_argument("--length", required=True)
    parser.add_argument("--port", type=int, default=32000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--container-image")
    parser.add_argument("--experimental-kernels")
    args = parser.parse_args()
    record = load_record(args.inputs, args.length)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    root = f"http://127.0.0.1:{args.port}"
    model_argument = str(args.model)
    command = [
        str(Path(sys.executable).with_name("max")),
        "serve",
        "--model",
        model_argument,
        "--devices",
        "gpu:0",
        "--port",
        str(args.port),
    ]
    if args.container_image:
        model_argument = "/model"
        command = [
            "docker",
            "run",
            "--rm",
            "--network=host",
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "-v",
            f"{args.model.resolve()}:/model:ro",
            args.container_image,
            "--model-path",
            model_argument,
            "--devices",
            "gpu:0",
            "--port",
            str(args.port),
        ]
    if args.experimental_kernels:
        command.extend(["--use-experimental-kernels", args.experimental_kernels])
    spec = ServerSpec(
        framework="max_mojo",
        command=command,
        health_url=root + "/health",
        generate_url=root + "/v1/completions",
        request={
            "model": model_argument,
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
