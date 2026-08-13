#!/usr/bin/env python3
"""Compile TTIR offline and retain Triton's generated GPU artifacts."""

from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path


def parse_target(value: str) -> tuple[str, int | str, int]:
    if value.startswith("cuda:sm_"):
        architecture = value.removeprefix("cuda:sm_").rstrip("af")
        return "cuda", int(architecture), 32
    if value.startswith("rocm:gfx"):
        return "hip", value.removeprefix("rocm:"), 64
    raise ValueError(
        f"offline Triton compilation needs cuda:sm_NN or rocm:gfxNNNN, received `{value}`"
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("ttir", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", str(arguments.output / ".cache"))
    import triton
    from triton.backends.compiler import GPUTarget

    backend, architecture, warp_size = parse_target(arguments.target)
    target = GPUTarget(backend, architecture, warp_size)
    kernel = triton.compile(str(arguments.ttir), target=target)
    artifacts: dict[str, dict[str, object]] = {}
    for extension, content in sorted(kernel.asm.items()):
        path = arguments.output / f"kernel.{extension}"
        if isinstance(content, bytes):
            path.write_bytes(content)
            payload = content
        else:
            path.write_text(content, encoding="utf-8")
            payload = content.encode("utf-8")
        artifacts[extension] = {
            "path": path.name,
            "bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    metadata = {
        "schema": 1,
        "triton_version": triton.__version__,
        "target": arguments.target,
        "entry": kernel.name,
        "artifacts": artifacts,
    }
    metadata_path = arguments.output / "compilation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Triton {triton.__version__} compiled {kernel.name} for {arguments.target}")
    for extension, artifact in artifacts.items():
        print(f"  {extension:6} {artifact['bytes']:>8} bytes  {artifact['path']}")


if __name__ == "__main__":
    main()
