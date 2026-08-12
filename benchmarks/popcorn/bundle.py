#!/usr/bin/env python3
"""Bundle a generated Severian kernel with a benchmark-specific adapter.

Popcorn currently imports one Python file. Keeping this bundler in the
benchmark tree lets the compiler emit a reusable kernel module without knowing
anything about Popcorn's task protocol.
"""

from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("kernel", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--gpu")
    arguments = parser.parse_args()

    header = f"#!POPCORN leaderboard {arguments.leaderboard}\n"
    if arguments.gpu:
        header += f"#!POPCORN gpu {arguments.gpu}\n"
    kernel = arguments.kernel.read_text(encoding="utf-8")
    adapter = arguments.adapter.read_text(encoding="utf-8")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        header + "\n" + kernel.rstrip() + "\n\n" + adapter.lstrip(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
