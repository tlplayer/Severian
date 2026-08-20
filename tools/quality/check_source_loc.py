#!/usr/bin/env python3
"""Fail when any repository source file exceeds the line limit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MAX_LINES = 800
SOURCE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hh",
    ".hpp",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".mlir",
    ".py",
    ".rs",
    ".sev",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
}
SOURCE_NAMES = {"Dockerfile", "Makefile"}


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [Path(name) for name in result.stdout.decode().split("\0") if name]


def is_source(path: Path) -> bool:
    return path.name in SOURCE_NAMES or path.suffix.lower() in SOURCE_SUFFIXES


def line_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _ in source)


def main() -> int:
    violations = sorted(
        (line_count(path), path)
        for path in repository_files()
        if path.is_file() and is_source(path) and line_count(path) > MAX_LINES
    )
    if violations:
        print(f"source files may not exceed {MAX_LINES} lines:", file=sys.stderr)
        for lines, path in violations:
            print(f"  {lines:>5}  {path}", file=sys.stderr)
        return 1
    print(f"source LOC limit passed: every source file is <= {MAX_LINES} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
