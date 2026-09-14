#!/usr/bin/env python3
"""Execute the source parser/type checks using a Rust-seed-built harness.

This isolates frontend regressions from unfinished prelude collection wiring.
It does not replace the native source-compiler tests in alias_extensions.py.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

from resource_guard import run


ROOT = Path(__file__).resolve().parents[2]


def read_manifest(path):
    return json.loads("\n".join(
        line for line in path.read_text().splitlines()
        if not line.lstrip().startswith("//")
    ))


def main(fixture_path=None):
    compiler = Path(os.environ.get(
        "SEVERIAN_RUST_COMPILER", ROOT / "package.pkg/release/sev"
    ))
    with tempfile.TemporaryDirectory(prefix="sev-parser-alias-check-") as temporary:
        project = Path(temporary)
        (project / "src").mkdir()
        fixture = (fixture_path or Path(__file__).with_suffix(".sev")).read_text()
        fixture = fixture.replace("../../sev_compiler/", str(ROOT / "sev_compiler") + "/")
        (project / "src/check.sev").write_text(fixture)
        dependencies = {}
        for alias, dependency in read_manifest(ROOT / "sev_compiler/package.json")["dependencies"].items():
            location = (ROOT / "sev_compiler" / dependency["path"]).resolve()
            dependencies[alias] = {
                "path": str(location),
                "package": read_manifest(location / "package.json")["package"]["name"],
            }
        (project / "package.json").write_text(json.dumps({
            "package": {"name": "parser-alias-check", "version": "0.1.0", "edition": "2026"},
            "bin": [{"name": "check", "path": "src/check.sev"}],
            "dependencies": dependencies,
            "test": {"memory-max": "6GB", "timeout-seconds": 240},
        }, indent=2))
        result = run([compiler, "test", project], cwd=ROOT,
                     timeout=240, memory_bytes=6_000_000_000)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
