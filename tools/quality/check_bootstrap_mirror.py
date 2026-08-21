#!/usr/bin/env python3
"""Keep the source-language compiler structurally aligned with the Rust stage."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def counterpart(path: Path, host: Path, bootstrap: Path) -> Path:
    relative = path.relative_to(host)
    if path.name == "Cargo.toml":
        relative = relative.with_name("package.toml")
    elif path.suffix == ".rs":
        relative = relative.with_suffix(".sev")
    return bootstrap / relative


def check_mirror(root: Path = ROOT) -> tuple[list[str], int, int]:
    host = root / "compiler"
    bootstrap_root = root / "sev_compiler"
    failures: list[str] = []
    if not bootstrap_root.is_dir():
        return (["missing bootstrap compiler directory: sev_compiler"], 0, 0)

    host_directories = sorted(path for path in host.rglob("*") if path.is_dir())
    for directory in host_directories:
        expected = bootstrap_root / directory.relative_to(host)
        if not expected.is_dir():
            failures.append(
                f"missing mirrored directory: {expected.relative_to(root)}"
            )

    rust_sources = sorted(host.rglob("*.rs"))
    for source in rust_sources:
        expected = counterpart(source, host, bootstrap_root)
        if not expected.is_file():
            failures.append(f"missing source mirror: {expected.relative_to(root)}")

    manifests = sorted(host.rglob("Cargo.toml"))
    for manifest in manifests:
        expected = counterpart(manifest, host, bootstrap_root)
        if not expected.is_file():
            failures.append(f"missing package mirror: {expected.relative_to(root)}")
            continue
        try:
            document = tomllib.loads(expected.read_text(encoding="utf-8"))
            library = document.get("lib", {}).get("path")
            if library != "src/lib.sev":
                failures.append(
                    f"mirrored package has no src/lib.sev target: "
                    f"{expected.relative_to(root)}"
                )
        except (OSError, tomllib.TOMLDecodeError) as error:
            failures.append(f"invalid {expected.relative_to(root)}: {error}")

    contract = bootstrap_root / "bootstrap.toml"
    try:
        document = tomllib.loads(contract.read_text(encoding="utf-8"))
        mirror = document["mirror"]
        for invariant in (
            "require-directory-parity",
            "require-file-parity",
            "require-package-parity",
        ):
            if mirror.get(invariant) is not True:
                failures.append(f"bootstrap mirror must enable {invariant}")
        validation = bootstrap_root / document["acceptance"]["validation-package"]
        examples = bootstrap_root / document["acceptance"]["canonical-examples"]
        if not validation.resolve().is_dir():
            failures.append("bootstrap acceptance validation package does not exist")
        if not examples.resolve().is_dir():
            failures.append("bootstrap canonical example directory does not exist")
        if document["acceptance"].get("allow-skips") is not False:
            failures.append("bootstrap acceptance must deny skipped examples")
        if document["acceptance"].get("allow-source-rewrites") is not False:
            failures.append("bootstrap acceptance must deny source rewrites")
        stages = document.get("stage", [])
        compilers = [stage.get("compiler") for stage in stages]
        if compilers != ["sev0", "sev1", "sev2", "sev3"]:
            failures.append("bootstrap stages must be ordered sev0 through sev3")
        for stage in stages:
            if "test test/validation/examples" not in stage.get("validation", ""):
                failures.append(
                    f"bootstrap stage {stage.get('compiler', '<unknown>')} lacks the canonical validation gate"
                )
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        failures.append(f"invalid sev_compiler/bootstrap.toml: {error}")

    return failures, len(manifests), len(rust_sources)


def main() -> int:
    failures, packages, sources = check_mirror()
    if failures:
        print("bootstrap compiler mirror violations:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(
        "bootstrap mirror check passed: "
        f"{packages} packages, {sources} Rust/Severian source pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
