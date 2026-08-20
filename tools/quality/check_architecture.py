#!/usr/bin/env python3
"""Reject Cargo workspace dependency edges that violate central layer policy."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_policy() -> tuple[dict[str, str], dict[str, set[str]], list[dict[str, str]]]:
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    architecture = manifest["workspace"]["metadata"]["architecture"]
    packages = architecture["packages"]
    allow = {name: set(dependencies) for name, dependencies in architecture["allow"].items()}
    return packages, allow, architecture.get("reject", [])


def cargo_metadata() -> dict[str, object]:
    result = subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--no-deps"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    try:
        assignments, allow, rejection_tests = load_policy()
        metadata = cargo_metadata()
        members = set(metadata["workspace_members"])
        packages = {
            package["name"]: package
            for package in metadata["packages"]
            if package["id"] in members
        }
        missing = sorted(set(packages) - set(assignments))
        stale = sorted(set(assignments) - set(packages))
        missing_allow = sorted(set(packages) - set(allow))
        stale_allow = sorted(set(allow) - set(packages))
        if missing or stale or missing_allow or stale_allow:
            details = []
            if missing:
                details.append(f"unassigned workspace packages: {', '.join(missing)}")
            if stale:
                details.append(f"unknown policy packages: {', '.join(stale)}")
            if missing_allow:
                details.append(f"packages without allowlists: {', '.join(missing_allow)}")
            if stale_allow:
                details.append(f"unknown allowlist packages: {', '.join(stale_allow)}")
            raise ValueError("; ".join(details))

        for name, dependencies in allow.items():
            unknown = sorted(dependencies - set(packages))
            if unknown:
                raise ValueError(f"{name} allows unknown packages: {', '.join(unknown)}")

        for test in rejection_tests:
            source = test["from"]
            target = test["to"]
            if source not in assignments or target not in assignments:
                raise ValueError(f"architecture rejection test names unknown package: {test}")
            if target in allow[source]:
                raise ValueError(
                    f"expected rejection is now allowed: {source} -> {target}"
                )

        violations: list[str] = []
        for name, package in sorted(packages.items()):
            for dependency in package["dependencies"]:
                target = dependency["name"]
                if dependency["kind"] == "dev" or target not in packages:
                    continue
                if target not in allow[name]:
                    violations.append(
                        f"{name} ({assignments[name]}) -> "
                        f"{target} ({assignments[target]})"
                    )
        if violations:
            print("architecture dependency violations:", file=sys.stderr)
            for violation in violations:
                print(f"  {violation}", file=sys.stderr)
            return 1
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"invalid architecture policy: {error}", file=sys.stderr)
        return 2

    print(
        "architecture check passed: "
        f"{len(packages)} packages, {len(rejection_tests)} rejection tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
