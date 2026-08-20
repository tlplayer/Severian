#!/usr/bin/env python3
"""Enforce aggregate line and branch coverage from llvm-cov JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


THRESHOLD = 95.0


def percentage(summary: dict[str, object], kind: str) -> float:
    metric = summary.get(kind)
    if not isinstance(metric, dict):
        raise ValueError(f"coverage report has no {kind} summary")
    count = metric.get("count")
    covered = metric.get("covered")
    if not isinstance(count, int) or not isinstance(covered, int):
        raise ValueError(f"coverage report has an invalid {kind} summary")
    return 100.0 if count == 0 else covered * 100.0 / count


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_coverage.py <llvm-cov.json>", file=sys.stderr)
        return 2
    try:
        report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        data = report.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError("coverage report has no data summary")
        totals = data[0].get("totals")
        if not isinstance(totals, dict):
            raise ValueError("coverage report has no totals")
        results = {kind: percentage(totals, kind) for kind in ("lines", "branches")}
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"invalid coverage report: {error}", file=sys.stderr)
        return 2

    failed = False
    for kind, percent in results.items():
        print(f"{kind} coverage: {percent:.2f}% (required: {THRESHOLD:.2f}%)")
        failed |= percent < THRESHOLD
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
