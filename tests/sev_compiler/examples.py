#!/usr/bin/env python3
"""Audit documentation examples with the built Severian source compiler.

Build first with `package.pkg/debug/sev build sev_compiler`. This runner deliberately
keeps native build/run results separate from test results: a build excludes test
bodies, and unsupported test modes must not count as passing tests.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/examples"


def source_code(text):
    """Mask strings and comments before looking for top-level declarations."""
    masked = list(text)
    index = 0
    while index < len(text):
        start = index
        if text[index] == "#":
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif text[index] in "\"'":
            delimiter = text[index]
            if text.startswith(delimiter * 3, index):
                delimiter *= 3
            index += len(delimiter)
            while index < len(text) and not text.startswith(delimiter, index):
                index += 2 if text[index] == "\\" else 1
            index = min(index + len(delimiter), len(text))
        else:
            index += 1
            continue
        for position in range(start, index):
            if masked[position] != "\n":
                masked[position] = " "
    return "".join(masked)


def execute(command, directory, stem, timeout):
    started = time.monotonic()
    with (directory / f"{stem}.stdout").open("wb") as stdout, (directory / f"{stem}.stderr").open("wb") as stderr:
        process = subprocess.Popen(
            list(map(str, command)), cwd=directory, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            status = "pass" if code == 0 else "fail"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            status = "timeout"
    diagnostic = (directory / f"{stem}.stderr").read_text(errors="replace")
    first_error = next((line for line in diagnostic.splitlines() if "error:" in line.lower()), "")
    if not first_error and code:
        first_error = next(iter(diagnostic.splitlines()), f"process exited {code}")
    return dict(status=status, exit_code=code, seconds=round(time.monotonic() - started, 3),
                command=list(map(str, command)), diagnostic=first_error)


def audit(source, compiler, artifacts, timeout):
    relative = source.relative_to(EXAMPLES)
    directory = artifacts / relative.with_suffix("")
    directory.mkdir(parents=True)
    text = source_code(source.read_text())
    result = dict(path=str(source.relative_to(ROOT)),
                  has_main=bool(re.search(r"^def main\(", text, re.M)),
                  test_modes=re.findall(r"^test with ([^:\n]+)", text, re.M))
    # Each command has its own output path, including identically named examples.
    result["build"] = execute([compiler, "build", source, "--sysroot", ROOT,
                               "-o", directory / "program"], directory, "build", timeout)
    if result["build"]["status"] == "pass":
        result["run"] = execute([directory / "program"], directory, "run", timeout)
        for stream in ("stdout", "stderr"):
            fixture = source.with_suffix(f".{stream}")
            if fixture.exists() and fixture.read_bytes() != (directory / f"run.{stream}").read_bytes():
                result["run"].update(status="fail", diagnostic=f"{stream} differs from {fixture.relative_to(ROOT)}")
    else:
        result["run"] = dict(status="blocked")
    if re.search(r"^test(?:\s|:)", text, re.M):
        result["test"] = execute([compiler, "test", source, "--sysroot", ROOT,
                                  "-o", directory / "tests"], directory, "test", timeout)
    else:
        result["test"] = dict(status="absent")
    print(f"{relative}: build={result['build']['status']} run={result['run']['status']} test={result['test']['status']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Files or directories beneath docs/examples; defaults to all")
    parser.add_argument("--compiler", type=Path, default=ROOT / "sev_compiler/package.pkg/host/dev/bin/sev_compiler")
    parser.add_argument("--output", type=Path, help="New artifact directory (default: timestamped under target)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30, help="Seconds per build, execution, or test")
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        parser.error(f"compiler is not executable: {compiler}")
    if args.jobs < 1 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    sources = set()
    for requested in args.paths or [str(EXAMPLES)]:
        path = Path(requested).resolve()
        if not path.is_relative_to(EXAMPLES) or not path.exists():
            parser.error(f"expected an existing path beneath {EXAMPLES}: {path}")
        sources.update(path.rglob("*.sev") if path.is_dir() else [path])
    if not sources or any(path.suffix != ".sev" for path in sources):
        parser.error("select at least one .sev example")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    artifacts = (args.output or ROOT / "sev_compiler/package.pkg/examples" / timestamp).resolve()
    artifacts.mkdir(parents=True, exist_ok=False)
    metadata = dict(compiler=str(compiler), compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                    timestamp_utc=timestamp, timeout_seconds=args.timeout)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda source: audit(source, compiler, artifacts, args.timeout), sorted(sources)))
    counts = {stage: dict(Counter(result[stage]["status"] for result in results))
              for stage in ("build", "run", "test")}
    (artifacts / "results.json").write_text(json.dumps(dict(metadata=metadata, counts=counts, results=results), indent=2) + "\n")
    lines = ["# Source compiler example audit", "", f"Compiler: `{compiler}`", "",
             f"SHA-256: `{metadata['compiler_sha256']}`", "", f"Examples: {len(results)}", "",
             "Builds exclude test bodies. Runs execute module initializers and main when present.",
             "Tests are attempted independently, including unsupported modes; absent tests are not passes.",
             "Output is captured; exact comparisons are performed only where adjacent fixtures exist.", "",
             "| Stage | Pass | Fail | Timeout | Blocked | Absent |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage, count in counts.items():
        lines.append(f"| {stage} | " + " | ".join(str(count.get(key, 0)) for key in ("pass", "fail", "timeout", "blocked", "absent")) + " |")
    lines += ["", "| Example | Build | Run | Test | First failure |", "| --- | --- | --- | --- | --- |"]
    for result in results:
        failure = next((result[stage].get("diagnostic", "") for stage in ("build", "run", "test")
                        if result[stage]["status"] in ("fail", "timeout")), "")
        failure = failure.replace("|", "\\|").replace("\n", " ")
        log = Path(result["path"]).relative_to("docs/examples").with_suffix("")
        lines.append(f"| [{result['path']}]({log}/) | {result['build']['status']} | {result['run']['status']} | {result['test']['status']} | {failure} |")
    (artifacts / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"Report: {artifacts / 'REPORT.md'}")
    return int(any(result[stage]["status"] in ("fail", "timeout") for result in results for stage in counts))


if __name__ == "__main__":
    sys.exit(main())
