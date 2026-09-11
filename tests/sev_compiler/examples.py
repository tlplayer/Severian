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
import math
import sys
import time
from resource_guard import run as guarded_run, DEFAULT_MEMORY


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


def execute(command, directory, stem, timeout, timings=False, memory_bytes=DEFAULT_MEMORY):
    started = time.monotonic()
    environment = os.environ.copy()
    environment.pop('SEVERIAN_TIMINGS', None)
    timing_path = directory / f'{stem}.timings.tsv'
    if timings:
        environment['SEVERIAN_TIMINGS'] = str(timing_path)
    completed = guarded_run(command, cwd=directory, timeout=timeout, memory_bytes=memory_bytes,
                            env=environment, stdout=directory / f'{stem}.stdout',
                            stderr=directory / f'{stem}.stderr',
                            metrics_path=directory / f'{stem}.resources.txt')
    code = completed.returncode
    status = completed.resources['limit'] or ('pass' if code == 0 else 'fail')
    diagnostic = (directory / f"{stem}.stderr").read_text(errors="replace")
    first_error = next((line for line in diagnostic.splitlines() if "error:" in line.lower()), "")
    if not first_error and code:
        first_error = next(iter(diagnostic.splitlines()), f"process exited {code}")
    result = dict(status=status, exit_code=code, seconds=round(time.monotonic() - started, 3),
                  command=list(map(str, command)), diagnostic=first_error)
    result.update(completed.resources)
    if timings and timing_path.exists():
        result['compiler_seconds'] = {
            stage: float(seconds) for stage, seconds in
            (line.split('\t') for line in timing_path.read_text().splitlines()[1:])
        }
    return result


def audit(source, compiler, artifacts, timeout, timings=False, memory_bytes=DEFAULT_MEMORY):
    relative = source.relative_to(EXAMPLES)
    directory = artifacts / relative.with_suffix("")
    directory.mkdir(parents=True)
    text = source_code(source.read_text())
    result = dict(path=str(source.relative_to(ROOT)),
                  has_main=bool(re.search(r"^def main\(", text, re.M)),
                  test_modes=re.findall(r"^test with ([^:\n]+)", text, re.M))
    # Each command has its own output path, including identically named examples.
    result["build"] = execute([compiler, "build", source, "--sysroot", ROOT,
                               "-o", directory / "program"], directory, "build", timeout, timings, memory_bytes)
    if result["build"]["status"] == "pass":
        result["run"] = execute([directory / "program"], directory, "run", timeout, memory_bytes=memory_bytes)
        for stream in ("stdout", "stderr"):
            fixture = source.with_suffix(f".{stream}")
            if fixture.exists() and fixture.read_bytes() != (directory / f"run.{stream}").read_bytes():
                result["run"].update(status="fail", diagnostic=f"{stream} differs from {fixture.relative_to(ROOT)}")
    else:
        result["run"] = dict(status="blocked")
    if re.search(r"^test(?:\s|:)", text, re.M):
        result["test"] = execute([compiler, "test", source, "--sysroot", ROOT,
                                  "-o", directory / "tests"], directory, "test", timeout, timings, memory_bytes)
    else:
        result["test"] = dict(status="absent")
    print(f"{relative}: build={result['build']['status']} run={result['run']['status']} test={result['test']['status']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Files or directories beneath docs/examples; defaults to all")
    parser.add_argument("--compiler", type=Path, default=ROOT / "sev_compiler/package.pkg/host/dev/bin/sev_compiler")
    parser.add_argument("--output", type=Path, help="New artifact directory (default: timestamped under target)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--memory-bytes", type=int, default=DEFAULT_MEMORY,
                        help="Per-command address-space and sampled process-tree RSS limit")
    parser.add_argument("--timeout", type=float, default=30, help="Seconds per build, execution, or test")
    parser.add_argument("--timings", action="store_true", help="Capture compiler stage timings; use --jobs 1 for profiling")
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        parser.error(f"compiler is not executable: {compiler}")
    if args.jobs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.memory_bytes <= 0:
        parser.error("jobs, timeout and memory limit must be positive")
    if args.jobs * args.memory_bytes > 6_000_000_000:
        parser.error("concurrent command budgets must total at most 6 GB; lower --jobs")
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
                    timestamp_utc=timestamp, timeout_seconds=args.timeout, jobs=args.jobs,
                    timings=args.timings, memory_bytes=args.memory_bytes)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda source: audit(source, compiler, artifacts, args.timeout, args.timings, args.memory_bytes), sorted(sources)))
    counts = {stage: dict(Counter(result[stage]["status"] for result in results))
              for stage in ("build", "run", "test")}
    (artifacts / "results.json").write_text(json.dumps(dict(metadata=metadata, counts=counts, results=results), indent=2) + "\n")
    lines = ["# Source compiler example audit", "", f"Compiler: `{compiler}`", "",
             f"SHA-256: `{metadata['compiler_sha256']}`", "", f"Examples: {len(results)}", "",
             "Builds exclude test bodies. Runs execute module initializers and main when present.",
             "Tests are attempted independently, including unsupported modes; absent tests are not passes.",
             "Output is captured; exact comparisons are performed only where adjacent fixtures exist.", "",
             "| Stage | Pass | Fail | Timeout | Memory | Blocked | Absent |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stage, count in counts.items():
        lines.append(f"| {stage} | " + " | ".join(str(count.get(key, 0)) for key in ("pass", "fail", "timeout", "memory", "blocked", "absent")) + " |")
    lines += ["", "| Example | Build | Run | Test | First failure |", "| --- | --- | --- | --- | --- |"]
    for result in results:
        failure = next((result[stage].get("diagnostic", "") for stage in ("build", "run", "test")
                        if result[stage]["status"] in ("fail", "timeout", "memory")), "")
        failure = failure.replace("|", "\\|").replace("\n", " ")
        log = Path(result["path"]).relative_to("docs/examples").with_suffix("")
        lines.append(f"| [{result['path']}]({log}/) | {result['build']['status']} | {result['run']['status']} | {result['test']['status']} | {failure} |")
    lines += ['', '## Native resource measurements', '',
              'Peak RSS is GNU time’s maximum for a command and its waited children, not a sum. '
              'The watchdog separately samples aggregate RSS across the process session. '
              'Address-space limits are inherited per process; failed runs are not performance passes.', '',
              '| Example | Command | Status | Seconds | CPU seconds | Peak RSS MiB | Sampled tree MiB |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for result in results:
        for stage in ('build', 'test'):
            row = result[stage]
            if 'seconds' in row:
                peak = row.get('peak_rss_kib')
                peak_text = f'{peak / 1024:.1f}' if peak is not None else 'unavailable'
                cpu = row.get('user_seconds', 0) + row.get('system_seconds', 0)
                tree = row.get('sampled_tree_peak_rss_bytes', 0) / 1048576
                lines.append(f"| {result['path']} | {stage} | {row['status']} | {row['seconds']:.3f} | {cpu:.2f} | {peak_text} | {tree:.1f} |")
    if args.timings:
        lines += ['', '## Slowest compiler commands', '',
                  'Times include failed commands; partial stage timings remain available in the logs.', '',
                  '| Example | Command | Status | Wall seconds | Compiler stages (seconds) |',
                  '| --- | --- | --- | ---: | --- |']
        commands = [(result['path'], stage, result[stage]) for result in results
                    for stage in ('build', 'test') if 'seconds' in result[stage]]
        for path, stage, result in sorted(commands, key=lambda row: row[2]['seconds'], reverse=True)[:20]:
            stages = ', '.join(f'{name}: {seconds:.6f}' for name, seconds in result.get('compiler_seconds', {}).items())
            lines.append(f"| {path} | {stage} | {result['status']} | {result['seconds']:.3f} | {stages} |")
    (artifacts / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"Report: {artifacts / 'REPORT.md'}")
    return int(any(result[stage]["status"] in ("fail", "timeout", "memory") for result in results for stage in counts))


if __name__ == "__main__":
    sys.exit(main())
