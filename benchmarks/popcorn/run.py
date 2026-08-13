#!/usr/bin/env python3
"""Compile a Severian kernel and benchmark it through the Popcorn CLI."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from bundle import package_submission


POPCORN_MODES = ("prepare", "test", "benchmark", "leaderboard", "profile")


def run_checked(command: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def find_sev(repository: Path, explicit: str | None, skip_build: bool) -> Path:
    if explicit:
        compiler = Path(explicit).expanduser().resolve()
        if not compiler.is_file():
            raise FileNotFoundError(f"Severian compiler not found: {compiler}")
        return compiler
    compiler = repository / "target/debug/sev"
    if not skip_build:
        run_checked(["cargo", "build", "-p", "severian-driver"], repository)
    if not compiler.is_file():
        raise FileNotFoundError(
            f"Severian compiler not found at {compiler}; remove --skip-build or pass --sev"
        )
    return compiler


def find_popcorn(explicit: str | None) -> str:
    if explicit:
        path = shutil.which(explicit) if "/" not in explicit else explicit
        if path and Path(path).is_file():
            return str(path)
        raise FileNotFoundError(f"Popcorn CLI not found: {explicit}")
    for candidate in ("popcorn", "popcorn-cli"):
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError(
        "Popcorn CLI is not installed; prepare succeeded, but remote execution requires "
        "`popcorn` or `popcorn-cli` on PATH"
    )


def load_problem(popcorn_root: Path, name: str) -> tuple[Path, dict[str, object]]:
    problem = (popcorn_root / name).resolve()
    manifest_path = problem / "benchmark.json"
    if not manifest_path.is_file():
        choices = sorted(
            path.parent.name for path in popcorn_root.glob("*/benchmark.json")
        )
        raise FileNotFoundError(
            f"unknown Popcorn benchmark `{name}`; available: {', '.join(choices) or 'none'}"
        )
    return problem, json.loads(manifest_path.read_text(encoding="utf-8"))


def target_for(manifest: dict[str, object], gpu: str, override: str | None) -> str:
    if override:
        return override
    targets = manifest.get("targets", {})
    if not isinstance(targets, dict) or gpu not in targets:
        raise ValueError(
            f"benchmark has no Severian target for Popcorn GPU `{gpu}`; pass --target"
        )
    target = targets[gpu]
    if not isinstance(target, str):
        raise TypeError(f"target mapping for `{gpu}` must be a string")
    return target


def prepare(
    repository: Path,
    problem: Path,
    manifest: dict[str, object],
    compiler: Path,
    gpu: str,
    target: str,
    triton_python: str | None,
) -> tuple[Path, Path, Path, Path | None]:
    source = problem / str(manifest["source"])
    adapter = problem / str(manifest["adapter"])
    entry = str(manifest["entry"])
    leaderboard = str(manifest["leaderboard"])
    build = problem / "build"
    build.mkdir(parents=True, exist_ok=True)
    ttir = build / f"{entry}.ttir"
    submission = build / "submission.py"
    inspection = build / "inspection.txt"

    inspect_command = [
        str(compiler),
        "kernel",
        "inspect",
        str(source),
        "--entry",
        entry,
        "--backend",
        "triton",
        "--target",
        target,
    ]
    inspected = run_checked(inspect_command, repository, capture=True)
    inspection.write_text(inspected.stdout, encoding="utf-8")
    print(inspected.stdout, end="")

    emit_command = [
        str(compiler),
        "kernel",
        "emit",
        str(source),
        "--entry",
        entry,
        "--backend",
        "triton",
        "--target",
        target,
        "--output",
        str(ttir),
    ]
    started = time.perf_counter()
    run_checked(emit_command, repository)
    compile_seconds = time.perf_counter() - started

    packaged = package_submission(
        ttir.read_text(encoding="utf-8"),
        adapter.read_text(encoding="utf-8"),
        leaderboard,
        gpu,
    )
    compile(packaged, str(submission), "exec")
    submission.write_text(packaged, encoding="utf-8")

    metadata = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": manifest["name"],
        "leaderboard": leaderboard,
        "gpu": gpu,
        "target": target,
        "entry": entry,
        "backend": "triton",
        "compile_seconds": compile_seconds,
        "source": str(source.relative_to(repository)),
        "source_sha256": file_hash(source),
        "ttir": str(ttir.relative_to(repository)),
        "ttir_sha256": file_hash(ttir),
        "submission": str(submission.relative_to(repository)),
        "submission_sha256": file_hash(submission),
        "inspect_command": inspect_command,
        "emit_command": emit_command,
    }
    metadata_path = build / "build.json"
    local_compilation = None
    if triton_python:
        local_compilation = build / "compiled" / target.replace(":", "-")
        validation_command = [
            str(Path(triton_python).expanduser()),
            str(repository / "benchmarks/popcorn/compile_ttir.py"),
            str(ttir),
            "--target",
            target,
            "--output",
            str(local_compilation),
        ]
        validation_started = time.perf_counter()
        run_checked(validation_command, repository)
        compilation_path = local_compilation / "compilation.json"
        metadata["offline_compilation"] = {
            "command": validation_command,
            "seconds": time.perf_counter() - validation_started,
            "metadata": str(compilation_path.relative_to(repository)),
            "metadata_sha256": file_hash(compilation_path),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared TTIR:       {ttir.relative_to(repository)}")
    print(f"Prepared submission: {submission.relative_to(repository)}")
    print(f"Build metadata:      {metadata_path.relative_to(repository)}")
    return ttir, submission, metadata_path, local_compilation


def benchmark(
    repository: Path,
    problem: Path,
    manifest: dict[str, object],
    ttir: Path,
    submission: Path,
    metadata_path: Path,
    local_compilation: Path | None,
    mode: str,
    gpu: str,
    popcorn: str,
    benchmark_index: int | None,
) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    result_dir = problem / "results" / f"{stamp}-{gpu}-{mode}"
    result_dir.mkdir(parents=True)
    snapshot_ttir = result_dir / ttir.name
    snapshot_submission = result_dir / "submission.py"
    snapshot_build = result_dir / "build.json"
    shutil.copy2(ttir, snapshot_ttir)
    shutil.copy2(submission, snapshot_submission)
    shutil.copy2(metadata_path, snapshot_build)
    if local_compilation is not None:
        shutil.copytree(
            local_compilation,
            result_dir / "local_compilation",
            ignore=shutil.ignore_patterns(".cache"),
        )
    result_json = result_dir / "popcorn.json"

    command = [
        popcorn,
        "submit",
        str(snapshot_submission),
        "--leaderboard",
        str(manifest["leaderboard"]),
        "--gpu",
        gpu,
        "--mode",
        mode,
        "--no-tui",
        "--output",
        str(result_json),
    ]
    if benchmark_index is not None:
        command.extend(["--benchmark-index", str(benchmark_index)])
    command_path = result_dir / "command.json"
    command_path.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    run_checked(command, repository)
    print(f"Experiment snapshot: {result_dir.relative_to(repository)}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("problem", help="benchmark directory name, for example vectorsum_v2")
    parser.add_argument("--mode", choices=POPCORN_MODES, default="prepare")
    parser.add_argument("--gpu")
    parser.add_argument("--target", help="Severian target, for example cuda:sm_90")
    parser.add_argument("--sev", help="path to the sev compiler")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--popcorn", help="path or command name for the Popcorn CLI")
    parser.add_argument(
        "--triton-python",
        help="Python environment containing Triton; saves TTGIR, LLVM IR, PTX/HSACO, and binary",
    )
    parser.add_argument(
        "--benchmark-index",
        type=int,
        help="Popcorn B200_Brev profile index (ignored by other Popcorn modes)",
    )
    arguments = parser.parse_args()

    popcorn_root = Path(__file__).resolve().parent
    repository = popcorn_root.parents[1]
    problem, manifest = load_problem(popcorn_root, arguments.problem)
    gpu = arguments.gpu or str(manifest["default_gpu"])
    if arguments.benchmark_index is not None and not (
        arguments.mode == "profile" and gpu.casefold() == "b200_brev"
    ):
        raise ValueError(
            "Popcorn only applies --benchmark-index to --mode profile on B200_Brev"
        )
    target = target_for(manifest, gpu, arguments.target)
    compiler = find_sev(repository, arguments.sev, arguments.skip_build)
    ttir, submission, metadata, local_compilation = prepare(
        repository,
        problem,
        manifest,
        compiler,
        gpu,
        target,
        arguments.triton_python,
    )
    if arguments.mode == "prepare":
        return
    popcorn = find_popcorn(arguments.popcorn)
    benchmark(
        repository,
        problem,
        manifest,
        ttir,
        submission,
        metadata,
        local_compilation,
        arguments.mode,
        gpu,
        popcorn,
        arguments.benchmark_index,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
