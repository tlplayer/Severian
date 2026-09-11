#!/usr/bin/env python3
"""Measure primitive support without treating a declaration body as execution.

No compiler is rebuilt. Each invocation records exact commands, exits and output.
The default is an inventory run; --fail-on-unsupported makes failed stages fatal.
Artifacts (including parsed declarations and expanded Agent IR) are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from bootstrap_mlir import ROOT, SEED
from resource_guard import run as guarded_run


PRIMITIVES = ROOT / "sev_compiler/universal/primitive"
COMPILER = Path(os.environ.get(
    "SEVERIAN_SOURCE_COMPILER", ROOT / "sev_compiler/package.pkg/host/dev/bin/sev_compiler"))
STAGES = ("seed_parse", "seed_check", "source_check", "source_tests", "agent_ir", "native")
# Top-level test helpers are real parsed declarations, but not primitive API.
API_FIXTURE_TYPES = {"bool.sev": {"truth_box"}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(arguments, directory, name, timeout):
    command = list(map(str, arguments))
    measurements = {}
    try:
        result = guarded_run(command, cwd=ROOT, timeout=timeout,
                             metrics_path=directory / (name + '.resources.txt'))
        code, stdout, stderr = result.returncode, result.stdout.encode(), result.stderr.encode()
        measurements = result.resources
        status = measurements['limit'] or ("pass" if code == 0 else "crash" if code < 0 else "fail")
    except OSError as error:
        code, stdout, stderr = None, b"", str(error).encode()
        status = "unavailable"
    (directory / (name + ".stdout")).write_bytes(stdout)
    (directory / (name + ".stderr")).write_bytes(stderr)
    return {"status": status, "exit": code, "command": command,
            "stdout": name + ".stdout", "stderr": name + ".stderr", **measurements}


def seed_declarations(ast, subject):
    """Read declaration nodes and byte spans from the seed's parsed AST dump.

    This deliberately does not scan .sev lines for declaration-looking prose.
    Debug AST is not a stable serialization: fail closed on missing node spans.
    Nodes inside tests are classified separately, never counted as public API.
    """
    data = subject.read_bytes()
    declarations, stack = [], []
    active = False
    kinds = ("ClassDeclaration", "TraitDeclaration", "EnumDeclaration", "EnumVariant",
             "TypeDeclaration", "ExtensionDeclaration", "FunctionDeclaration",
             "OperatorDeclaration", "OperatorImplementation", "PropertyDeclaration",
             "TestDeclaration", "CompilerTestCase")
    for line in ast.splitlines():
        if line.startswith("// module "):
            active = Path(line[len("// module "):]).resolve() == subject.resolve()
            stack = []
            continue
        if not active:
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stack:
            stack[-1]["lines"].append(line)
        if stripped.endswith(" {") and stripped[:-2] in kinds:
            stack.append({"kind": stripped[:-2], "indent": indent, "lines": [],
                          "test": any(n["kind"] in ("TestDeclaration", "CompilerTestCase") for n in stack)})
        elif stack and indent == stack[-1]["indent"] and stripped == "},":
            node = stack.pop()
            body = "\n".join(node["lines"])
            margin = " " * (indent + 4)
            span = re.search(r"^" + margin + r"span: Span \{\n(.*?)^" + margin + r"\},", body, re.M | re.S)
            if not span:
                raise ValueError(f"AST declaration without a span: {node['kind']}")
            start = int(re.search(r"start: (\d+)", span[1])[1])
            end = int(re.search(r"end: (\d+)", span[1])[1])
            name = re.search(r"^" + margin + r'name: "(.*)",', body, re.M)
            line_number = data[:start].count(b"\n") + 1
            spelling = data[start:end].decode("utf-8")
            declarations.append({"kind": node["kind"], "name": name[1] if name else None,
                                 "line": line_number, "start": start, "end": end,
                                 "header": spelling.splitlines()[0] if spelling else "",
                                 "in_test": node["test"],
                                 "private": bool(name and name[1].startswith("_"))})
            if node["kind"] == "ClassDeclaration":
                aliases = re.search(r"^" + margin + r"aliases: \[\n(.*?)^" + margin + r"\],", body, re.M | re.S)
                if aliases:
                    alias_margin = margin + " " * 8
                    for alias in re.finditer(r"^" + alias_margin + r"span: Span \{\n(.*?)^" + alias_margin + r"\},", aliases[1], re.M | re.S):
                        alias_start = int(re.search(r"start: (\d+)", alias[1])[1])
                        alias_end = int(re.search(r"end: (\d+)", alias[1])[1])
                        spelling = data[alias_start:alias_end].decode("utf-8")
                        declarations.append({"kind": "ClassAlias", "name": spelling,
                                             "line": data[:alias_start].count(b"\n") + 1,
                                             "start": alias_start, "end": alias_end,
                                             "header": spelling, "in_test": node["test"], "private": False})
    if stack:
        raise ValueError("incomplete seed AST dump")
    fixtures = [d for d in declarations if d["kind"] == "ClassDeclaration"
                and d["name"] in API_FIXTURE_TYPES.get(subject.name, set())]
    for declaration in declarations:
        in_fixture = any(f["start"] <= declaration["start"] < f["end"] for f in fixtures)
        if not subject.is_relative_to(PRIMITIVES) or in_fixture:
            declaration["role"] = "fixture"
        elif declaration["in_test"] or declaration["kind"] in ("TestDeclaration", "CompilerTestCase"):
            declaration["role"] = "test"
        elif declaration["private"]:
            declaration["role"] = "private"
        else:
            declaration["role"] = "api"
    return sorted(declarations, key=lambda item: (item["start"], -item["end"]))


def measure(subject, output, timeout):
    relative = subject.relative_to(ROOT)
    directory = output / str(relative).replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    row = {"file": str(relative), "sha256": digest(subject),
           "artifacts": directory.name, "stages": {}}
    stages = row["stages"]
    ast = directory / "seed.ast"
    stages["seed_parse"] = invoke([SEED, "build", subject, "--emit", "ast", "-o", ast], directory, "seed_parse", timeout)
    if stages["seed_parse"]["status"] == "pass":
        try:
            declarations = seed_declarations(ast.read_text(), subject)
            (directory / "declarations.json").write_text(json.dumps(declarations, indent=2) + "\n")
            row["parsed_nodes"] = len(declarations)
            row["api_declarations"] = sum(d["role"] == "api" for d in declarations)
            row["parsed_tests"] = sum(d["kind"] in ("TestDeclaration", "CompilerTestCase") for d in declarations)
        except (ValueError, OSError) as error:
            row["inventory_error"] = str(error)
    else:
        row["inventory_error"] = "seed parsing failed; use expanded source IR where available"
    stages["seed_check"] = invoke([SEED, "check", subject], directory, "seed_check", timeout)
    common = [subject, "--sysroot", ROOT]
    stages["source_check"] = invoke([COMPILER, "check", *common], directory, "source_check", timeout)
    mlir = directory / "tests.mlir"
    stages["source_tests"] = invoke([COMPILER, "test", *common, "--emit", "mlir", "-o", mlir], directory, "source_tests", timeout)
    agent = directory / "agent.json"
    stages["agent_ir"] = invoke([COMPILER, "test", *common, "--emit", "agent-ir", "-o", agent], directory, "agent_ir", timeout)
    if stages["agent_ir"]["status"] == "pass":
        try:
            ir = json.loads(agent.read_text())
            if ir.get("schema_version") != 1 or ir.get("stage") != "cfg":
                raise ValueError("expected CFG Agent IR schema 1")
            # Retain the entire compiler-produced definition inventory: this
            # includes generated specializations, their IDs and origin spans.
            row["all_expanded_definitions"] = len(ir["definitions"])
            expanded = [definition for definition in ir["definitions"]
                        if definition.get("source", {}).get("path")
                        and Path(definition["source"]["path"]).resolve() == subject.resolve()]
            row["expanded_definitions"] = len(expanded)
            (directory / "expanded-declarations.json").write_text(json.dumps(expanded, indent=2) + "\n")
        except (ValueError, KeyError, OSError) as error:
            stages["agent_ir"]["status"] = "invalid-artifact"
            stages["agent_ir"]["error"] = str(error)
    if stages["source_tests"]["status"] == "pass":
        stages["native"] = invoke([COMPILER, "test", *common, "-o", directory / "tests.exe"], directory, "native", timeout)
    else:
        stages["native"] = {"status": "blocked", "reason": "source test compilation failed"}
    # A file with no executable tests can compile and run an empty entry point.
    # That proves executable lowering only, never per-operation coverage.
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "sev_compiler/package.pkg/primitive-ledger")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--snapshot", type=Path, help="write a portable checkpoint with inline diagnostics")
    parser.add_argument("--fail-on-unsupported", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = {str(p): digest(p) for p in (SEED, COMPILER)}
    rows = []
    subjects = sorted(PRIMITIVES.rglob("*.sev"))
    subjects += sorted((ROOT / "tests/sev_compiler/fixtures/primitives").glob("*.sev"))
    for subject in subjects:
        row = measure(subject, output, args.timeout)
        rows.append(row)
        print(row["file"] + ": " + ", ".join(f"{s}={row['stages'][s]['status']}" for s in STAGES), flush=True)
        (output / "results.json").write_text(json.dumps({"schema_version": 1, "compilers": before, "files": rows}, indent=2) + "\n")
    if before != {str(p): digest(p) for p in (SEED, COMPILER)}:
        raise RuntimeError("compiler binary changed during measurement")
    summary = ["# Measured primitive support", "", "Generated by `python3 tests/sev_compiler/primitives.py`.", "",
               "A pass measures the named stage, not complete API coverage. Native runs execute",
               "the file's existing tests; files without tests can run an empty entry point.", "",
               "| File | Seed parse | Seed check | Source check | Source tests | Agent IR | Native |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        summary.append("| " + row["file"] + " | " + " | ".join(row["stages"][s]["status"] for s in STAGES) + " |")
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    if args.snapshot:
        snapshot = {"schema_version": 1, "compilers": before, "files": rows}
        for row in rows:
            for stage in row["stages"].values():
                for stream in ("stdout", "stderr"):
                    if stream in stage:
                        stage[stream + "_text"] = (output / row["artifacts"] / stage[stream]).read_text(errors="replace")
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snapshot, indent=2) + "\n")
    if args.fail_on_unsupported and any(row["stages"][s]["status"] != "pass" for row in rows for s in STAGES):
        parser.exit(1, "Primitive capability failures are recorded in results.json.\n")


if __name__ == "__main__":
    main()
