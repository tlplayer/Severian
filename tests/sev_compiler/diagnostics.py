#!/usr/bin/env python3
"""Check source locations at the source compiler's actual CLI boundary."""
import os
from bootstrap_mlir import ROOT, ARTIFACTS, SEED, run


def check(compiler, directory, name, text, code, message, line, column,
          *, imported=False):
    subject = directory / (name + ".sev")
    subject.write_text(text)
    entry = subject
    if imported:
        entry = directory / (name + "_entry.sev")
        entry.write_text(f'import "{subject.name}"\n')
    # Match invocation from the directory containing the user's test file.
    result = run([compiler, "test", entry.name, "--sysroot", ROOT],
                 cwd=directory, succeeds=False)
    assert result.stdout == "", result.stdout
    assert f"error: {code}: {message}" in result.stderr, result.stderr
    assert f" --> {subject.name}:{line}:{column}\n" in result.stderr, result.stderr
    source_line = text.splitlines()[line - 1]
    assert f"{line} | {source_line}\n" in result.stderr, result.stderr
    assert " " * len(str(line)) + " | " + " " * (column - 1) + "^" in result.stderr, result.stderr
    print(f"PASS: {name} diagnostic location", flush=True)


def main():
    if not os.environ.get("SEVERIAN_SKIP_BUILD"):
        run([SEED, "build"], cwd=ROOT / "sev_compiler")
    compiler = ROOT / "sev_compiler/package.pkg/host/dev/bin/sev_compiler"
    directory = ARTIFACTS / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    check(compiler, directory, "syntax",
          'test "syntax":\n    assert(1 +* 2 == 0)\n',
          "E000120", "expected an expression", 2, 15)
    check(compiler, directory, "semantic",
          'test "semantic":\n    assert(1 + true == 0)\n',
          "E000200", "a boolean cannot initialize an integer", 2, 16)
    check(compiler, directory, "lexer",
          'test "lexer":\n    value = "λ\\q"\n',
          "E000101", "unsupported literal escape", 2, 13)
    check(compiler, directory, "unicode",
          'test "unicode":\n    print("λ", 1 +* 2)\n',
          "E000120", "expected an expression", 2, 19)
    check(compiler, directory, "import_syntax",
          'def broken() -> int:\n    return 1 +* 2\n',
          "E000120", "expected an expression", 2, 15, imported=True)
    check(compiler, directory, "import_semantic",
          'def broken() -> int:\n    return unknown\n',
          "E000200", "unknown name unknown", 2, 12, imported=True)


if __name__ == "__main__":
    main()
