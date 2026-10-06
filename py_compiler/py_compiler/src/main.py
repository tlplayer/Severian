import argparse
import json
from pathlib import Path
import subprocess
import sys
from py_compiler.package import build
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.syntax.recognition import Syntax


def main(argv=None):
    parser = argparse.ArgumentParser(prog="py_compiler", description="Severian primitive bootstrap: syntax -> source -> lexer -> parser -> HIR -> MLIR")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="compile package/src/**/*.sev and publish verified MLIR")
    command.add_argument("package", nargs="?", default=".")
    command.add_argument("--target", default="x86_64-unknown-linux-gnu")
    command.add_argument("--pointer-bits", type=int, choices=(32, 64), default=64)
    command.add_argument("--jobs", type=int, default=4)
    command.add_argument("--mlir-opt", help="MLIR verifier executable; auto-detected when omitted")
    tokens = commands.add_parser("tokens", help="inspect primitive and keyword recognition")
    tokens.add_argument("source")
    commands.add_parser("test", help="run tests embedded beside compiler implementations")
    args = parser.parse_args(argv)
    try:
        if args.command == "test":
            import unittest
            modules = ["frontend.source.source", "syntax.primitive.catalog", "syntax.symbol.forms",
                       "frontend.lexer.lexer", "frontend.parser.parser", "hir.hir.src.program",
                       "frontend.src.lib", "mlir.src.lib", "package"]
            suite = unittest.defaultTestLoader.loadTestsFromNames(["py_compiler." + m for m in modules])
            return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        if args.command == "tokens":
            source = SourceFile(args.source, Path(args.source).read_bytes().decode("utf-8"))
            for token in lex(source, Syntax()):
                print(json.dumps({"kind": token.kind, "text": token.text, "span": [token.span.start, token.span.end]}, ensure_ascii=False))
            return 0
        result = build(args.package, args.target, args.pointer_bits, args.jobs, args.mlir_opt)
        if result["success"]:
            print("MLIR: " + result["ir"])
            return 0
        print(f"Build failed: {result['diagnostics']} diagnostics; showing {len(result['sample'])}.")
        print("\n\n".join(result["sample"]))
        print("Full log: " + result["log"])
        return 1
    except (OSError, ValueError, Diagnostic, subprocess.TimeoutExpired) as failure:
        print(str(failure), file=sys.stderr)
        if args.command == "build":
            print("Full log: " + str(Path(args.package).resolve() / "package.pkg/debug/build/log.txt"), file=sys.stderr)
        return 1
