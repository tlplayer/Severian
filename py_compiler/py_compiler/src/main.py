import argparse
import json
from pathlib import Path
import subprocess
import sys
from py_compiler.testing import build_package, test_package
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.syntax.recognition import Syntax


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sev", description="Severian bootstrap: syntax -> tokens -> submodules -> modules -> MIR -> LIR -> object")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="compile package/src/**/*.sev and publish verified LIR and relocatable objects")
    command.add_argument("package", nargs="?", default=".")
    command.add_argument("--target", default="x86_64-unknown-linux-gnu")
    command.add_argument("--pointer-bits", type=int, choices=(32, 64), default=64)
    command.add_argument("--jobs", type=int, default=4)
    command.add_argument("--mlir-opt", help="MLIR verifier executable; auto-detected when omitted")
    command.add_argument("--include-tests", action="store_true", help="include development scopes in the compiled artifact")
    command.add_argument('--stop-after', choices=('thir', 'mir'), help='write non-linkable typed metadata without LIR or native codegen')
    tokens = commands.add_parser("tokens", help="inspect primitive and keyword recognition")
    tokens.add_argument("source")
    test = commands.add_parser("test", help="build and run the current package's test blocks natively")
    test.add_argument('package', nargs='?', default='.')
    test.add_argument('--compiler', action='store_true', help='run Python compiler unit tests instead')
    test.add_argument('--target', default='x86_64-unknown-linux-gnu')
    test.add_argument('--pointer-bits', type=int, choices=(32, 64), default=64)
    test.add_argument('--jobs', type=int, default=4)
    test.add_argument('--mlir-opt')
    test.add_argument('--sanitize', choices=('address',))
    test.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == "test" and args.compiler:
            import unittest
            modules = ["frontend.source.source", "syntax.primitive.catalog", "syntax.symbol.forms", "syntax.symbol.symbol", "syntax.symbol.absent",
                       "frontend.lexer.lexer", "frontend.parser.parser", "hir.hir.src.program",
                       "frontend.src.lib", "frontend.parser.blocks", "thir.src.lib", "mir.registration", "mir.cfg.cfg", "mir.ownership.flow",
                       "mir.memory", "syntax.complex.shared", "mir.lowering", "syntax.primitive.int.operations", "syntax.primitive.float.operations",
                       "syntax.primitive.bool.declarations", "syntax.primitive.char.declarations",
                       "syntax.primitive.byte.byte", "syntax.primitive.pointer.operations",
                       "syntax.primitive.numeric.conversion", "syntax.complex.array", "syntax.complex.generic", "syntax.function.contracts", "syntax.dynamic.dynamic", "syntax.complex.string", "syntax.operator.operation",
                       "syntax.block.while", "syntax.function.layout", "syntax.function.allocate", "syntax.grammar.expression", "syntax.grammar.call", "syntax.block.class", "syntax.block.trait", "syntax.block.unsafe", "syntax.block.test", "syntax.prelude", "mlir.src.lib", "package", "testing"]
            suite = unittest.defaultTestLoader.loadTestsFromNames(["py_compiler." + m for m in modules])
            return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        if args.command == 'test':
            result = test_package(args.package, target=args.target, pointer_bits=args.pointer_bits,
                                  jobs=args.jobs, mlir_opt=args.mlir_opt, sanitize=args.sanitize, timeout=args.timeout)
            print(json.dumps(result, indent=2))
            return 0 if result['success'] else 1
        if args.command == "tokens":
            source = SourceFile(args.source, Path(args.source).read_bytes().decode("utf-8"))
            for token in lex(source, Syntax()):
                print(json.dumps({"kind": token.kind, "text": token.text, "span": [token.span.start, token.span.end]}, ensure_ascii=False))
            return 0
        result = build_package(args.package, target=args.target, pointer_bits=args.pointer_bits, jobs=args.jobs,
                               mlir_opt=args.mlir_opt, include_tests=args.include_tests, stop_after=args.stop_after)
        if result["success"]:
            if args.stop_after:
                print('Metadata: ' + result['metadata'])
                return 0
            print("MLIR: " + result["ir"])
            print("Object: " + result["object"])
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
