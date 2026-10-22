import argparse
import json
from pathlib import Path
import subprocess
import sys
from py_compiler.packaging.testing import build_package, test_package
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.syntax.recognition import Syntax


def source_command(argv):
    from py_compiler.packaging.executable import run_source
    parser = argparse.ArgumentParser(prog='sev', usage='sev FILE.sev [--o FILE.exe] [-- ARGS...]')
    parser.add_argument('source')
    parser.add_argument('--o', '-o', '--output', dest='output', help='executable output path')
    parser.add_argument('--no-run', action='store_true', help='compile and link without running')
    arguments = []
    if '--' in argv:
        split = argv.index('--')
        argv, arguments = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    try:
        return run_source(args.source, args.output, arguments, run=not args.no_run)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, Diagnostic, subprocess.TimeoutExpired) as failure:
        print(str(failure), file=sys.stderr)
        return 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ('init', 'build', 'publish', 'tokens', 'test', '-h', '--help'):
        return source_command(argv)
    parser = argparse.ArgumentParser(prog="sev", description="Severian bootstrap compiler",
                                     epilog='Run a source: sev FILE.sev [--o FILE.exe] [-- ARGS...]')
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser('init', help='initialize an empty package directory')
    initialize.add_argument('directory', nargs='?', default='.')
    initialize.add_argument('--name', help='package name (default: directory name)')
    initialize.add_argument('--lib', action='store_true', help='create a library package')
    command = commands.add_parser("build", help="compile package/src/**/*.sev into reusable development outputs")
    command.add_argument("package", nargs="?", default=".")
    command.add_argument("-o", "--o", "--output", dest="executable_output", help="executable path (requires one binary target)")
    command.add_argument("--target", default="x86_64-unknown-linux-gnu")
    command.add_argument("--pointer-bits", type=int, choices=(32, 64), default=64)
    command.add_argument("--jobs", type=int, default=4)
    command.add_argument("--mlir-opt", help="MLIR verifier executable; auto-detected when omitted")
    command.add_argument("--include-tests", action="store_true", help="include development scopes in the compiled artifact")
    command.add_argument('--stop-after', choices=('thir', 'mir'), help='write non-linkable typed metadata without LIR or c codegen')
    command.add_argument('--publish', action='store_true', help='prepare a timestamped local publication snapshot')
    publish = commands.add_parser('publish', help='publish source and compatible c artifacts to the local registry')
    publish.add_argument('reference', nargs='?', help='package name or name:version matching package.json')
    publish.add_argument('--package', default='.', help='package directory (default: current directory)')
    publish.add_argument('--local', action='store_true', help='publish to the local registry')
    publish.add_argument('--target', default='x86_64-unknown-linux-gnu')
    publish.add_argument('--pointer-bits', type=int, choices=(32, 64), default=64)
    publish.add_argument('--jobs', type=int, default=4)
    publish.add_argument('--mlir-opt')
    tokens = commands.add_parser("tokens", help="inspect primitive and keyword recognition")
    tokens.add_argument("source")
    test = commands.add_parser("test", help="build and run a source file or package's test blocks as compiled executables")
    test.add_argument('package', nargs='?', default='.', metavar='FILE_OR_PACKAGE',
                      help='.sev source file or package directory (default: current package)')
    test.add_argument('--compiler', action='store_true', help='run Python compiler unit tests instead')
    test.add_argument('--target', default='x86_64-unknown-linux-gnu')
    test.add_argument('--pointer-bits', type=int, choices=(32, 64), default=64)
    test.add_argument('--jobs', type=int, default=4)
    test.add_argument('--mlir-opt')
    test.add_argument('--sanitize', choices=('address',))
    test.add_argument('--memory', action='store_true', help='check all tests for leaks and invalid memory accesses')
    test.add_argument('--backtrace', action='store_true', help='capture c signal backtraces using gdb')
    test.add_argument('--timeout', type=float, default=30)
    test.add_argument('--parallel', type=int, default=1, help='maximum simultaneous tests marked parallel')
    test.add_argument('--update-snapshots', action='store_true', help='accept values recorded by passing snapshot tests')
    test.add_argument('--coverage', action='store_true', help='write c file coverage under debug/test/coverage/dev')
    test.add_argument('--profile', action='store_true', help='run callable input growth profiling campaigns')
    test.add_argument('--fuzz', action='store_true', help='run fuzz inputs with coverage and append discovered regression tests')
    test.add_argument('--differential', action='store_true', help='compare callable results on shared generated inputs and append differing cases')
    args = parser.parse_args(argv)
    try:
        if args.command == 'publish':
            from py_compiler.packaging.registry import publish_package
            result = publish_package(args.package, args.reference, target=args.target, pointer_bits=args.pointer_bits,
                                     jobs=args.jobs, mlir_opt=args.mlir_opt)
            print('Published: ' + result['path'])
            return 0
        if args.command == 'init':
            from py_compiler.packaging.initialize import init_package
            result = init_package(args.directory, name=args.name, library=args.lib)
            print('Initialized package: ' + result['root'])
            for path in result['created']:
                print('Created: ' + path)
            return 0
        if args.command == "test" and args.compiler:
            import unittest
            modules = ["syntax.complex.chaos", "syntax.complex.interpolation", "syntax.function.trait_dispatch", "syntax.block.mlir", "frontend.source.source", "syntax.primitive.catalog", "syntax.symbol.forms", "syntax.symbol.symbol", "syntax.symbol.absent",
                       "frontend.lexer.lexer", "frontend.parser.parser", "hir.hir.src.program",
                       "frontend.src.lib", "frontend.parser.blocks", "hir.thir.src.lib", "hir.thir.src.contracts", "mir.registration", "mir.cfg.cfg", "mir.ownership.flow",
                       "mir.memory", "syntax.complex.shared", "mir.lowering", "packaging.executable", "syntax.primitive.int.operations", "syntax.primitive.float.operations",
                       "syntax.primitive.bool.declarations", "syntax.primitive.char.declarations",
                       "syntax.primitive.byte.byte", "syntax.primitive.pointer.operations",
                       "syntax.primitive.numeric.conversion", "syntax.complex.array", "syntax.complex.generic", "syntax.function.contracts", "syntax.dynamic.dynamic", "syntax.complex.string", "syntax.operator.operation",
                       "syntax.block.if", "syntax.block.while", "syntax.block.with", "syntax.block.grammar", "syntax.block.procedural", "syntax.complex.iterator", "syntax.block.union", "syntax.function.layout", "syntax.function.allocate", "syntax.grammar.expression", "syntax.grammar.call", "syntax.block.class", "syntax.block.trait", "syntax.block.unsafe", "syntax.block.test", "syntax.prelude", "mlir.src.lib", "mlir.src.cfg", "lir.src.lib", "packaging.package", "packaging.testing", "packaging.coverage", "syntax.complex.fuzz"]
            modules += ['syntax.symbol.none', 'syntax.block.for', 'syntax.complex.campaign', 'packaging.campaigns', 'packaging.initialize', 'packaging.registry', 'syntax.dynamic.any', 'syntax.keywords.defer', 'syntax.block.when',
                        'syntax.function.assertion', 'syntax.block.accept', 'syntax.function.lambda',
                        'syntax.complex.error', 'syntax.function.calls', 'syntax.block.import']
            suite = unittest.defaultTestLoader.loadTestsFromNames(["py_compiler." + m for m in modules])
            return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        if args.command == 'test':
            result = test_package(args.package, parallel=args.parallel, target=args.target, pointer_bits=args.pointer_bits,
                                  jobs=args.jobs, mlir_opt=args.mlir_opt, sanitize=args.sanitize, memory=args.memory, timeout=args.timeout,
                                  backtrace=args.backtrace, update_snapshots=args.update_snapshots,
                                  coverage=args.coverage, fuzz=args.fuzz, profile=args.profile, differential=args.differential,
                                  progress=lambda message: print(message, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
            return 0 if result['success'] else 1
        if args.command == "tokens":
            source = SourceFile(args.source, Path(args.source).read_bytes().decode("utf-8"))
            for token in lex(source, Syntax()):
                print(json.dumps({"kind": token.kind, "text": token.text, "span": [token.span.start, token.span.end]}, ensure_ascii=False))
            return 0
        result = build_package(args.package, target=args.target, pointer_bits=args.pointer_bits, jobs=args.jobs,
                               mlir_opt=args.mlir_opt, include_tests=args.include_tests, stop_after=args.stop_after,
                               publish=args.publish, executable_output=args.executable_output)
        if result["success"]:
            if args.stop_after:
                print('Metadata: ' + result['metadata'])
                return 0
            for path in result['irs']:
                print("MLIR: " + path)
            for path in result['objects']:
                print("Object: " + path)
            for path in result.get("executables", ()):
                print("Executable: " + path)
            print(f"Partitions: {len(result['partitions'])}; reused: {sum(p['reused'] for p in result['partitions'])}")
            return 0
        print(f"Build failed: {result['diagnostics']} diagnostics; showing {len(result['sample'])}.")
        print("\n\n".join(result["sample"]))
        print("Full log: " + result["log"])
        return 1
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        return 130
    except (OSError, ValueError, Diagnostic, subprocess.TimeoutExpired) as failure:
        print(str(failure), file=sys.stderr)
        if args.command == "build":
            print("Full log: " + str(Path(args.package).resolve() / "package.pkg/debug/build/log.txt"), file=sys.stderr)
        return 1


import unittest


class InitCommandTests(unittest.TestCase):
    def test_init_dispatches_without_running_a_source(self):
        import contextlib
        import io
        from unittest.mock import patch
        with patch('py_compiler.packaging.initialize.init_package', return_value={'root': '/project', 'created': []}) as initialize:
            with patch(__name__ + '.source_command') as source, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['init']), 0)
                initialize.assert_called_once_with('.', name=None, library=False)
                source.assert_not_called()
                self.assertEqual(main(['init', '/project', '--name', 'demo', '--lib']), 0)
                initialize.assert_called_with('/project', name='demo', library=True)


class CoverageCommandTests(unittest.TestCase):
    def test_file_path_options_and_failed_status_reach_c_runner(self):
        import contextlib
        import io
        from unittest.mock import patch
        with patch(__name__ + '.test_package', return_value={'success': False}) as runner:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['test', 'nested/example.sev', '--coverage', '--fuzz', '--differential',
                                       '--timeout', '5', '--parallel', '2']), 1)
        self.assertEqual(runner.call_args.args, ('nested/example.sev',))
        self.assertTrue(runner.call_args.kwargs['coverage'])
        self.assertTrue(runner.call_args.kwargs['fuzz'])
        self.assertTrue(runner.call_args.kwargs['differential'])
        self.assertEqual(runner.call_args.kwargs['timeout'], 5)
        self.assertEqual(runner.call_args.kwargs['parallel'], 2)

    def test_coverage_and_fuzz_reach_c_runner(self):
        import contextlib
        import io
        from unittest.mock import patch
        with patch(__name__ + '.test_package', return_value={'success': True}) as runner:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['test', '--coverage', '--fuzz']), 0)
            self.assertTrue(runner.call_args.kwargs['coverage'])
            self.assertTrue(runner.call_args.kwargs['fuzz'])


class InterruptTests(unittest.TestCase):
    def test_keyboard_interrupt_has_a_clean_exit(self):
        import contextlib
        import io
        from unittest.mock import patch
        output = io.StringIO()
        with patch(__name__ + '.test_package', side_effect=KeyboardInterrupt), contextlib.redirect_stderr(output):
            self.assertEqual(main(['test']), 130)
        self.assertEqual(output.getvalue(), 'Interrupted.\n')


class SourceCommandTests(unittest.TestCase):
    def test_direct_source_output_and_literal_program_arguments(self):
        from unittest.mock import patch
        with patch('py_compiler.packaging.executable.run_source', return_value=7) as runner:
            self.assertEqual(main(['hello.sev', '--o', 'out space/app.exe', '--', '--flag', 'two words']), 7)
            runner.assert_called_once_with('hello.sev', 'out space/app.exe',
                                           ['--flag', 'two words'], run=True)

    def test_direct_source_compile_only(self):
        from unittest.mock import patch
        with patch('py_compiler.packaging.executable.run_source', return_value=0) as runner:
            self.assertEqual(main(['hello.sev', '-o', 'hello.exe', '--no-run']), 0)
            runner.assert_called_once_with('hello.sev', 'hello.exe', [], run=False)


class BuildOutputTests(unittest.TestCase):
    def test_executable_output_options_and_report(self):
        import contextlib
        import io
        from unittest.mock import patch
        result = {'success': True, 'irs': [], 'objects': [], 'partitions': [], 'executables': ['/tmp/sort.exe']}
        for option in ('-o', '--o', '--output'):
            with patch(__name__ + '.build_package', return_value=result) as builder:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(['build', option, 'sort.exe']), 0)
                self.assertEqual(builder.call_args.kwargs['executable_output'], 'sort.exe')
                self.assertIn('Executable: /tmp/sort.exe', output.getvalue())
