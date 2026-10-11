"""Explicit pipeline: syntax -> source -> lexer -> parser -> HIR -> THIR -> MIR."""
from dataclasses import dataclass, replace
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.frontend.parser.parser import parse
from py_compiler.hir.hir.src.program import partition, Program


@dataclass(frozen=True)
class FrontendResult:
    source: SourceFile
    program: Program
    diagnostics: tuple[Diagnostic, ...]


def compile_source(path, text, syntax, *, stop_after='mir', interface=None, native_packages=None, debug_path=None):
    if stop_after not in ('thir', 'mir'):
        raise ValueError('stop_after must be thir or mir')
    source = SourceFile(path, text, debug_path)
    try:
        tokens = lex(source, syntax)
    except Diagnostic as failure:
        return FrontendResult(source, Program((), ()), (failure,))
    from py_compiler.frontend.parser.blocks import parse_blocks
    from py_compiler.mir.lowering import Builder
    from py_compiler.thir.src.lib import prepare, prepare_literals
    from py_compiler.mir.registration import register
    from py_compiler.hir.hir.src.modules import resolve
    root, block_errors = parse_blocks(source, tokens, syntax)
    from py_compiler.syntax.generic.grammar import SourceWindow
    if block_errors:
        # Recovery retains independently recognized literal declarations for diagnostics.
        graph, _ = parse(source, tokens, syntax)
        program, hir_errors = partition(source, graph, syntax)
        return FrontendResult(source, program, tuple(block_errors + hir_errors))
    if not text:
        return FrontendResult(source, Program((), (), stage=stop_after), ())
    literal_match = syntax.registry.recognize('X.literal-module', SourceWindow(source, 0, len(text), tuple(tokens)), optional=True)
    if literal_match is not None:
        graph, errors = literal_match.provider.construct(literal_match)
        program, hir_errors = partition(source, graph, syntax)
        if errors or hir_errors:
            return FrontendResult(source, program, tuple(errors + hir_errors))
        typed = prepare_literals(source, program, interface)
        modules = (register(typed),) if stop_after == 'mir' else ()
        return FrontendResult(source, replace(program, typed_modules=(typed,), mir_modules=modules, stage=stop_after), ())
    try:
        module = resolve(source, root, syntax, Builder, native_packages, interface)
        typed = prepare(module, interface)
        constants = tuple(o.payload for b in typed.bodies for block in b.blocks for o in block.operations if o.kind == "constant")
        if stop_after == 'thir':
            return FrontendResult(source, Program(constants, module.submodules, (), typed.declarations,
                                                  (typed,), (), 'thir'), ())
        mir = register(typed)
        return FrontendResult(source, Program(constants, module.submodules, mir.bodies, typed.declarations,
                                              (typed,), (mir,), 'mir'), ())
    except Diagnostic as failure:
        return FrontendResult(source, Program((), ()), (failure,))
    except ValueError as failure:
        return FrontendResult(source, Program((), ()), (Diagnostic("HIR/THIR/MIR", str(failure), source, root.span),))


import unittest
from py_compiler.syntax.recognition import Syntax


class FrontendTests(unittest.TestCase):
    def test_primitives_to_hir_and_later_keyword_diagnostic(self):
        result = compile_source("x.sev", "small: i8 = -128\nletter = '😀'\nbytes = 4B\nif true:\n", Syntax())
        self.assertEqual([c.type.name for c in result.program.constants], ["i8", "char", "byte"])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].stage, "parser")
