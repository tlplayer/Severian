"""Explicit pipeline: syntax -> source -> lexer -> parser -> HIR."""
from dataclasses import dataclass
from py_compiler.frontend.source.source import SourceFile, Diagnostic
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.frontend.parser.parser import parse
from py_compiler.hir.hir.src.program import partition, Program


@dataclass(frozen=True)
class FrontendResult:
    source: SourceFile
    program: Program
    diagnostics: tuple[Diagnostic, ...]


def compile_source(path, text, syntax):
    source = SourceFile(path, text)
    try:
        tokens = lex(source, syntax)
    except Diagnostic as failure:
        return FrontendResult(source, Program((), ()), (failure,))
    from py_compiler.frontend.parser.blocks import parse_blocks
    from py_compiler.mir.lowering import prepare
    root, block_errors = parse_blocks(source, tokens, syntax)
    graph, parse_errors = parse(source, tokens, syntax)
    # Preserve primitive materializer interfaces for the original literal-only milestone.
    if not parse_errors and all(n.kind == "sentence" for n in root.children):
        program, hir_errors = partition(source, graph, syntax)
        return FrontendResult(source, program, tuple(block_errors + hir_errors))
    if block_errors:
        program, hir_errors = partition(source, graph, syntax)
        return FrontendResult(source, program, tuple(block_errors + hir_errors))
    try:
        bodies, declarations = prepare(source, root, syntax)
        constants = tuple(o.payload for b in bodies for block in b.blocks for o in block.operations if o.kind == "constant")
        return FrontendResult(source, Program(constants, (), bodies, declarations), ())
    except Diagnostic as failure:
        return FrontendResult(source, Program((), ()), (failure,))
    except ValueError as failure:
        return FrontendResult(source, Program((), ()), (Diagnostic("HIR/MIR", str(failure), source, root.span),))


import unittest
from py_compiler.syntax.recognition import Syntax


class FrontendTests(unittest.TestCase):
    def test_primitives_to_hir_and_later_keyword_diagnostic(self):
        result = compile_source("x.sev", "small: i8 = -128\nletter = '😀'\nbytes = 4B\nif true:\n", Syntax())
        self.assertEqual([c.type.name for c in result.program.constants], ["i8", "char", "byte"])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].stage, "parser")
