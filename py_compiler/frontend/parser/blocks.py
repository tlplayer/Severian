"""Indentation groups blocks before sentence resolution; no evaluation in parsing."""
from dataclasses import dataclass, field
from importlib import import_module
from py_compiler.syntax.grammar.expression import expression
from py_compiler.frontend.source.source import Diagnostic


@dataclass
class Node:
    identity: str
    kind: str
    span: object
    tokens: tuple
    parent: str
    children: list = field(default_factory=list)
    header: tuple = ()
    provider: object = None
    syntax: object = None
    imports: tuple = ()
    scope: object = None


def parse_blocks(source, tokens, syntax=None):
    from py_compiler.syntax.recognition import Syntax
    from py_compiler.syntax.generic.grammar import SourceWindow
    syntax = syntax or Syntax()
    # An empty source has an empty module without attempting a zero-width match.
    if not source.text:
        from py_compiler.syntax.grammar.blocks import recognize_blocks
        return recognize_blocks(source, tokens, syntax)
    return syntax.registry.construct('B.source', SourceWindow(source, 0, len(source.text), tuple(tokens)))


import unittest
from py_compiler.frontend.source.source import SourceFile
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.syntax.recognition import Syntax


class BlockParserTests(unittest.TestCase):
    def test_provider_registration_adds_syntax_without_parser_or_lexer_changes(self):
        from importlib import import_module
        from py_compiler.frontend.src.lib import compile_source
        class When(import_module("py_compiler.syntax.block.if").Conditional):
            spelling = "when"
        class ExtendedSyntax(Syntax):
            @property
            def block_providers(self):
                return {**super().block_providers, "when": When()}
        result = compile_source("extension.sev", "when true:\n    unimplemented\n", ExtendedSyntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertTrue(any(b.terminator.kind == "conditional" for b in result.program.bodies[0].blocks))

    def test_containment_and_sibling_conditionals(self):
        source = SourceFile("x", "def example():\n    if true:\n        unimplemented\n    elif false:\n        unimplemented\n    else:\n        unimplemented\n")
        root, errors = parse_blocks(source, lex(source, Syntax()))
        self.assertFalse(errors)
        self.assertEqual([n.kind for n in root.children[0].children], ["if", "elif", "else"])
        self.assertEqual(root.children[0].children[0].parent, root.children[0].identity)

    def test_orphan_else_and_bad_dedent(self):
        for text in ("else:\n    unimplemented\n", "def example():\n    unimplemented\n  unimplemented\n"):
            source = SourceFile("x", text)
            self.assertTrue(parse_blocks(source, lex(source, Syntax()))[1])
