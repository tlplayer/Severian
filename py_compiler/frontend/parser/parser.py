"""Initial grammar: primitive literal sentences and named literal declarations."""
from py_compiler.frontend.parser.contract import Block, BlockGraph, Literal, Sentence
from py_compiler.frontend.source.source import Diagnostic


def parse(source, tokens, syntax):
    from py_compiler.syntax.grammar.literals import recognize_literals
    return recognize_literals(source, tokens, syntax)


import unittest
from py_compiler.frontend.source.source import SourceFile
from py_compiler.frontend.lexer.lexer import lex
from py_compiler.syntax.recognition import Syntax


class ParserTests(unittest.TestCase):
    def test_recognition_does_not_decode_and_errors_accumulate(self):
        source = SourceFile("x", "x: i8 = -128\nif true:\nwhile false:\n")
        graph, failures = parse(source, lex(source, Syntax()), Syntax())
        self.assertEqual(len(failures), 2)
        self.assertEqual(next(iter(graph.terms.values())).spelling, "-128")
        block = next(iter(graph.blocks.values()))
        self.assertEqual(len(block.contents), 1)
