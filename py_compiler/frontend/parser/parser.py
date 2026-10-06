"""Initial grammar: primitive literal sentences and named literal declarations."""
from py_compiler.frontend.parser.contract import Block, BlockGraph, Literal, Sentence
from py_compiler.frontend.source.source import Diagnostic


def parse(source, tokens, syntax):
    graph = BlockGraph()
    root = source.identity + ":root"
    graph.blocks[root] = Block(root, source.span(0, len(source.text)), root)
    diagnostics, line, names = [], [], set()

    def sentence(items):
        if not items or all(t.kind == "INDENT" for t in items):
            return
        first = items[0]
        def fail(message, token=first):
            raise Diagnostic("parser", message, source, token.span)
        if first.kind == "INDENT":
            fail("nested blocks are recognized but not implemented in the primitive milestone")
        name, annotation, binding, position = None, None, None, 0
        if len(items) > 1 and first.kind == "IDENTIFIER" and items[1].text in (":", "=", ":="):
            name, position = first.text, 1
            if name in names:
                fail(f"duplicate declaration {name!r}")
            if items[position].text == ":":
                if position + 1 >= len(items):
                    fail("missing type annotation")
                annotation, position = items[position + 1].text, position + 2
            if position >= len(items) or items[position].text not in ("=", ":="):
                fail("expected = or := followed by a primitive literal")
            binding, position = items[position].text, position + 1
        values = items[position:]
        # Explicit primitive construction supplies the expected type; no Python eval.
        if len(values) >= 3 and values[0].text in syntax.types and values[1].text == "(" and values[-1].text == ")":
            constructed = values[0].text
            if annotation and syntax.types.get(annotation) != syntax.types[constructed]:
                fail("constructor and annotation require different types", values[0])
            annotation, values = constructed, values[2:-1]
        sign = ""
        if values and values[0].text in ("+", "-"):
            sign, values = values[0].text, values[1:]
        if len(values) != 1:
            fail("expected one primitive literal; expressions and blocks require later lowering")
        value = values[0]
        if value.kind not in ("NUMBER", "CHAR", "STRING") and value.text not in ("true", "false", "None", "absent", "unit"):
            fail(f"{value.text!r} is recognized but has no primitive-literal lowering", value)
        if sign and value.kind != "NUMBER":
            fail("a numeric sign requires a numeric literal", value)
        identity = source.identity + ":" + str(first.span.start)
        term_id = identity + ":literal"
        graph.terms[term_id] = Literal(term_id, source.span(items[position].span.start, items[-1].span.end), sign + value.text, value.kind, annotation)
        span = source.span(first.span.start, items[-1].span.end)
        graph.sentences[identity] = Sentence(identity, root, span, term_id, name, binding)
        graph.blocks[root].contents.append(identity)
        if name:
            names.add(name)

    for token in tokens:
        if token.kind in ("NEWLINE", "EOF"):
            try:
                sentence(line)
            except Diagnostic as failure:
                diagnostics.append(failure)
            line = []
        else:
            line.append(token)
    return graph, diagnostics


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
