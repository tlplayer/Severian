"""Indentation groups blocks before sentence resolution; no evaluation in parsing."""
from dataclasses import dataclass, field
from importlib import import_module
from py_compiler.syntax.sentence.syntax import expression
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


def parse_blocks(source, tokens, syntax=None):
    from py_compiler.syntax.recognition import Syntax
    syntax = syntax or Syntax()
    providers = syntax.block_providers
    root = Node(source.identity + ":root", "global", source.span(0, len(source.text)), (), "")
    root.provider = import_module("py_compiler.syntax.block.global").Global()
    lines, current, errors = [], [], []
    for token in tokens:
        if token.kind in ("NEWLINE", "EOF"):
            if current and any(t.kind != "INDENT" for t in current):
                lines.append(current)
            current = []
        else:
            current.append(token)
    stack = [(0, root)]
    pending = None
    for line in lines:
        indentation = line[0].text if line[0].kind == "INDENT" else ""
        items = line[1:] if indentation else line
        try:
            depth = stack[-1][1].provider.indentation(indentation)
            if pending:
                if depth <= stack[-1][0]:
                    errors.append(Diagnostic("parser", "block requires an indented body", source, pending.span))
                else:
                    if depth != stack[-1][0] + len(pending.provider.indentation_unit):
                        raise ValueError("body must start one declared indentation unit below its parent")
                    stack.append((depth, pending))
                pending = None
            while depth < stack[-1][0]:
                stack.pop()
            if depth != stack[-1][0]:
                raise ValueError("indentation does not match an enclosing block")
            parent = stack[-1][1]
            first = items[0]
            provider = providers.get(first.text)
            kind = first.text if provider else "sentence"
            header = ()
            if provider:
                header = provider.parse_header(items)
                provider.attach(parent.children[-1] if parent.children else None, parent)
            node = Node(source.identity + ":" + str(first.span.start), kind,
                        source.span(first.span.start, items[-1].span.end), tuple(items), parent.identity, header=header, provider=provider)
            parent.children.append(node)
            if provider:
                pending = node
        except ValueError as failure:
            errors.append(Diagnostic("parser", str(failure), source, items[0].span))
    if pending:
        errors.append(Diagnostic("parser", "block requires an indented body", source, pending.span))
    def extend(node):
        for child in node.children:
            extend(child)
        if node.children:
            node.span = source.span(node.span.start, max(node.span.end, node.children[-1].span.end))
    extend(root)
    return root, errors


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
