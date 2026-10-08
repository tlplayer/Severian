"""Procedural B grammar: bounded indentation, headers and scoped expansion."""
from importlib import import_module
from py_compiler.frontend.source.source import Diagnostic
from py_compiler.frontend.parser.blocks import Node
from py_compiler.syntax.generic.grammar import Grammar, Match


class SourceGrammar(Grammar):
    role = 'B.source'

    def __init__(self, syntax):
        self.syntax = syntax

    def recognize(self, window):
        return Match(self, window, window.end, {'tokens': window.tokens})

    def construct(self, match):
        return recognize_blocks(match.window.source, match.captures['tokens'], self.syntax)


def recognize_blocks(source, tokens, syntax=None):
    from py_compiler.syntax.recognition import Syntax
    syntax = syntax or Syntax()
    root = Node(source.identity + ":root", "module", source.span(0, len(source.text)), (), "")
    root.syntax = syntax
    root.provider = import_module("py_compiler.syntax.block.global").Global()
    root.scope = root.provider.bind_scope(root)
    lines, current, errors, delimiters = [], [], [], []
    for token in tokens:
        if token.kind in ("NEWLINE", "EOF"):
            if (delimiters or (current and current[-1].text == "with")) and token.kind != "EOF":
                continue
            if current and any(t.kind != "INDENT" for t in current):
                lines.append(current)
            current = []
        else:
            if token.kind == "INDENT" and (delimiters or (current and current[-1].text == "with")):
                continue
            if token.text in ("(", "[", "{"):
                delimiters.append(token.text)
            elif token.text in (')', ']', '}'):
                expected = {')': '(', ']': '[', '}': '{'}[token.text]
                if not delimiters or delimiters[-1] != expected:
                    errors.append(Diagnostic('parser', 'mismatched grammar delimiter', source, token.span))
                else:
                    delimiters.pop()
            current.append(token)
    if delimiters:
        errors.append(Diagnostic("parser", "unclosed grammar delimiter", source, tokens[-1].span))
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
            active_syntax = parent.syntax
            from dataclasses import replace
            items = [replace(token, syntax=active_syntax) for token in items]
            from py_compiler.syntax.generic.grammar import SourceWindow
            window = SourceWindow(source, first.span.start, items[-1].span.end, tuple(items))
            match = active_syntax.registry.recognize('B.header', window, optional=True)
            provider = match.provider if match else None
            kind = first.text if provider else "sentence"
            header = ()
            if provider:
                header = provider.construct(match)
                provider.attach(parent.children[-1] if parent.children else None, parent)
            node = Node(source.identity + ":" + str(first.span.start), kind,
                        source.span(first.span.start, items[-1].span.end), tuple(items), parent.identity, header=header, provider=provider)
            node.syntax, node.imports = provider.expand(header, active_syntax) if provider else (active_syntax, ())
            node.scope = provider.bind_scope(node, parent.scope) if provider else parent.scope
            parent.children.append(node)
            if provider and provider.has_body(node):
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
        if node.provider:
            node.scope.span = node.span
    extend(root)
    return root, errors
