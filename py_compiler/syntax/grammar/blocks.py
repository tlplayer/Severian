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
            test_header = [item for item in current if item.kind != 'INDENT']
            multiline_test = test_header and test_header[0].text == 'test' and test_header[-1].text != ':'
            if (delimiters or (current and current[-1].text == "with") or multiline_test) and token.kind != "EOF":
                continue
            if current and any(t.kind != "INDENT" for t in current):
                lines.append(current)
            current = []
        else:
            if token.kind == "INDENT" and (delimiters or (current and current[-1].text == "with")):
                continue
            if token.text in ("(", "[", "{"):
                delimiters.append([token.text, False])
            elif token.text == ':' and delimiters:
                delimiters[-1][1] = True
            elif token.text in (')', ']', '}'):
                expected = {')': '(', ']': '[', '}': '{'}[token.text]
                interval = (delimiters and delimiters[-1][1] and
                            delimiters[-1][0] in ('(', '[') and token.text in (')', ']'))
                if not delimiters or (delimiters[-1][0] != expected and not interval):
                    errors.append(Diagnostic('parser', 'mismatched grammar delimiter', source, token.span))
                else:
                    delimiters.pop()
            current.append(token)
    if delimiters:
        errors.append(Diagnostic("parser", "unclosed grammar delimiter", source, tokens[-1].span))
    stack = [(0, root)]
    pending = None
    procedural = {}
    for line in lines:
        indentation = line[0].text if line[0].kind == "INDENT" else ""
        items = line[1:] if indentation else line
        try:
            depth = stack[-1][1].provider.indentation(indentation)
            if pending:
                if depth <= stack[-1][0]:
                    if not getattr(pending.provider, 'allow_empty_body', False):
                        errors.append(Diagnostic("parser", "block requires an indented body", source, pending.span))
                else:
                    if depth != stack[-1][0] + len(pending.provider.indentation_unit):
                        raise ValueError("body must start one declared indentation unit below its parent")
                    stack.append((depth, pending))
                pending = None
            while depth < stack[-1][0]:
                _, completed = stack.pop()
                from py_compiler.syntax.block.procedural import Procedure, ProceduralBlock
                if (getattr(completed.provider, 'spelling', '') == 'grammar'
                        and isinstance(completed.header[1], Procedure)):
                    # Validate the interpreter's body independently of whether
                    # any source later chooses this declaration.
                    def returns(nodes):
                        return any(n.tokens[0].text == 'return' or returns(n.children) for n in nodes)
                    if not returns(completed.children):
                        raise ValueError('procedural grammar requires an explicit recognition return')
                    if completed.header[2] == 'B':
                        procedural.setdefault(completed.parent, []).append(ProceduralBlock(completed, completed.syntax))
            if depth != stack[-1][0]:
                raise ValueError("indentation does not match an enclosing block")
            parent = stack[-1][1]
            match_target = ()
            for position, token in enumerate(items):
                if token.text == 'match' and position > 0 and items[position-1].text in ('=', ':='):
                    match_target = tuple(items[:position])
                    items = items[position:]
                    break
            returning_match = len(items)>2 and items[0].text == 'return' and items[1].text == 'match'
            if returning_match:
                items = items[1:]
            first = items[0]
            active_syntax = parent.syntax
            from dataclasses import replace
            items = [replace(token, syntax=active_syntax) for token in items]
            from py_compiler.syntax.generic.grammar import SourceWindow
            window = SourceWindow(source, first.span.start, items[-1].span.end, tuple(items))
            from py_compiler.syntax.generic.grammar import Registry
            extensions = [provider for _, ancestor in stack for provider in procedural.get(ancestor.identity, ())]
            registry = Registry((*active_syntax.registry.providers, *extensions))
            match = registry.recognize('B.header', window, optional=True)
            provider = match.provider if match else None
            kind = first.text if provider else "sentence"
            header = ()
            if provider:
                header = provider.construct(match)
                provider.attach(parent.children[-1] if parent.children else None, parent)
            node = Node(source.identity + ":" + str(first.span.start), kind,
                        source.span(first.span.start, items[-1].span.end), tuple(items), parent.identity, header=header, provider=provider)
            node.returning_match = returning_match
            node.match_target = match_target
            node.syntax, node.imports = provider.expand(header, active_syntax) if provider else (active_syntax, ())
            if provider and provider.spelling in ('def', 'operator'):
                from py_compiler.syntax.block.mlir import NamespaceAnnotation
                from py_compiler.syntax.complex.namespaces import namespace_syntax
                annotations = []
                for previous in reversed(parent.children):
                    if not isinstance(previous.header, NamespaceAnnotation):
                        break
                    annotations.append(previous.header)
                node.syntax = namespace_syntax(node.syntax, annotations)
            node.scope = provider.bind_scope(node, parent.scope) if provider else parent.scope
            parent.children.append(node)
            if provider and provider.has_body(node):
                pending = node
        except ValueError as failure:
            errors.append(Diagnostic("parser", str(failure), source, items[0].span))
    if pending and not getattr(pending.provider, 'allow_empty_body', False):
        errors.append(Diagnostic("parser", "block requires an indented body", source, pending.span))
    def extend(node):
        from py_compiler.syntax.block.mlir import Mlir
        attached = []
        annotation = None
        decorators = []
        annotation_start = None
        for child in node.children:
            if isinstance(child.provider, Mlir):
                annotation_start = child.span.start if annotation_start is None else annotation_start
                from py_compiler.syntax.block.mlir import NamespaceAnnotation
                if isinstance(child.header, NamespaceAnnotation):
                    decorators.append(child.header)
                    continue
                if annotation is not None:
                    errors.append(Diagnostic('parser', 'duplicate @mlir annotation', source, child.span))
                annotation = child
                continue
            if annotation is not None:
                if getattr(child.provider, 'spelling', '') != 'def' or child.children:
                    errors.append(Diagnostic('parser', '@mlir requires a bodyless def', source, child.span))
                else:
                    child.mlir_operation = annotation.header
                annotation = None
            if any(getattr(a, 'phases', ()) for a in decorators) and getattr(child.provider, 'spelling', '') != 'def':
                errors.append(Diagnostic('parser', 'hook phases currently require a function target; T bindings have no execution phase', source, child.span))
            child.decorators = tuple(decorators)
            if annotation_start is not None:
                child.span = source.span(annotation_start, child.span.end)
                annotation_start = None
            decorators = []
            attached.append(child)
        if decorators:
            errors.append(Diagnostic('parser', 'annotation requires a following declaration', source, node.span))
        if annotation is not None:
            errors.append(Diagnostic('parser', '@mlir requires a following def', source, annotation.span))
        node.children = attached
        for child in node.children:
            extend(child)
        if getattr(node, 'returning_match', False):
            for case in node.children:
                if not case.children or case.children[-1].provider is not None:
                    errors.append(Diagnostic('parser', 'value-producing case requires a final expression', source, case.span))
                    continue
                last = case.children[-1]
                if last.tokens[0].text not in ('return', 'throw'):
                    last.tokens = (last.tokens[0].with_text('return'), *last.tokens)
        if node.children:
            node.span = source.span(node.span.start, max(node.span.end, node.children[-1].span.end))
        if node.provider:
            node.scope.span = node.span
    extend(root)
    return root, errors
