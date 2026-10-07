"""Initial grammar: primitive literal sentences and named literal declarations."""
from py_compiler.frontend.parser.contract import Block, BlockGraph, Literal, Sentence
from py_compiler.frontend.source.source import Diagnostic


def recognize_literals(source, tokens, syntax):
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
        explicit_constructor = False
        # Explicit primitive construction supplies the expected type; no Python eval.
        if len(values) >= 3 and values[0].text in syntax.types and values[1].text == "(" and values[-1].text == ")":
            constructed = values[0].text
            if annotation and syntax.types.get(annotation) != syntax.types[constructed]:
                fail("constructor and annotation require different types", values[0])
            annotation, values = constructed, values[2:-1]
            explicit_constructor = True
        sign = ""
        if values and values[0].text in ("+", "-"):
            sign, values = values[0].text, values[1:]
        if len(values) != 1:
            fail("expected one primitive literal; expressions and blocks require later lowering")
        value = values[0]
        if value.kind not in ("NUMBER", "CHAR", "STRING") and value.text not in ("true", "false", "None", "absent"):
            fail(f"{value.text!r} is recognized but has no primitive-literal lowering", value)
        if sign and value.kind != "NUMBER":
            fail("a numeric sign requires a numeric literal", value)
        identity = source.identity + ":" + str(first.span.start)
        term_id = identity + ":literal"
        graph.terms[term_id] = Literal(term_id, source.span(items[position].span.start, items[-1].span.end), sign + value.text, value.kind, annotation, explicit_constructor)
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


from py_compiler.syntax.generic.grammar import Grammar, Match


class LiteralModuleGrammar(Grammar):
    role = 'X.literal-module'

    def __init__(self, syntax):
        self.syntax = syntax

    def recognize(self, window):
        lines, current = [], []
        for token in window.tokens:
            if token.kind in ('NEWLINE', 'EOF'):
                if current:
                    lines.append(current)
                current = []
            else:
                current.append(token)
        for items in lines:
            cursor = 0
            if items[0].kind == 'INDENT':
                return None
            if len(items) > 1 and items[0].kind == 'IDENTIFIER' and items[1].text in (':', '=', ':='):
                cursor = 3 if items[1].text == ':' else 1
                if cursor >= len(items) or items[cursor].text not in ('=', ':='):
                    return None
                cursor += 1
            values = items[cursor:]
            if len(values) >= 3 and values[0].text in self.syntax.types and values[1].text == '(' and values[-1].text == ')':
                values = values[2:-1]
            if values and values[0].text in ('+', '-'):
                values = values[1:]
            if len(values) != 1 or (values[0].kind not in ('NUMBER', 'CHAR', 'STRING') and values[0].text not in ('true', 'false', 'None', 'absent')):
                return None
        return Match(self, window, window.end, {'tokens': window.tokens})

    def construct(self, match):
        return recognize_literals(match.window.source, match.captures['tokens'], self.syntax)
