"""Shared field contract tools used by record-like syntax providers."""
from dataclasses import dataclass
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.sentence.syntax import expression
from py_compiler.frontend.parser.contract import Literal
from py_compiler.hir.hir.src.program import resolve_literal


@dataclass(frozen=True)
class TypeDeclaration:
    identity: str
    name: str
    kind: str
    fields: tuple
    variants: tuple
    traits: tuple
    span: object
    source: str = ""
    defaults: tuple = ()


class RecordProvider(DeclarationProvider):
    def parse_header(self, items):
        header = super().parse_header(items)
        if len(header) != 1 or header[0].kind != "IDENTIFIER":
            raise ValueError(f"{self.spelling} requires a name")
        return header

    def field(self, child, context):
        items = list(child.tokens)
        if items[0].kind != "IDENTIFIER":
            raise ValueError("expected a named field")
        position, annotation = 1, None
        if len(items) > 2 and items[1].text == ":":
            annotation = items[2].text
            if annotation not in context.syntax.types:
                raise ValueError(f"unresolved field type {annotation!r}")
            position = 3
        default = None
        if position < len(items):
            if items[position].text not in ("=", ":="):
                raise ValueError("expected a field initializer")
            term = expression(items[position + 1:])
            if term.kind != "literal":
                raise ValueError("field initializer requires a literal provider in this milestone")
            type_, default = resolve_literal(Literal(child.identity, term.token.span, term.token.text, term.token.kind, annotation), context.syntax)
            annotation = type_.name
        if annotation is None:
            raise ValueError("field requires a type or initializer")
        return (items[0].text, context.syntax.types[annotation].name), default

    def fields(self, node, context):
        fields, defaults, members = [], [], []
        for child in node.children:
            if child.provider:
                members.append(child)
                continue
            field, default = self.field(child, context)
            if any(f[0] == field[0] for f in fields):
                raise ValueError("duplicate field declaration")
            fields.append(field)
            defaults.append(default)
        return tuple(fields), tuple(defaults), members
