"""Enum owns its declaration block, variant identity, ordering and representation."""
from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration
from dataclasses import dataclass
from py_compiler.syntax.primitive.numeric.types import Integer
from py_compiler.syntax.primitive.numeric.grammar import scalar_grammars
from py_compiler.syntax.generic.owned import WordGrammar


@dataclass(frozen=True)
class EnumType(Integer):
    variants: tuple = ()

    def grammars(self):
        return (WordGrammar(self, self.name, "IDENTIFIER"), *scalar_grammars(self, arithmetic=False))

    def variant(self, name):
        if name not in self.variants:
            raise ValueError(f"unknown {self.name} variant {name!r}")
        return self.variants.index(name)

    def accept_literal(self, owner, value):
        if owner != self:
            raise ValueError(f"{self.name} requires a named variant")
        return value


class Enum(RecordProvider):
    spelling = "enum"

    def declare(self, node, context):
        variants = []
        for child in node.children:
            if child.provider or len(child.tokens) != 1 or child.tokens[0].kind != "IDENTIFIER":
                raise ValueError("enum provider requires payload-free named variants")
            name = child.tokens[0].text
            if name in variants:
                raise ValueError("duplicate enum variant")
            variants.append(name)
        context.register(TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                         (), tuple(variants), (), node.span, context.source.path))

        name = ".".join((*context.scope, node.header[0].text))
        context.types[name] = EnumType(name, "enum", "i32", 32, False, tuple(variants))
