"""B: syntax providers own recognition, attachment, declarations and execution."""


from dataclasses import dataclass, field
from py_compiler.syntax.generic.grammar import Grammar, Match


@dataclass
class BlockScope:
    identity: str
    span: object
    owner: object
    parent: object = None
    children: list = field(default_factory=list)

    @property
    def kind(self):
        return self.owner.scope_kind

    @property
    def development(self):
        return self.owner.development_scope or bool(self.parent and self.parent.development)

    def contains_kind(self, kind):
        return self.kind == kind or bool(self.parent and self.parent.contains_kind(kind))

    def allows(self, dependency):
        return self.development or not dependency.development

    def at(self, symbol):
        """Resolve Y(span), or a span, to its innermost owning block."""
        span = getattr(symbol, 'span', symbol)
        if span.source != self.span.source or not self.span.start <= span.start <= span.end <= self.span.end:
            return None
        for child in self.children:
            found = child.at(span)
            if found is not None:
                return found
        return self


class BlockProvider(Grammar):
    role = "B.header"
    scope_kind = 'block'
    development_scope = False
    required_scope = None

    def bind_scope(self, node, parent=None):
        if self.required_scope and (parent is None or not parent.contains_kind(self.required_scope)):
            raise ValueError(f'{self.spelling} requires {self.required_scope} scope')
        scope = BlockScope(node.identity, node.span, self, parent)
        if parent is not None:
            parent.children.append(scope)
        return scope

    def grammars(self):
        from py_compiler.syntax.generic.owned import WordGrammar
        return (self, WordGrammar(self, self.spelling)) if self.spelling else (self,)

    def recognize(self, window):
        if not window.tokens or window.tokens[0].text != self.spelling:
            return None
        return Match(self, window, window.end, {"header": window.tokens})

    def construct(self, match):
        return self.parse_header(match.captures["header"])

    def expand(self, header, syntax):
        return syntax, ()

    spelling = ""
    attachment = None
    continuation = False
    indentation_unit = "    "

    def has_body(self, node):
        return True

    def parse_header(self, items):
        if items[-1].text != ":":
            raise ValueError(f"{self.spelling} header requires ':'")
        return tuple(items[1:-1])

    def attach(self, previous, parent):
        if self.continuation and (previous is None or getattr(previous.provider, "attachment", None) != self.attachment):
            raise ValueError(f"{self.spelling} requires a preceding compatible block")

    def indentation(self, prefix):
        unit = self.indentation_unit
        if not unit or prefix != unit * (len(prefix) // len(unit)):
            raise ValueError("indentation must contain complete block indentation units")
        return len(prefix)

    def declare(self, node, context):
        context.declare_nodes(node.children, (*context.scope, node.identity))

    def declare_member(self, node, context, owner):
        raise ValueError(f"{self.spelling} does not provide member declaration behavior")

    def lower(self, node, cfg, env, nodes, index):
        raise ValueError(f"no execution provider for {self.spelling}")


class DeclarationProvider(BlockProvider):
    def lower(self, node, cfg, env, nodes, index):
        if node.identity not in cfg.declared_nodes:
            raise ValueError("declaration has not been resolved in its owning scope")
        return 1
