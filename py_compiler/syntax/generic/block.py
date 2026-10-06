"""B: syntax providers own recognition, attachment, declarations and execution."""


from py_compiler.syntax.generic.grammar import Grammar, Match


class BlockProvider(Grammar):
    role = "B.header"

    def grammars(self):
        return (self,)

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
