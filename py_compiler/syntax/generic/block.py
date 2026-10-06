"""B: syntax providers own recognition, attachment, declarations and execution."""


class BlockProvider:
    spelling = ""
    attachment = None
    continuation = False
    indentation_unit = "    "

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
        return None

    def declare_member(self, node, context, owner):
        raise ValueError(f"{self.spelling} does not provide member declaration behavior")

    def lower(self, node, cfg, env, nodes, index):
        raise ValueError(f"no execution provider for {self.spelling}")


class DeclarationProvider(BlockProvider):
    def lower(self, node, cfg, env, nodes, index):
        if node.identity not in cfg.declared_nodes:
            raise ValueError("declaration has not been resolved in its owning scope")
        return 1
