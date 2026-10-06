from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration


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
        context.register(TypeDeclaration(node.identity, node.header[0].text, self.spelling,
                                         (), tuple(variants), (), node.span, context.source.path))
