from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration


class Trait(RecordProvider):
    spelling = "trait"

    def declare(self, node, context):
        fields, defaults, members = self.fields(node, context)
        if members:
            raise ValueError("trait callable contracts require a dispatch provider")
        context.register(TypeDeclaration(node.identity, node.header[0].text, self.spelling,
                                         fields, (), (), node.span, context.source.path, defaults))

    def satisfy(self, target, implementation):
        for field in target.fields:
            if field not in implementation.fields:
                raise ValueError(f"{implementation.name} does not satisfy {target.name}.{field[0]}: {field[1]}")
