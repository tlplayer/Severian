from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration
from py_compiler.syntax.function.contracts import signature
from py_compiler.syntax.complex.object import ObjectType


class Trait(RecordProvider):
    spelling = "trait"

    def declare(self, node, context):
        fields, defaults, members = self.fields(node, context)
        self.context = context
        methods = tuple(signature(member, ".".join((*context.scope, node.header[0].text))) for member in members)
        for method in methods:
            if any(annotation is None for _, annotation in method.parameters):
                raise ValueError("trait parameter contracts require type annotations")
        context.register(TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                         fields, (), (), node.span, context.source.path, defaults, methods))
        name = ".".join((*context.scope, node.header[0].text))
        context.types[name] = ObjectType(name, "trait", "!llvm.struct<(!llvm.ptr, i32)>", declaration=context.by_name[name])

    def satisfy(self, target, implementation):
        for field in target.fields:
            if field not in implementation.fields:
                raise ValueError(f"{implementation.name} does not satisfy {target.name}.{field[0]}: {field[1]}")
        for method in target.methods:
            name = method.name.rsplit(".", 1)[-1]
            candidates = self.context.functions.get(implementation.name + "." + name, [])
            expected = tuple(self.context.type(t) for _, t in method.parameters)
            if not candidates:
                if method.node.children:
                    from py_compiler.syntax.block.function import Function
                    from importlib import import_module
                    Function().register(method.node, self.context, import_module("py_compiler.syntax.block.class").Receiver(implementation))
                    candidates = self.context.functions[implementation.name + "." + name]
                else:
                    raise ValueError(f"{implementation.name} is missing {target.name}.{name}")
            for entry in candidates:
                actual = tuple(self.context.type(t) for _, t in entry.parameters)
                if actual != expected or self.context.type(entry.result or "absent") != self.context.type(method.result or "absent"):
                    raise ValueError(f"{implementation.name}.{name} violates the trait callable contract")
