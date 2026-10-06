from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration


class Class(RecordProvider):
    spelling = "class"

    def parse_header(self, items):
        header = tuple(items[1:-1] if items[-1].text == ":" else items[1:])
        if not header or header[0].kind != "IDENTIFIER":
            raise ValueError("class requires a name")
        if len(header) > 1 and (len(header) != 3 or header[1].text != ":"):
            raise ValueError("expected class Name: or class Name: Trait")
        return header

    def declare(self, node, context):
        fields, defaults, members = self.fields(node, context)
        traits = (node.header[2].text,) if len(node.header) > 1 else ()
        declaration = TypeDeclaration(node.identity, node.header[0].text, self.spelling,
                                      fields, (), traits, node.span, context.source.path, defaults)
        context.register(declaration)
        for trait_name in traits:
            context.require(trait_name, lambda provider, target: provider.satisfy(target, declaration))
        for member in members:
            context.defer(lambda member=member: member.provider.declare_member(member, context, Receiver(declaration)))


class Receiver:
    """Class syntax owns receiver layout and field access, not callable lowering."""
    def __init__(self, declaration):
        self.declaration = declaration
        self.name = declaration.name

    def configure(self, builder, node, parameters):
        from py_compiler.mir.ownership.flow import Binding
        from py_compiler.syntax.type.storage import FieldPlace
        syntax = builder.syntax
        self_value = builder.value(syntax.types["pointer"])
        parameters.append(self_value)
        builder.body.bindings.append(Binding(node.identity + ":self", "self", syntax.types["pointer"], True, "borrow", node.span))
        fields = {}
        field_types = [syntax.types[type_name] for _, type_name in self.declaration.fields]
        if any(t.family not in ("integer", "float", "bool", "char", "byte", "pointer", "absence") or t.mlir == "index" for t in field_types):
            raise ValueError("receiver layout requires scalar LLVM-compatible field providers")
        record_type = "!llvm.struct<(" + ", ".join(t.mlir for t in field_types) + ")>"
        for index, ((field_name, _), type_) in enumerate(zip(self.declaration.fields, field_types)):
            field = Binding(node.identity + ":self." + field_name, field_name, type_, False, "borrow", node.span)
            fields[field_name] = field
            builder.storage[field.identity] = FieldPlace(self_value, index, record_type, type_)
        builder.namespaces["self"] = fields
        builder.fallback_scopes.insert(0, fields)
