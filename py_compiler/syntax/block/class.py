from py_compiler.syntax.block.record import RecordProvider, TypeDeclaration
from py_compiler.syntax.complex.object import ObjectType


class Class(RecordProvider):
    spelling = "class"
    scope_kind = 'self'

    def bindings(self, cfg, env):
        return cfg.namespaces.get('self', {})

    def parse_header(self, items):
        header = tuple(items[1:-1] if items[-1].text == ":" else items[1:])
        if not header or header[0].kind != "IDENTIFIER":
            raise ValueError("class requires a name")
        if len(header) > 1 and (header[1].text != ":" or any(t.text != "+" for t in header[3::2])):
            raise ValueError("expected class Name: or class Name: Trait")
        return header

    def declare(self, node, context):
        fields, defaults, members = self.fields(node, context)
        traits = tuple(context.declaration_name(t.text, context.scope) for t in node.header[2::2]) if len(node.header) > 1 else ()
        declaration = TypeDeclaration(node.identity, ".".join((*context.scope, node.header[0].text)), self.spelling,
                                      fields, (), traits, node.span, context.source.path, defaults)
        context.register(declaration)
        context.types[declaration.name] = ObjectType(declaration.name, "record", "!llvm.ptr", declaration=declaration)
        context.tags[declaration.name] = len(context.tags) + 1
        for trait_name in traits:
            context.require(trait_name, lambda provider, target: provider.satisfy(target, declaration))
        for member in members:
            member.provider.declare_member(member, context, Receiver(declaration))

    def layout(self, declaration, context):
        types = [context.type(name) for _, name in declaration.fields]
        if any(t.family not in ("integer", "float", "bool", "char", "byte", "pointer", "absence") or t.mlir == "index" for t in types):
            raise ValueError("class storage currently requires scalar LLVM-compatible fields")
        return "!llvm.struct<(" + ", ".join(t.mlir for t in types) + ")>"

    def field_place(self, value, field, cfg, span, write=False):
        from py_compiler.syntax.generic.storage import FieldPlace
        declaration = cfg.context.by_name[value.type.name]
        if field.startswith("__") or (write and field.startswith("_")):
            raise ValueError(f"field {field!r} is not externally {'writable' if write else 'readable'}")
        for index, (name, type_name) in enumerate(declaration.fields):
            if name == field:
                return FieldPlace(value, index, self.layout(declaration, cfg.context), cfg.context.type(type_name))
        raise ValueError(f"{declaration.name} has no field {field!r}")

    def instantiate(self, declaration, arguments, cfg, env, span):
        from py_compiler.syntax.complex.object import Allocate
        from py_compiler.syntax.generic.storage import FieldPlace
        from py_compiler.syntax.function.calls import invoke, literal_value, select
        layout = self.layout(declaration, cfg.context)
        value = cfg.emit("allocate", cfg.context.type(declaration.name), (), Allocate(layout), span)
        cfg.stack_values.add(value.identity)
        constructors = cfg.context.functions.get(declaration.name + "." + declaration.name.rsplit(".", 1)[-1], [])
        if constructors:
            entry, operands = select(constructors, arguments, cfg, env)
            invoke(entry, operands, cfg, span, value)
            return value
        if len(arguments) > len(declaration.fields):
            raise ValueError("too many constructor arguments")
        for index, ((_, type_name), default) in enumerate(zip(declaration.fields, declaration.defaults)):
            type_ = cfg.context.type(type_name)
            if index < len(arguments):
                field = cfg.expr(arguments[index], env, type_)
            elif default is not None:
                field = literal_value(default, type_, cfg, span)
            else:
                raise ValueError(f"constructor requires field {declaration.fields[index][0]!r}")
            FieldPlace(value, index, layout, type_).write(cfg, field, span)
        return value

    def copy_value(self, value, cfg, span):
        from py_compiler.syntax.complex.object import Allocate
        from py_compiler.syntax.generic.storage import FieldPlace
        declaration = cfg.context.by_name[value.type.name]
        layout = self.layout(declaration, cfg.context)
        result = cfg.emit("allocate", value.type, (), Allocate(layout), span)
        cfg.stack_values.add(result.identity)
        for index, (_, name) in enumerate(declaration.fields):
            type_ = cfg.context.type(name)
            field = FieldPlace(value, index, layout, type_).read(cfg, span)
            FieldPlace(result, index, layout, type_).write(cfg, field, span)
        return result


class Receiver:
    """Class syntax owns receiver layout and field access, not callable lowering."""
    def __init__(self, declaration):
        self.declaration = declaration
        self.name = declaration.name

    def configure(self, builder, node, parameters):
        from py_compiler.mir.ownership.flow import Binding
        from py_compiler.syntax.generic.storage import FieldPlace
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
