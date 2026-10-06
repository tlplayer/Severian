from hashlib import sha256
from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.mir.ownership.flow import Binding


class Function(DeclarationProvider):
    spelling = "def"

    def declare(self, node, context):
        context.defer(lambda: self.compile(node, context))

    def declare_member(self, node, context, receiver):
        self.compile(node, context, receiver)

    def compile(self, node, context, receiver=None):
        source, syntax = context.source, context.syntax
        header = list(node.header)
        if len(header) < 3 or header[0].kind != "IDENTIFIER" or header[1].text != "(":
            raise ValueError("expected def name(parameters) -> Type:")
        closing = next((i for i, t in enumerate(header) if t.text == ")"), -1)
        if closing < 2:
            raise ValueError("function parameter list is incomplete")
        result = None
        suffix = header[closing + 1:]
        if suffix:
            if len(suffix) != 2 or suffix[0].text != "->" or suffix[1].text not in syntax.types:
                raise ValueError("function result must name a primitive type")
            result = syntax.types[suffix[1].text]
        name = (receiver.name + "." if receiver else "") + header[0].text
        if name in context.callables:
            raise ValueError(f"duplicate callable {name!r}")
        context.callables.add(name)
        builder = context.builder(source, syntax, node.identity, "__sev_fn_" + sha256(node.identity.encode()).hexdigest(), result)
        builder.body.declaration = name
        builder.declared_nodes = context.declared_nodes
        builder.namespaces["global"] = context.global_bindings
        builder.storage = dict(context.global_storage)
        builder.flow = context.initial_flow.fork()
        builder.fallback_scopes = [context.global_bindings]
        env, parameters, position = {}, [], 2
        if receiver:
            receiver.configure(builder, node, parameters)
        while position < closing:
            parameter = header[position]
            if parameter.kind != "IDENTIFIER" or parameter.text in env:
                raise ValueError("parameter name is invalid or duplicate")
            position += 1
            type_ = None
            if position < closing and header[position].text == ":":
                if position + 1 >= closing or header[position + 1].text not in syntax.types:
                    raise ValueError("parameter annotation requires a resolved type")
                type_ = syntax.types[header[position + 1].text]
                position += 2
                if not type_.mlir:
                    raise ValueError("parameter has no value representation")
            binding = Binding(node.identity + ":" + parameter.text, parameter.text, type_, False, "own", parameter.span)
            value = builder.value(type_)
            env[binding.name], builder.values[binding.identity] = binding, value
            builder.body.bindings.append(binding)
            parameters.append(value)
            if position < closing:
                if header[position].text != ",":
                    raise ValueError("expected comma between parameters")
                position += 1
        builder.current.parameters = tuple(parameters)
        builder.scope(node.children, env, local_names=set(env))
        if any(value.type is None for value in builder.body.blocks[0].parameters):
            raise ValueError("unused/unconstrained parameter requires a type annotation")
        context.bodies.append(builder.finish())
