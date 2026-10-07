from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.function.contracts import signature, pure_predicate
from py_compiler.mir.ownership.flow import Binding


class Function(DeclarationProvider):
    spelling = "def"

    def parse_header(self, items):
        return tuple(items[1:-1] if items[-1].text == ":" else items[1:])

    def has_body(self, node):
        return node.tokens[-1].text == ":"

    def declare(self, node, context):
        self.register(node, context)

    def declare_member(self, node, context, receiver):
        self.register(node, context, receiver)

    def register(self, node, context, receiver=None):
        entry = signature(node, receiver.name if receiver else ".".join(context.scope))
        entry.scope = context.scope
        entry.receiver = receiver
        imported = []
        active = node.syntax
        while hasattr(active, "parent"):
            imported[0:0] = active.imports
            active = active.parent
        entry.imports = tuple(imported)
        if not self.has_body(node):
            raise ValueError("a non-trait callable requires a body; use unimplemented for a stub")
        for guard in entry.guards:
            pure_predicate(guard, {name for name, _ in entry.parameters})
        context.add_callable(entry, self)
        context.declare_nodes(node.children, (*context.scope, node.identity))
        return entry

    def compile(self, entry, context):
        source, syntax, node = context.source, context.bound_syntax, entry.node
        result = context.type(entry.result) if entry.result else None
        builder = context.builder(source, syntax, entry.symbol, entry.symbol, result)
        builder.imported = {name: value for imported in entry.imports for name, value in imported.exports}
        builder.context = context
        builder.declaration_scope = (*entry.scope, node.identity)
        builder.body.declaration = entry.name
        builder.declared_nodes = context.declared_nodes
        builder.namespaces["module"] = context.global_bindings
        builder.storage = dict(context.global_storage)
        builder.flow = context.initial_flow.fork()
        builder.fallback_scopes = [context.global_bindings]
        env, parameters = {}, []
        if entry.receiver:
            entry.receiver.configure(builder, node, parameters)
            builder.receiver = entry.receiver
            builder.receiver_value = parameters[0]
        builder.constructor = bool(entry.receiver and entry.name.rsplit(".", 1)[-1] == entry.receiver.name.rsplit(".", 1)[-1])
        builder.uninitialized_fields = {b.identity for b in builder.namespaces.get("self", {}).values()} if builder.constructor else set()
        for parameter, annotation in entry.parameters:
            type_ = context.type(annotation) if annotation else None
            if type_ and not type_.mlir:
                raise ValueError("parameter has no value representation")
            mode = entry.parameter_ownership.get(parameter, getattr(type_, 'parameter_ownership', 'view'))
            binding = Binding(node.identity + ":" + parameter, parameter, type_, False, mode, node.span)
            value = builder.value(type_)
            env[binding.name], builder.values[binding.identity] = binding, value
            builder.body.bindings.append(binding)
            parameters.append(value)
        builder.current.parameters = tuple(parameters)
        context.scope = builder.declaration_scope
        from py_compiler.syntax.block.phases import ContractPhases
        builder.phases = ContractPhases(entry, builder, env)
        builder.phases.enter()
        builder.scope(node.children, env, local_names=set(env))
        if builder.current.terminator is None:
            builder.phases.leave()
        if any(value.type is None for value in builder.body.blocks[0].parameters):
            raise ValueError("unused/unconstrained parameter requires a type annotation")
        if builder.uninitialized_fields and (builder.current.terminator is None or builder.current.terminator.kind != "panic"):
            raise ValueError("constructor must initialize every field")
        entry.body = builder.finish()
        entry.body.complexity = {key: {"name": value.name, "rank": int(value)} for key, value in entry.complexity.items()}
        entry.result = entry.body.result_type.name
        offset = 1 if entry.receiver else 0
        entry.parameters = tuple((name, value.type.name) for (name, _), value in
                                 zip(entry.parameters, entry.body.blocks[0].parameters[offset:]))
        context.bodies.append(entry.body)
        return entry.body
