from py_compiler.syntax.generic.block import DeclarationProvider
from py_compiler.syntax.function.contracts import signature, pure_predicate
from py_compiler.mir.ownership.flow import Binding


class Function(DeclarationProvider):
    spelling = "def"
    scope_kind = 'local'

    def bindings(self, cfg, env):
        return {name: env[name] for name in cfg.local_names if name in env}

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
        if entry.receiver_ownership is not None and receiver is None:
            raise ValueError('self is only valid in a class method')
        entry.type_arguments = dict(context.template_types)
        if receiver:
            entry.type_arguments['Self'] = context.types[receiver.name]
            if not entry.templates:
                def concrete(annotation):
                    if annotation == 'Self':
                        return receiver.name
                    return context.type(annotation).name if annotation else annotation
                entry.parameters = tuple((name, concrete(annotation)) for name, annotation in entry.parameters)
                entry.result = concrete(entry.result)
        imported = []
        active = node.syntax
        while hasattr(active, "parent"):
            imported[0:0] = active.imports
            active = active.parent
        entry.imports = tuple(imported)
        if not self.has_body(node) and node.mlir_operation is None:
            raise ValueError("a non-trait callable requires a body; use unimplemented for a stub")
        for guard in entry.guards:
            pure_predicate(guard, {name for name, _ in entry.parameters} | ({'self'} if receiver else set()))
        context.add_callable(entry, self)
        context.declare_nodes(node.children, (*context.scope, node.identity))
        return entry

    def compile(self, entry, context):
        source, syntax, node = entry.node.tokens[0].lexeme.source or context.source, context.bound_syntax, entry.node
        result = context.type(entry.result) if entry.result else None
        builder = context.builder(source, syntax, entry.symbol, entry.symbol, result)
        builder.imported = {name: value for imported in entry.imports for name, value in imported.exports}
        builder.context = context
        builder.block_scope = context.callable_scope(entry)
        builder.declaration_scope = (*entry.scope, node.identity)
        builder.body.declaration = entry.name
        builder.body.result_ownership = entry.result_ownership
        contract = getattr(entry, 'semantic_result', None)
        if contract is not None:
            builder.body.result_sources = contract.sources
        builder.result_contract = entry if entry.infer_result_ownership else None
        builder.expected_failure_node = getattr(entry, 'expected_failure_node', None)
        builder.body.dispatch_guards = tuple(' '.join(token.text for token in guard.tokens) for guard in entry.guards)
        builder.body.parameter_defaults = {name: ' '.join(token.text for token in value.tokens)
                                           for name, value in entry.defaults.items()}
        builder.body.source_span = node.span
        builder.declared_nodes = context.declared_nodes
        builder.namespaces["module"] = context.global_bindings
        builder.storage = dict(context.global_storage)
        builder.flow = context.initial_flow.fork()
        builder.fallback_scopes = [context.global_bindings]
        env, parameters = {}, []
        if entry.receiver:
            builder.receiver_mode = entry.receiver_ownership or 'borrow'
            entry.receiver.configure(builder, node, parameters)
            builder.receiver = entry.receiver
            builder.receiver_value = parameters[0]
        receiver_declaration = entry.receiver.declaration if entry.receiver else None
        constructor_name = ((receiver_declaration.template_name or receiver_declaration.name.split('[', 1)[0]).rsplit('.', 1)[-1]
                            if receiver_declaration else None)
        builder.constructor = bool(entry.receiver and entry.name.rsplit('.', 1)[-1] == constructor_name)
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
            builder.parameter_bindings[binding.identity] = len(env) - 1
            if type_ and type_.binding_ownership != 'copy' and mode in ('view', 'borrow'):
                builder.value_sources[value.identity] = {len(env) - 1}
        builder.current.parameters = tuple(parameters)
        if node.mlir_operation is not None:
            from py_compiler.syntax.generic.atom import Atom
            from py_compiler.hir.thir.src.model import Terminator
            implementation = node.mlir_operation
            implementation.validate(tuple(v.type for v in parameters), result)
            modes = tuple(entry.parameter_ownership.get(n, 'view') for n, _ in entry.parameters)
            atom = Atom(entry, tuple(v.type for v in parameters), result if result and result.mlir else None,
                        modes, ('IO', 'mutation'), implementation)
            value = builder.emit('mlir', result, tuple(parameters), atom, node.span) if result and result.mlir else None
            if value is None:
                builder.effect('mlir', tuple(parameters), atom, node.span)
            builder.current.terminator = Terminator('finish', value=value, span=node.span)
            entry.body = builder.finish()
            entry.result = entry.body.result_type.name
            context.bodies.append(entry.body)
            return entry.body
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
        entry.result_ownership = entry.body.result_ownership
        entry.body.complexity = {key: {"name": value.name, "rank": int(value)} for key, value in entry.complexity.items()}
        entry.result = entry.body.result_type.name
        offset = 1 if entry.receiver else 0
        entry.parameters = tuple((name, value.type.name) for (name, _), value in
                                 zip(entry.parameters, entry.body.blocks[0].parameters[offset:]))
        context.bodies.append(entry.body)
        return entry.body
