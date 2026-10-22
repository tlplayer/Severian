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
        from py_compiler.syntax.block.hooks import register_hook
        if register_hook(node,context):
            return None
        entry = signature(node, receiver.name if receiver else ".".join(context.scope))
        for name, _ in entry.parameters:
            context.check_prelude_conflict(name)
        entry.scope = context.scope
        entry.receiver = receiver
        if entry.receiver_ownership is not None and receiver is None:
            raise ValueError('self is only valid in a class method')
        entry.type_arguments = dict(context.template_types)
        from py_compiler.syntax.complex.foreign import register as register_foreign
        register_foreign(node, entry, context)
        if receiver:
            entry.type_arguments['Self'] = context.types[receiver.name]
            if not entry.templates:
                def concrete(annotation):
                    if annotation == 'Self':
                        return receiver.name
                    from py_compiler.syntax.block.trait import contract_annotation
                    return contract_annotation(annotation, context) if annotation else annotation
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
        if not entry.templates:
            for guard in entry.guards:
                pure_predicate(guard, dict(entry.parameters) | ({'self': receiver.name} if receiver else {}), context)
        context.add_callable(entry, self)
        context.declare_nodes(node.children, (*context.scope, node.identity))
        return entry

    def compile(self, entry, context):
        source, syntax, node = entry.node.tokens[0].lexeme.source or context.source, context.bound_syntax, entry.node
        for guard in entry.guards:
            pure_predicate(guard, dict(entry.parameters) | ({'self': entry.receiver.name} if entry.receiver else {}), context)
        result = context.type(entry.result) if entry.result else None
        builder = context.builder(source, syntax, entry.symbol, entry.symbol, result)
        builder.imported = {name: value for imported in entry.imports for name, value in imported.exports}
        builder.context = context
        builder.mock_cases = getattr(entry, "mock_cases", {})
        builder.mock_scope = getattr(entry, "mock_scope", node.identity)
        builder.operator_namespaces = tuple(node.decorators)
        builder.integration_test = getattr(entry, 'test_mode', '') == 'integ'
        builder.integration_scope = node.identity
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
        builder.inferred_parameters = {}
        builder.callable_entry = entry
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
            from py_compiler.syntax.generic.storage import owns_storage
            if (type_ and owns_storage(type_) and parameter not in entry.parameter_ownership
                    and not entry.infer_result_ownership and node.mlir_operation is None):
                mode = 'borrow'
                builder.inferred_parameters[parameter] = len(env)
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
        builder.profiling = getattr(entry, 'test_mode', '') == 'profile'
        if builder.profiling:
            from py_compiler.syntax.complex.profile import ProfileOperation
            builder.effect('profile-begin', (), ProfileOperation('begin'), node.span)
        builder.phases.enter()
        from py_compiler.syntax.block.hooks import Hooks
        builder.hooks=Hooks(entry,builder,env)
        builder.scope(node.children, env, local_names=set(env), cleanup=False)
        if builder.current.terminator is None:
            builder.hooks.finish()
            builder.phases.leave()
            builder.cleanup_memory(env.values(), node.span)
        if any(value.type is None for value in builder.body.blocks[0].parameters):
            raise ValueError("unused/unconstrained parameter requires a type annotation")
        if builder.uninitialized_fields and (builder.current.terminator is None or builder.current.terminator.kind != "panic"):
            raise ValueError("constructor must initialize every field")
        entry.body = builder.finish()
        for parameter, position in builder.inferred_parameters.items():
            if entry.parameter_ownership.get(parameter) == 'move':
                continue
            identity = node.identity + ':' + parameter
            mutated = any(mode in ('borrow', 'move') and
                          (identity in builder.value_origins.get(value.identity, ()) or
                           position in builder.value_sources.get(value.identity, ()))
                          for block in builder.body.blocks for operation in block.operations
                          for value, mode in zip(operation.operands, operation.contract.ownership))
            entry.parameter_ownership[parameter] = 'borrow' if mutated else 'view'
        if builder.inferred_parameters:
            from dataclasses import replace
            builder.body.bindings = [replace(binding, ownership=entry.parameter_ownership[binding.name])
                if binding.identity in builder.parameter_bindings and binding.name in builder.inferred_parameters
                else binding for binding in builder.body.bindings]
        if entry.receiver and entry.receiver_ownership is None:
            receiver_ids = {builder.self_binding.identity}
            receiver_ids.update(b.identity for b in builder.namespaces.get('self', {}).values())
            # Infer from checked operations, including transitive call contracts.
            # Constructors and recursive/unresolved calls stay exclusive.
            mutates_receiver = builder.constructor or any(
                mode in ('borrow', 'move') and
                (builder.value_origins.get(value.identity, set()) & receiver_ids or
                 -1 in builder.value_sources.get(value.identity, set()))
                for block in builder.body.blocks for operation in block.operations
                for value, mode in zip(operation.operands, operation.contract.ownership))
            entry.receiver_ownership = 'borrow' if mutates_receiver else 'view'
        entry.result_ownership = entry.body.result_ownership
        entry.body.complexity = {key: {"name": value.name, "rank": int(value)} for key, value in entry.complexity.items()}
        entry.result = entry.body.result_type.name
        offset = 1 if entry.receiver else 0
        entry.parameters = tuple((name, value.type.name) for (name, _), value in
                                 zip(entry.parameters, entry.body.blocks[0].parameters[offset:]))
        context.bodies.append(entry.body)
        return entry.body


class Macro(Function):
    """Typed macro bodies expand into the caller's control flow."""
    spelling = '->'

    def grammars(self):
        return (self,)

    def compile(self, entry, context):
        # A macro has no separately callable runtime body.
        return None

    def expand_call(self, entry, values, cfg, span):
        from types import SimpleNamespace
        from py_compiler.syntax.block.grammar import lower_result
        from py_compiler.syntax.complex.tensor import ArgumentValue
        if entry.node.identity in cfg.active_grammars:
            raise ValueError('recursive macro expansion')
        context = cfg.context
        previous = context.template_types, context.template_functions, context.scope, cfg.declaration_scope
        context.template_types = entry.type_arguments
        context.template_functions = entry.callable_arguments
        context.scope = entry.scope
        cfg.declaration_scope = entry.scope
        try:
            owners = {cfg.values[b.identity].identity: b for scope in cfg.active_scopes
                      for b in scope.values() if b.identity in cfg.values}
            captures = {name: ArgumentValue(value, owners.get(value.identity))
                        for (name, _), value in zip(entry.parameters, values)}
            provider = SimpleNamespace(node=entry.node, result=context.type(entry.result or 'absent'), env={})
            return lower_result(SimpleNamespace(provider=provider, captures=captures), cfg)
        finally:
            context.template_types, context.template_functions, context.scope, cfg.declaration_scope = previous


import unittest


class MacroExpansionTests(unittest.TestCase):
    def test_generic_macro_expands_without_a_runtime_call(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('macro.sev', '''-> twice[T](value: T) -> T:
    return value + value
def compute() -> int:
    return twice(21)
''', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertFalse(any('twice' in body.name for body in result.program.bodies))
        for body in result.program.bodies:
            body.verify()
            self.assertFalse(any(op.kind == 'call' and 'twice' in str(op.payload)
                                 for block in body.blocks for op in block.operations))
