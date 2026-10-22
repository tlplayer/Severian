"""Declaration identities and dependency scheduling; providers own semantics."""


class BoundSyntax:
    def __init__(self, context):
        self.context = context

    @property
    def types(self):
        result = dict(self.context.types)
        result.update(self.context.template_types)
        for length in range(len(self.context.scope) + 1):
            prefix = '.'.join(self.context.scope[:length])
            prefix = prefix + '.' if prefix else ''
            for name, type_ in self.context.types.items():
                if name.startswith(prefix) and '.' not in name[len(prefix):]:
                    result[name[len(prefix):]] = type_
        return result

    def __getattr__(self, name):
        return getattr(self.context.syntax, name)


class DeclarationContext:
    def __init__(self, source, syntax, builder):
        self.source, self.syntax, self.builder = source, syntax, builder
        self.interface = None
        self.declarations, self.bodies, self.pending, self.requirements = [], [], [], []
        self.by_name, self.provider_by_name = {}, {}
        self.active_provider = None
        self.callables, self.functions, self.compilers = [], {}, {}
        self.global_bindings, self.global_storage = {}, {}
        self.imported_globals = {}
        self.sentence_grammars = []
        self.types = dict(syntax.types)
        from py_compiler.syntax.prelude import context_type_definitions
        self.types.update(context_type_definitions())
        self.types["Error"].install(self)
        self.bound_syntax = BoundSyntax(self)
        self.tags = {}
        self.scope = ()
        self.tests = []
        self.unsafe_depth = 0
        self.template_types, self.template_functions, self.realizations = {}, {}, {}
        self.scopes = {}
        self.record_templates = {}
        self.realizing_records = set()
        self.function_aliases = {}
        self.resolving_aliases = set()

    def callable_scope(self, entry):
        scope = entry.node.scope
        for argument in entry.callable_arguments.values():
            dependency = self.callable_scope(argument)
            if dependency is not None and dependency.development:
                return dependency
        for argument in entry.type_arguments.values():
            declaration = self.by_name.get(argument.name)
            dependency = self.scopes.get(declaration.identity) if declaration else None
            if dependency is not None and dependency.development:
                return dependency
        return scope

    def declaration_name(self, name, scope):
        for length in range(len(scope), -1, -1):
            candidate = '.'.join((*scope[:length], name))
            if candidate in self.by_name or candidate in self.record_templates:
                return candidate
        return name

    def lookup_functions(self, name, scope):
        if name in self.template_functions:
            return (self.template_functions[name],)
        for length in range(len(scope), -1, -1):
            key = ".".join((*scope[:length], name))
            if key in self.functions:
                return self.functions[key]
            if key in self.function_aliases:
                if key in self.resolving_aliases:
                    raise ValueError('cyclic callable alias ' + key)
                self.resolving_aliases.add(key)
                try:
                    return self.function_aliases[key].resolve(self)
                finally:
                    self.resolving_aliases.remove(key)
        return ()

    def check_prelude_conflict(self, name):
        binding = self.function_aliases.get(name)
        if binding is not None and binding.prelude:
            raise ValueError(f'{name!r} conflicts with the prelude; rename its binding with an explicit alias')

    def declare_nodes(self, nodes, scope):
        previous = self.scope
        previous_provider = self.active_provider
        self.scope = scope
        try:
            from importlib import import_module
            import_module('py_compiler.syntax.block.with').register_packs(nodes, self)
            for node in nodes:
                if node.provider is None and any(t.text == 'as' for t in node.tokens):
                    from py_compiler.syntax.complex.generic import declare_alias
                    declare_alias(node, self)
                    self.declared_nodes.add(node.identity)
                if node.provider and node.identity not in self.declared_nodes:
                    self.active_provider = node.provider
                    self.declared_nodes.add(node.identity)
                    node.provider.declare(node, self)
        finally:
            self.scope = previous
            self.active_provider = previous_provider

    def type(self, name):
        name = name.replace(" ", "")
        if name.startswith('(') and '->' in name:
            from py_compiler.syntax.complex.function import parse_callable_type
            callable_ = parse_callable_type(name, self)
            if callable_ is not None:
                return callable_
        if '|' in name:
            from py_compiler.syntax.complex.generic import split_arguments
            members = split_arguments(name, '|')
            if len(members) > 1:
                from py_compiler.syntax.block.union import Union
                union = Union().resolve((self.type(member) for member in members), self)
                self.types.setdefault(union.name, union)
                return self.types[union.name]
        if name.startswith('(') and name.endswith(')'):
            from py_compiler.syntax.complex.generic import split_arguments
            from py_compiler.syntax.complex.tuple import product_type
            return product_type(tuple(self.type(t) for t in split_arguments(name[1:-1])), self)
        if name in self.template_types:
            return self.template_types[name]
        template_name = name.split('[', 1)[0]
        for length in range(len(self.scope), -1, -1):
            concrete = '.'.join((*self.scope[:length], name))
            if concrete in self.types and concrete not in self.record_templates:
                return self.types[concrete]
            candidate = '.'.join((*self.scope[:length], template_name))
            if candidate in self.record_templates:
                from py_compiler.syntax.complex.generic import split_arguments
                arguments = split_arguments(name.split('[', 1)[1][:-1]) if '[' in name else ()
                return self.record_templates[candidate].realize(arguments, self)
        # A declaration such as shared.List is a symbol, not a share modifier.
        for length in range(len(self.scope), -1, -1):
            candidate = '.'.join((*self.scope[:length], name))
            if candidate in self.types:
                return self.types[candidate]
        if name.startswith('share'):
            from py_compiler.syntax.complex.shared import Shared
            atomic = name.startswith('shareatomic')
            payload = self.type(name[len('shareatomic') if atomic else len('share'):])
            if not callable(getattr(payload, 'release', None)):
                raise ValueError('shared type requires an owning memory representation')
            return Shared(payload, self.type('usize').bits, atomic)
        if "[" in name and name.endswith("]"):
            base, argument = name[:-1].split("[", 1)
            definition = self.template_types.get(base, self.types.get(base))
            if callable(getattr(definition, 'realize', None)):
                from py_compiler.syntax.complex.generic import split_arguments
                return definition.realize(split_arguments(argument), self)
            specialize = getattr(definition, "specialize", None)
            if specialize is None:
                raise ValueError(f"{base} has no generic type provider")
            from py_compiler.syntax.complex.generic import split_arguments
            arguments = split_arguments(argument)
            resolver = getattr(definition, 'specialize_arguments', None)
            result = resolver(arguments, self) if resolver else specialize(*(self.type(item) for item in arguments))
            self.types[result.name] = result
            return result
        for length in range(len(self.scope), -1, -1):
            candidate = ".".join((*self.scope[:length], name))
            if candidate in self.types:
                return self.types[candidate]
        if name not in self.types:
            raise ValueError(f"unresolved type {name!r}")
        return self.types[name]

    def register(self, declaration):
        self.check_prelude_conflict(declaration.name)
        if declaration.name in self.by_name or declaration.name in self.syntax.types:
            raise ValueError(f"duplicate declaration {declaration.name!r}")
        self.by_name[declaration.name] = declaration
        self.provider_by_name[declaration.name] = self.active_provider
        self.declarations.append(declaration)

    def add_callable(self, entry, provider):
        if entry.receiver is None:
            self.check_prelude_conflict(entry.name.rsplit('.', 1)[-1])
        if entry.name in self.function_aliases:
            raise ValueError('callable declaration conflicts with alias ' + entry.name)
        entry.symbol_namespace = self.interface or ''
        origin = entry.node.tokens[0].lexeme.source
        if origin is not None and origin.identity != self.source.identity:
            entry.symbol_namespace += ':' + self.source.identity
        self.callables.append(entry)
        self.functions.setdefault(entry.name, []).append(entry)
        self.compilers[id(entry)] = provider
        self.declared_nodes.add(entry.node.identity)

    def compile(self, entry):
        if entry.templates or entry.variadic:
            return None
        from py_compiler.hir.thir.src.contracts import infer_result
        infer_result(entry, self)
        if entry.body is not None:
            return entry.body
        if entry.compiling:
            if not entry.result or any(annotation is None for _, annotation in entry.parameters):
                raise ValueError("recursive call requires a complete callable signature")
            return None
        entry.compiling = True
        previous = self.scope
        self.scope = entry.scope
        previous_types, previous_functions = self.template_types, self.template_functions
        self.template_types, self.template_functions = entry.type_arguments, entry.callable_arguments
        try:
            return self.compilers[id(entry)].compile(entry, self)
        finally:
            entry.compiling = False
            self.scope = previous
            self.template_types, self.template_functions = previous_types, previous_functions

    def require(self, name, operation):
        self.requirements.append((name, operation))

    def defer(self, operation):
        self.pending.append(operation)

    def resolve_contracts(self):
        for name, operation in self.requirements:
            if name == 'Error':
                continue
            if name not in self.by_name:
                raise ValueError(f"unresolved declaration {name!r}")
            provider = self.provider_by_name[name]
            if not hasattr(provider, "satisfy"):
                raise ValueError(f"{name!r} has no contract satisfaction provider")
            operation(provider, self.by_name[name])
        self.requirements.clear()

    def resolve(self):
        while getattr(self, 'pending_unions', []):
            pending, self.pending_unions = self.pending_unions, []
            previous = self.scope, self.active_provider
            for provider,node,scope in pending:
                self.scope,self.active_provider = scope,provider
                provider.declare(node,self)
            self.scope,self.active_provider = previous
            if len(self.pending_unions) >= len(pending):
                raise ValueError('unresolved forward union member')
        self.resolve_contracts()
        for operation in self.pending:
            operation()
        for entry in self.callables:
            if entry.receiver and any(name != 'Self' for name in entry.type_arguments):
                # Generic member bodies are obligations of actual calls, not
                # of merely constructing the containing specialization.
                continue
            self.compile(entry)
