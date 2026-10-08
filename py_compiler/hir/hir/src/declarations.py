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
        self.declarations, self.bodies, self.pending, self.requirements = [], [], [], []
        self.by_name, self.provider_by_name = {}, {}
        self.active_provider = None
        self.callables, self.functions, self.compilers = [], {}, {}
        self.global_bindings, self.global_storage = {}, {}
        self.types = dict(syntax.types)
        from py_compiler.syntax.prelude import context_type_definitions
        self.types.update(context_type_definitions())
        self.bound_syntax = BoundSyntax(self)
        self.tags = {}
        self.scope = ()
        self.tests = []
        self.unsafe_depth = 0
        self.template_types, self.template_functions, self.realizations = {}, {}, {}
        self.scopes = {}

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
            if candidate in self.by_name:
                return candidate
        return name

    def lookup_functions(self, name, scope):
        if name in self.template_functions:
            return (self.template_functions[name],)
        for length in range(len(scope), -1, -1):
            key = ".".join((*scope[:length], name))
            if key in self.functions:
                return self.functions[key]
        return ()

    def declare_nodes(self, nodes, scope):
        previous = self.scope
        self.scope = scope
        try:
            for node in nodes:
                if node.provider and node.identity not in self.declared_nodes:
                    self.active_provider = node.provider
                    self.declared_nodes.add(node.identity)
                    node.provider.declare(node, self)
        finally:
            self.scope = previous

    def type(self, name):
        name = name.replace(" ", "")
        if name in self.template_types:
            return self.template_types[name]
        if "[" in name and name.endswith("]"):
            base, argument = name[:-1].split("[", 1)
            definition = self.types.get(base)
            specialize = getattr(definition, "specialize", None)
            if specialize is None:
                raise ValueError(f"{base} has no generic type provider")
            result = specialize(self.type(argument))
            self.types[result.name] = result
            return result
        if '|' in name:
            from py_compiler.syntax.block.union import Union
            union = Union().resolve(self.type(part.strip()) for part in name.split('|'))
            if union.name not in self.types:
                self.types[union.name] = union
            return self.types[union.name]
        for length in range(len(self.scope), -1, -1):
            candidate = ".".join((*self.scope[:length], name))
            if candidate in self.types:
                return self.types[candidate]
        if name not in self.types:
            raise ValueError(f"unresolved type {name!r}")
        return self.types[name]

    def register(self, declaration):
        if declaration.name in self.by_name or declaration.name in self.syntax.types:
            raise ValueError(f"duplicate declaration {declaration.name!r}")
        self.by_name[declaration.name] = declaration
        self.provider_by_name[declaration.name] = self.active_provider
        self.declarations.append(declaration)

    def add_callable(self, entry, provider):
        self.callables.append(entry)
        self.functions.setdefault(entry.name, []).append(entry)
        self.compilers[id(entry)] = provider
        self.declared_nodes.add(entry.node.identity)

    def compile(self, entry):
        if entry.templates:
            return None
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
            if name not in self.by_name:
                raise ValueError(f"unresolved declaration {name!r}")
            provider = self.provider_by_name[name]
            if not hasattr(provider, "satisfy"):
                raise ValueError(f"{name!r} has no contract satisfaction provider")
            operation(provider, self.by_name[name])
        self.requirements.clear()

    def resolve(self):
        self.resolve_contracts()
        for operation in self.pending:
            operation()
        for entry in self.callables:
            self.compile(entry)
