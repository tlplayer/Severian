"""Declaration identities and dependency scheduling; providers own semantics."""


class BoundSyntax:
    def __init__(self, context):
        self.context = context

    @property
    def types(self):
        result = dict(self.context.types)
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
        from py_compiler.syntax.type.objects import ObjectType
        self.types["BigO"] = ObjectType("BigO", "integer", "i32", bits=32, binding_ownership="copy")
        self.types["Error"] = ObjectType("Error", "error", "!llvm.ptr")
        self.bound_syntax = BoundSyntax(self)
        self.tags = {}
        self.scope = ()
        self.tests = []

    def declaration_name(self, name, scope):
        for length in range(len(scope), -1, -1):
            candidate = '.'.join((*scope[:length], name))
            if candidate in self.by_name:
                return candidate
        return name

    def lookup_functions(self, name, scope):
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
        if '|' in name:
            from py_compiler.syntax.type.objects import ObjectType
            variants = tuple(self.type(part.strip()) for part in name.split('|'))
            if any(not t.mlir or t.mlir.startswith('memref') for t in variants):
                raise ValueError('union variant has no LLVM value representation')
            canonical = ' | '.join(t.name for t in variants)
            if canonical not in self.types:
                self.types[canonical] = ObjectType(canonical, 'union', '!llvm.struct<(i32, ' + ', '.join(t.mlir for t in variants) + ')>', variants=variants)
            return self.types[canonical]
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
        if entry.body is not None:
            return entry.body
        if entry.compiling:
            if not entry.result or any(annotation is None for _, annotation in entry.parameters):
                raise ValueError("recursive call requires a complete callable signature")
            return None
        entry.compiling = True
        previous = self.scope
        self.scope = entry.scope
        try:
            return self.compilers[id(entry)].compile(entry, self)
        finally:
            entry.compiling = False
            self.scope = previous

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
