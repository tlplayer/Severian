"""Declaration scheduling/indexing; syntax providers supply all declaration meaning."""


class DeclarationContext:
    def __init__(self, source, syntax, builder):
        self.source, self.syntax, self.builder = source, syntax, builder
        self.declarations, self.bodies, self.pending, self.requirements = [], [], [], []
        self.by_name, self.provider_by_name = {}, {}
        self.active_provider = None
        self.callables = set()
        self.global_bindings, self.global_storage = {}, {}

    def register(self, declaration):
        if declaration.name in self.by_name or declaration.name in self.syntax.types:
            raise ValueError(f"duplicate declaration {declaration.name!r}")
        self.by_name[declaration.name] = declaration
        self.provider_by_name[declaration.name] = self.active_provider
        self.declarations.append(declaration)

    def require(self, name, operation):
        self.requirements.append((name, operation))

    def defer(self, operation):
        self.pending.append(operation)

    def resolve(self):
        for name, operation in self.requirements:
            if name not in self.by_name:
                raise ValueError(f"unresolved declaration {name!r}")
            provider = self.provider_by_name[name]
            if not hasattr(provider, "satisfy"):
                raise ValueError(f"{name!r} has no contract satisfaction provider")
            operation(provider, self.by_name[name])
        for operation in self.pending:
            operation()
