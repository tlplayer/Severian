from py_compiler.syntax.generic.block import BlockProvider


class Global(BlockProvider):
    """The source window is the global block. No module: header exists."""
    scope_kind = 'module'

    def bindings(self, cfg, env):
        return cfg.namespaces.get('module', {})

    def parse_header(self, items):
        raise ValueError("module is a scope qualifier; use module.name, not module:")

    def configure(self, cfg, context):
        from py_compiler.syntax.generic.storage import GlobalPlace
        cfg.namespaces["module"] = context.global_bindings
        cfg.storage = context.global_storage
        import json
        cfg.storage_factory = lambda binding: GlobalPlace(
            json.dumps((context.interface, context.source.path, binding.name)), binding.type,
            exported_name=binding.name if context.interface else '', constant=binding.constant)
