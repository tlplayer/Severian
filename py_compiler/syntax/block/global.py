from py_compiler.syntax.generic.block import BlockProvider


class Global(BlockProvider):
    """The source window is the global block. No global: header exists."""
    def parse_header(self, items):
        raise ValueError("global is a scope qualifier; use global.name, not global:")

    def configure(self, cfg, context):
        from py_compiler.syntax.type.storage import GlobalPlace
        cfg.namespaces["global"] = context.global_bindings
        cfg.storage = context.global_storage
        cfg.storage_factory = lambda binding: GlobalPlace(binding.identity, binding.type)
