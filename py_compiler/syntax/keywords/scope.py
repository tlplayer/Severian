"""Scope qualifiers select bindings; they do not introduce source blocks."""


class Local:
    def bindings(self, cfg, env):
        return {name: env[name] for name in cfg.local_names if name in env}


class Namespace:
    def __init__(self, name):
        self.name = name

    def bindings(self, cfg, env):
        return cfg.namespaces.get(self.name, {})


SCOPES = {"local": Local(), "module": Namespace("module"), "self": Namespace("self")}
