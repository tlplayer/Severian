"""Registration only. Meaning lives in each implementing syntax provider."""
from importlib import import_module


def providers():
    declarations = (("if", "Conditional"), ("elif", "Elif"), ("else", "Else"),
                    ("class", "Class"), ("trait", "Trait"), ("enum", "Enum"),
                    ("function", "Function"), ("test", "Test"))
    instances = [getattr(import_module("py_compiler.syntax.block." + module), name)()
                 for module, name in declarations]
    return {provider.spelling: provider for provider in instances}
