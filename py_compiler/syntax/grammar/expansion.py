"""With expansions contribute imports and grammar to a bounded lexical scope."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GrammarImport:
    name: str
    grammars: tuple = ()
    exports: tuple = ()


class ScopedSyntax:
    def __init__(self, parent, imports):
        self.parent, self.imports = parent, tuple(imports)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def block_providers(self):
        result = dict(self.parent.block_providers)
        for imported in self.imports:
            for grammar in imported.grammars:
                if grammar.spelling in result:
                    raise ValueError(f'ambiguous imported grammar {grammar.spelling!r}')
                result[grammar.spelling] = grammar
        return result

    def recognition_order(self):
        from py_compiler.syntax.grammar.lexical import providers
        from py_compiler.syntax.function.grammar import ExpressionGrammar
        from py_compiler.syntax.grammar.blocks import SourceGrammar
        from py_compiler.syntax.grammar.literals import LiteralModuleGrammar
        return (*providers(self), ExpressionGrammar(self), SourceGrammar(self), LiteralModuleGrammar(self))

    @property
    def registry(self):
        from py_compiler.syntax.generic.grammar import Registry
        return Registry(self.recognition_order())


def visible_imports(syntax):
    result = {}
    chain = []
    while isinstance(syntax, ScopedSyntax):
        chain.append(syntax)
        syntax = syntax.parent
    for scope in reversed(chain):
        result.update((item.name, item) for item in scope.imports)
    return tuple(result.values())


def expand(syntax, names):
    inherited = visible_imports(syntax)
    known = {item.name for item in inherited}
    imports = []
    for name in names:
        if name in known:
            continue
        provider = syntax.expansions.get(name)
        if provider is None:
            raise ValueError(f'no grammar expansion supplies {name!r}')
        imports.append(provider())
        known.add(name)
    return ScopedSyntax(syntax, imports), (*inherited, *imports)


def compiler():
    from py_compiler.syntax.block.test import Fixture, Reject
    from py_compiler.frontend.src.lib import compile_source
    return GrammarImport('compiler', (Fixture(), Reject()), (('compile_source', compile_source),))


def integration():
    return GrammarImport('integ')


def standard_expansions():
    return {'compiler': compiler, 'integ': integration}
