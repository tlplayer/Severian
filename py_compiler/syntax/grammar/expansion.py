"""With expansions contribute imports and grammar to a bounded lexical scope."""
from functools import cached_property
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

    @cached_property
    def block_providers(self):
        result = dict(self.parent.block_providers)
        for imported in self.imports:
            for grammar in imported.grammars:
                key = grammar.spelling or type(grammar)
                if key in result:
                    raise ValueError(f'ambiguous imported grammar {grammar.spelling!r}')
                result[key] = grammar
        return result

    @cached_property
    def grammar_catalog(self):
        from py_compiler.syntax.catalog import collect_grammars, grammars
        selected = tuple(grammar for grammar in grammars(self.parent)
                         if getattr(grammar, 'lexical_namespace', None) is not None)
        return (*collect_grammars(self), *selected)

    def recognition_order(self):
        from py_compiler.syntax.grammar.lexical import providers
        from py_compiler.syntax.function.grammar import ExpressionGrammar
        from py_compiler.syntax.grammar.blocks import SourceGrammar
        from py_compiler.syntax.grammar.literals import LiteralModuleGrammar
        return (*providers(self), ExpressionGrammar(self), SourceGrammar(self), LiteralModuleGrammar(self))

    @cached_property
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
    from py_compiler.syntax.prelude import compiler_fixtures
    from py_compiler.frontend.src.lib import compile_source
    return GrammarImport('compiler', compiler_fixtures(), (('compile_source', compile_source),))


def integration():
    return GrammarImport('integ')


def chaos():
    from py_compiler.syntax.complex.chaos import ChaosAdd
    return GrammarImport('chaos', (ChaosAdd(),))


def benchmark():
    return GrammarImport('bench')


def test():
    from py_compiler.syntax.block.mock import Mock
    from py_compiler.syntax.block.throws import Throws
    from py_compiler.syntax.block.when import When
    from py_compiler.syntax.function.assertion import Assertion, Snapshot, AssertionBlock
    return GrammarImport('test', (Mock(), Throws(), When(), AssertionBlock()), (*utility().exports, ('expect', Assertion()), ('snapshot', Snapshot())))


def utility():
    from py_compiler.syntax.function.assertion import Assertion
    return GrammarImport('util', exports=(('assert', Assertion()),))


def standard_expansions():
    return {'compiler': compiler, 'integ': integration, 'bench': benchmark, 'chaos': chaos, 'test': test, 'util': utility}
