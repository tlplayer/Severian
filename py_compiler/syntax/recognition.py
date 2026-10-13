from functools import cached_property
from dataclasses import dataclass
from py_compiler.syntax.keywords.keyword import KEYWORDS
from py_compiler.syntax.symbol.catalog import SYMBOLS


@dataclass(frozen=True)
class Syntax:
    pointer_bits: int = 64
    keywords: frozenset = KEYWORDS
    symbols: tuple = SYMBOLS
    include_tests: bool = True

    @cached_property
    def block_providers(self):
        from py_compiler.syntax.prelude import block_providers
        return block_providers()

    @cached_property
    def sentence_providers(self):
        from py_compiler.syntax.prelude import sentence_providers
        return sentence_providers()

    @cached_property
    def expression_providers(self):
        from py_compiler.syntax.prelude import expression_providers
        return expression_providers()

    @cached_property
    def scope_providers(self):
        from importlib import import_module
        from py_compiler.syntax.block.function import Function
        owners = (Function(), import_module('py_compiler.syntax.block.global').Global(),
                  import_module('py_compiler.syntax.block.class').Class())
        return {owner.scope_kind: owner for owner in owners}

    @cached_property
    def expansions(self):
        from py_compiler.syntax.grammar.expansion import standard_expansions
        return standard_expansions()

    @cached_property
    def grammar_catalog(self):
        from py_compiler.syntax.catalog import collect_grammars
        return collect_grammars(self)

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

    @cached_property
    def types(self):
        from py_compiler.syntax.prelude import type_definitions
        return type_definitions(self.pointer_bits)

    def recognize(self, text, cursor):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.generic.grammar import SourceWindow
        return self.registry.construct('Y', SourceWindow(SourceFile('<recognition>', text), cursor, len(text)))
