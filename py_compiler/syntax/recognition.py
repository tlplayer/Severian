from dataclasses import dataclass
from py_compiler.syntax.keywords.keyword import KEYWORDS
from py_compiler.syntax.symbol.catalog import SYMBOLS


@dataclass(frozen=True)
class Syntax:
    pointer_bits: int = 64
    keywords: frozenset = KEYWORDS
    symbols: tuple = SYMBOLS

    @property
    def block_providers(self):
        from py_compiler.syntax.prelude import block_providers
        return block_providers()

    @property
    def sentence_providers(self):
        from py_compiler.syntax.prelude import sentence_providers
        return sentence_providers()

    @property
    def expression_providers(self):
        from py_compiler.syntax.prelude import expression_providers
        return expression_providers()

    @property
    def scope_providers(self):
        from py_compiler.syntax.keywords.scope import SCOPES
        return SCOPES

    @property
    def expansions(self):
        from py_compiler.syntax.grammar.expansion import standard_expansions
        return standard_expansions()

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

    @property
    def types(self):
        from py_compiler.syntax.prelude import type_definitions
        return type_definitions(self.pointer_bits)

    def recognize(self, text, cursor):
        from py_compiler.frontend.source.source import SourceFile
        from py_compiler.syntax.generic.grammar import SourceWindow
        return self.registry.construct('Y', SourceWindow(SourceFile('<recognition>', text), cursor, len(text)))
