"""Discover grammar from its defining objects; registry order never supplies meaning."""
from py_compiler.syntax.keywords.keyword import Keyword
from py_compiler.syntax.symbol.symbol import Symbol
from py_compiler.syntax.generic.owned import WordGrammar


def definitions(syntax):
    types = tuple({t.name: t for t in syntax.types.values()}.values())
    for owner in types:
        if not callable(getattr(owner, 'grammars', None)):
            raise ValueError(f"{owner.name} has no defining object supplying grammar")
    owned_words = {g.spelling for owner in types for g in owner.grammars() if isinstance(g, WordGrammar)}
    keywords = tuple(Keyword(word) for word in sorted(syntax.keywords - owned_words))
    symbols = tuple(Symbol(word, tuple(other for other in syntax.symbols if len(other) > len(word) and other.startswith(word))) for word in syntax.symbols)
    return (*types, *keywords, *symbols, *syntax.block_providers.values())


def grammars(syntax):
    return tuple(grammar for owner in definitions(syntax) for grammar in owner.grammars())


def operator_bindings(syntax):
    result = {}
    for grammar in grammars(syntax):
        if grammar.role != 'F.operator':
            continue
        previous = result.get(grammar.spelling)
        if previous and (previous.precedence, previous.associativity) != (grammar.precedence, grammar.associativity):
            raise ValueError(f'conflicting binding contracts for {grammar.spelling}')
        result[grammar.spelling] = grammar
    return result


import unittest


class DefinitionTests(unittest.TestCase):
    def test_missing_definition_is_a_diagnostic(self):
        from py_compiler.syntax.type.objects import ObjectType
        with self.assertRaisesRegex(ValueError, "defining object"):
            ObjectType("Missing", "integer", "i32").grammars()

    def test_metadata_family_does_not_supply_behavior(self):
        from py_compiler.syntax.type.objects import ObjectType, NamedDefinition
        owner = ObjectType("Named", "integer", "i32", declaration=NamedDefinition("Named"))
        self.assertEqual([g.role for g in owner.grammars()], ["Y"])
