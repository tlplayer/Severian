"""Discover grammar from its defining objects; registry order never supplies meaning."""
from py_compiler.syntax.keywords.keyword import Keyword
from py_compiler.syntax.symbol.symbol import Symbol
from py_compiler.syntax.generic.owned import WordGrammar


def definitions(syntax):
    types = tuple({t.name: t for t in syntax.types.values()}.values())
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
