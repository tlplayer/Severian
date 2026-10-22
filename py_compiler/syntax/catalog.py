"""Discover grammar from its defining objects; registry order never supplies meaning."""


from py_compiler.syntax.prelude import definitions


def grammars(syntax):
    # Only fixed lexical scopes cache. BoundSyntax resolves changing declarations.
    if getattr(type(syntax), 'grammar_catalog', None) is not None:
        return syntax.grammar_catalog
    return collect_grammars(syntax)


def collect_grammars(syntax):
    return tuple(grammar for owner in definitions(syntax) for grammar in owner.grammars())


def assignment_spellings(syntax):
    return {'=', ':=', '?='} | {grammar.spelling for grammar in grammars(syntax)
                         if grammar.role == 'O.assignment'}


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
    def test_grammar_cache_is_local_to_each_syntax_scope(self):
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.syntax.grammar.expansion import expand
        syntax = Syntax()
        scoped, _ = expand(syntax, ('test', 'compiler'))
        self.assertIs(grammars(syntax), grammars(syntax))
        self.assertIs(grammars(scoped), grammars(scoped))
        self.assertIsNot(grammars(scoped), grammars(syntax))
        self.assertTrue(any(getattr(g, 'spelling', None) == 'reject' for g in grammars(scoped)))
        self.assertFalse(any(getattr(g, 'spelling', None) == 'reject' for g in grammars(syntax)))

    def test_bound_syntax_still_discovers_new_types(self):
        from types import SimpleNamespace
        from py_compiler.hir.hir.src.declarations import BoundSyntax
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.syntax.primitive.numeric.types import Integer
        syntax = Syntax()
        context = SimpleNamespace(syntax=syntax, types=dict(syntax.types), template_types={}, scope=())
        bound = BoundSyntax(context)
        first = grammars(bound)
        context.types['i257'] = Integer('i257', 'integer', 'i257', 257, True)
        second = grammars(bound)
        self.assertGreater(len(second), len(first))

    def test_missing_definition_is_a_diagnostic(self):
        from py_compiler.syntax.complex.object import ObjectType
        with self.assertRaisesRegex(ValueError, "defining object"):
            ObjectType("Missing", "integer", "i32").grammars()

    def test_metadata_family_does_not_supply_behavior(self):
        from py_compiler.syntax.complex.object import ObjectType, NamedDefinition
        owner = ObjectType("Named", "integer", "i32", declaration=NamedDefinition("Named"))
        self.assertEqual([g.role for g in owner.grammars()], ["Y"])
