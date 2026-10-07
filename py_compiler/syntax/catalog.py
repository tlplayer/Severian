"""Discover grammar from its defining objects; registry order never supplies meaning."""


from py_compiler.syntax.prelude import definitions


def grammars(syntax):
    return tuple(grammar for owner in definitions(syntax) for grammar in owner.grammars())


def assignment_spellings(syntax):
    return {'=', ':='} | {grammar.spelling for grammar in grammars(syntax)
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
    def test_missing_definition_is_a_diagnostic(self):
        from py_compiler.syntax.complex.object import ObjectType
        with self.assertRaisesRegex(ValueError, "defining object"):
            ObjectType("Missing", "integer", "i32").grammars()

    def test_metadata_family_does_not_supply_behavior(self):
        from py_compiler.syntax.complex.object import ObjectType, NamedDefinition
        owner = ObjectType("Named", "integer", "i32", declaration=NamedDefinition("Named"))
        self.assertEqual([g.role for g in owner.grammars()], ["Y"])
