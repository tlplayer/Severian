"""sev_py syntax prelude: imports owners; compiler discovery applies their grammars."""
from importlib import import_module


BLOCKS = (
    ('block.if', 'Conditional'), ('block.elif', 'Elif'), ('block.else', 'Else'),
    ('block.class', 'Class'), ('block.trait', 'Trait'), ('block.enum', 'Enum'),
    ('block.function', 'Function'), ('block.test', 'Test'),
    ('block.operator', 'Operator'), ('block.extend', 'Extend'), ('block.match', 'Match'),
    ('block.with', 'With'), ('block.unsafe', 'Unsafe'), ('block.select', 'Select'),
)
CLAUSES = (('keywords.prefix', 'Prefix'), ('keywords.fix', 'Fix'),
           ('keywords.suffix', 'Suffix'), ('keywords.defer', 'Defer'))
STATEMENTS = (('keywords.yield', 'Yield'), ('keywords.static', 'Static'), ('keywords.atomic', 'Atomic'))
CALLABLES = (('operator.async', 'Async'), ('operator.await', 'Await'), ('function.lambda', 'Lambda'))
DYNAMIC_TYPES = (('dynamic.dynamic', 'DynamicType'), ('dynamic.any', 'AnyType'),
                 ('dynamic.numeric', 'NumericType'))
COMPLEX_TYPES = (('complex.type', 'Type'), ('block.union', 'Union'), ('complex.tuple', 'TupleType'),
                 ('complex.function', 'FunctionType'), ('complex.reference', 'ReferenceType'),
                 ('complex.generic', 'GenericType'), ('complex.iterator', 'IteratorType'))
TYPES = (*DYNAMIC_TYPES, *COMPLEX_TYPES)
FIXTURES = (('block.accept', 'Accept'), ('block.reject', 'Reject'))


def instantiate(entries):
    return tuple(getattr(import_module('py_compiler.syntax.' + module), name)() for module, name in entries)


def block_providers():
    return {owner.spelling: owner for owner in instantiate(BLOCKS)}


def clause_providers():
    return {owner.name: owner for owner in instantiate(CLAUSES)}


def expression_providers():
    from py_compiler.syntax.operator.ownership import OPERATORS
    return {owner.name: owner for owner in (*instantiate(CALLABLES), *OPERATORS)}


def type_definitions(pointer_bits):
    from py_compiler.syntax.primitive.catalog import primitives
    from py_compiler.syntax.complex.string import STRING
    result = primitives(pointer_bits)
    result[STRING.name] = STRING
    from py_compiler.syntax.complex.array import Array
    result['array'] = Array()
    result.update((owner.name, owner) for owner in instantiate(TYPES))
    return result


def builtin_call_providers():
    from py_compiler.syntax.function.allocate import Allocate
    from py_compiler.syntax.function.deallocate import Deallocate
    return {'allocate': Allocate(), 'deallocate': Deallocate()}


def context_type_definitions():
    from py_compiler.syntax.complex.error import ErrorType
    from py_compiler.syntax.complex.big_o import declaration as complexity_type
    owners = (ErrorType(), complexity_type())
    return {owner.name: owner for owner in owners}


def compiler_fixtures():
    # Imported by test with compiler, not leaked into ordinary source scopes.
    return instantiate(FIXTURES)


def sentence_providers():
    entries = (('operator.unimplemented', 'Unimplemented'), ('operator.drop', 'Drop'))
    tail = (('grammar.call', 'Call'), ('grammar.assignment', 'Assignment'))
    return (*instantiate(entries), import_module('py_compiler.syntax.grammar.return').Return(),
            *instantiate(STATEMENTS), *instantiate(tail))


def definitions(syntax):
    from py_compiler.syntax.generic.owned import WordGrammar
    from py_compiler.syntax.keywords.keyword import Keyword
    types = tuple({t.name: t for t in syntax.types.values()}.values())
    from py_compiler.syntax.operator.ownership import OPERATORS
    from py_compiler.syntax.operator.drop import Drop
    owners = (*types, *OPERATORS, Drop(), *instantiate(CLAUSES), *instantiate(STATEMENTS), *instantiate(CALLABLES),
              *syntax.block_providers.values())
    words = set()
    for owner in owners:
        if not callable(getattr(owner, 'grammars', None)):
            raise ValueError(f'{owner.name} has no defining object supplying grammar')
        words.update(g.spelling for g in owner.grammars() if isinstance(g, WordGrammar))
    keywords = tuple(Keyword(word) for word in sorted(syntax.keywords - words))
    return (*owners, *keywords, *syntax.symbols)


import unittest


class PreludeTests(unittest.TestCase):
    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('prelude.sev', text, Syntax())

    def test_owner_words_are_unambiguous_and_preserve_identifiers(self):
        from py_compiler.syntax.recognition import Syntax
        syntax = Syntax()
        for word in ('async', 'await', 'lambda', 'prefix', 'fix', 'suffix', 'defer',
                     'operator', 'extend', 'match', 'with', 'unsafe', 'select',
                     'yield', 'static', 'atomic', 'dynamic', 'any', 'numeric'):
            self.assertEqual(syntax.recognize(word, 0)[1], len(word))
            identifier = word + '_value'
            self.assertEqual(syntax.recognize(identifier, 0), ('IDENTIFIER', len(identifier)))

    def test_existing_compound_assignment_still_reaches_atoms(self):
        result = self.compile('def amount() -> byte:\n    size = 1B\n    size += 2B\n    return size\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        operation = next(o for body in result.program.bodies for block in body.blocks for o in block.operations if o.kind == 'scalar')
        self.assertEqual(operation.atom.owner.name, 'byte')

    def test_phase_owners_attach_to_the_existing_execution_contract(self):
        result = self.compile('def checked(x: int) -> int with {prefix x > 0, fix x > 0, suffix x > 0}:\n    return x\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        assertions = [o for body in result.program.bodies for block in body.blocks for o in block.operations if o.kind == 'assert']
        self.assertGreaterEqual(len(assertions), 3)

    def test_compiler_fixtures_stay_scoped_and_use_separate_owners(self):
        from py_compiler.syntax.recognition import Syntax
        self.assertNotIn('accept', Syntax().block_providers)
        self.assertNotIn('reject', Syntax().block_providers)
        result = self.compile('test with compiler "fixtures":\n    accept:\n        x = 1\n    reject:\n        x: bool = 1\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_missing_lowering_is_an_owner_diagnostic(self):
        for text, expected in (
            ('def work():\n    pending = async work() with self\n', 'async callable has no declared'),
            ('def work():\n    fn = lambda x: x + 1\n', 'lambda has no declared'),
            ('value: dynamic = 1\n', 'dynamic has no declared'),
        ):
            result = self.compile(text)
            self.assertTrue(result.diagnostics)
            self.assertIn(expected, str(result.diagnostics[0]))
