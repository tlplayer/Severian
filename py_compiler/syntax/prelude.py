"""sev_py syntax prelude: imports owners; compiler discovery applies their grammars."""
from importlib import import_module


BLOCKS = (
    ('block.mlir', 'Mlir'), ('block.import', 'FromImport'),
    ('block.for', 'For'), ('block.while', 'While'), ('block.import', 'Import'), ('block.if', 'Conditional'), ('block.elif', 'Elif'), ('block.else', 'Else'),
    ('block.class', 'Class'), ('block.trait', 'Trait'), ('block.enum', 'Enum'), ('block.union', 'UnionDeclaration'),
    ('block.function', 'Function'), ('block.function', 'Macro'), ('block.test', 'Test'), ('block.grammar', 'Grammar'),
    ('block.operator', 'Operator'), ('block.extend', 'Extend'), ('block.match', 'Match'), ('block.match', 'Case'),
    ('complex.error', 'Recover'), ('complex.error', 'Try'), ('complex.error', 'Catch'), ('block.hooks', 'Without'), ('block.with', 'With'), ('block.unsafe', 'Unsafe'), ('block.select', 'Select'),
)
CLAUSES = (('keywords.prefix', 'Prefix'), ('keywords.fix', 'Fix'),
           ('keywords.suffix', 'Suffix'), ('keywords.defer', 'Defer'))
STATEMENTS = (('keywords.yield', 'Yield'), ('keywords.static', 'Static'), ('keywords.atomic', 'Atomic'),
              ('block.while', 'Break'), ('block.while', 'Continue'))
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
    from dataclasses import replace
    result[STRING.name] = replace(STRING, pointer_bits=pointer_bits)
    from py_compiler.syntax.complex.array import Array
    result['array'] = Array()
    from py_compiler.syntax.complex.tensor import Tensor
    result['tensor'] = Tensor()
    from py_compiler.syntax.complex.units import quantity, UNITS
    result.update((owner.name, owner) for owner in (quantity(d) for d in set(v[0] for v in UNITS.values())))
    result.update((owner.name, owner) for owner in instantiate(TYPES))
    from py_compiler.syntax.complex.channel import Channel
    result.update({"channel": Channel(), "Channel": Channel("Channel")})
    from dataclasses import replace
    result['dynamic'] = replace(result['dynamic'], bits=pointer_bits)
    result['any'] = replace(result['any'], bits=pointer_bits)
    return result


def builtin_call_providers():
    from py_compiler.syntax.function.allocate import Allocate
    from py_compiler.syntax.function.deallocate import Deallocate
    from py_compiler.syntax.function.length import Length
    from py_compiler.syntax.function.layout import LayoutQuery, TargetQuery
    from py_compiler.syntax.complex.error import Panic
    from py_compiler.syntax.complex.iterator import Next
    from py_compiler.syntax.complex.type import ZipTypes
    return {'panic': Panic(), 'target': TargetQuery(), 'zip': ZipTypes(), 'next': Next(), 'allocate': Allocate(), 'deallocate': Deallocate(), 'len': Length(), 'size': Length(),
            **{name: LayoutQuery(name) for name in ('bytes', 'alignment', 'offset')}}


def context_type_definitions():
    from py_compiler.syntax.complex.error import ErrorType
    from py_compiler.syntax.complex.big_o import declaration as complexity_type
    owners = (ErrorType(), complexity_type())
    return {**{owner.name: owner for owner in owners}, 'E': owners[0]}


def install_library_prelude(context):
    """Register explicit aliases without resolving or building their providers."""
    from py_compiler.syntax.complex.generic import CallableAlias
    for name, provider, target in LIBRARY_ALIASES:
        if context.interface != provider:
            context.function_aliases[name] = CallableAlias(target, (), provider=provider, prelude=True)


LIBRARY_ALIASES = (('print', 'system.io', '__print'),)


def require_library(context, package_name):
    """Load a prelude provider when a resolved operation requires it."""
    from dataclasses import replace
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.frontend.lexer.lexer import lex
    importer = import_module('py_compiler.syntax.block.import').Import()
    available, visited = {}, set()

    def collect(packages):
        for alias, package in packages.items():
            identity = id(package)
            if identity in visited:
                continue
            visited.add(identity)
            available.setdefault(package['package']['name'], (alias, package))
        for package in packages.values():
            for alias, child in package.get('template-dependencies', {}).items():
                if id(child) not in visited:
                    collect({alias: child})

    collect(context.c_packages)
    if package_name not in available:
        resolver = getattr(context.c_packages, 'resolve_prelude', None)
        resolved = resolver(package_name) if resolver else None
        if resolved is not None:
            available[package_name] = resolved
    if package_name in available:
        preferred = {'core.string': 'string', 'system.io': 'io'}[package_name]
        alias, package = available[package_name]
        if alias in context.c_packages and context.c_packages[alias] is not package:
            alias = '__prelude_' + preferred
            while alias in context.c_packages and context.c_packages[alias] is not package:
                alias += '_'
        context.c_packages.setdefault(alias, package)
        if alias not in context.package_imports:
            token = lex(SourceFile('<prelude>', alias), context.syntax)[0]
            node = replace(context.root, header=(token,), provider=importer)
            previous = context.scope
            context.scope = ()
            try:
                importer.declare(node, context)
            finally:
                context.scope = previous
        return alias
    return None


def compiler_fixtures():
    # Imported by test with compiler, not leaked into ordinary source scopes.
    return instantiate(FIXTURES)


def sentence_providers():
    entries = (('complex.error', 'Throw'), ('operator.unimplemented', 'Unimplemented'), ('operator.drop', 'Drop'))
    tail = (('grammar.call', 'Call'), ('grammar.assignment', 'Assignment'))
    return (*instantiate(entries), import_module('py_compiler.syntax.grammar.return').Return(),
            import_module('py_compiler.syntax.keywords.defer').Defer(),
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
    def library_packages(self):
        def export(name, symbol, parameters, result):
            return {'name': name, 'symbol': symbol,
                    'parameters': [{'name': n, 'type': t, 'ownership': 'view'} for n, t in parameters],
                    'result': result, 'result-sources': []}
        strings = {'package': {'name': 'core.string'}, 'objects': ['string.o'], 'types': [],
                   'exports': [export('_equal', 'string_equal', [('left', 'string'), ('right', 'string')], 'bool')]}
        io = {'package': {'name': 'system.io'}, 'objects': ['io.o'], 'types': [],
              'template-dependencies': {'text': strings},
              'exports': [export('__print', 'io_print', [('value', 'string')], 'int')],
              'aliases': [{'name': 'print', 'target': '__print'}]}
        return {'console': io}

    def test_print_and_string_operations_need_no_source_import(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('automatic-prelude.sev',
            'test:\n    print("hello")\n    assert("a" == "a")\n',
            Syntax(), c_packages=self.library_packages())
        self.assertFalse(result.diagnostics, result.diagnostics)
        symbols = {getattr(operation.atom.implementation, 'symbol', None)
                   for body in result.program.bodies for block in body.blocks for operation in block.operations}
        self.assertIn('io_print', symbols)
        self.assertIn('string_equal', symbols)

    def test_conflicting_functions_imports_parameters_and_values_are_errors(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for source in (
            'def print(value: int) -> int:\n    return value + 1\n'
            'test:\n    assert(print(2) == 3)\n',
            'from console import print\ntest:\n    print("hello")\n',
            'def work(print: int) -> int:\n    return print\n',
            'test:\n    print = 3\n',
        ):
            with self.subTest(source=source):
                result = compile_source('prelude-shadowing.sev', source, Syntax(),
                                        c_packages=self.library_packages())
                self.assertTrue(result.diagnostics)
                self.assertIn('conflicts with the prelude', str(result.diagnostics[0]))

    def test_explicit_prelude_alias_preserves_provider_and_allows_local_name(self):
        source = ('print() as output()\n'
                  'def print(value: int) -> int:\n    return value + 1\n'
                  'test:\n    assert(print(2) == 3)\n    output("hello")\n')
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('renamed.sev', source, Syntax(), c_packages=self.library_packages())
        self.assertFalse(result.diagnostics, result.diagnostics)
        symbols = {getattr(operation.atom.implementation, 'symbol', None)
                   for body in result.program.bodies for block in body.blocks for operation in block.operations}
        self.assertIn('io_print', symbols)

    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('prelude.sev', text, Syntax())

    def test_unregistered_c_collections_are_rejected(self):
        from py_compiler.syntax.recognition import Syntax
        for name, arguments in (('vector', 'int,1'), ('dict', 'int,int'), ('map', 'int,int')):
            with self.subTest(collection=name):
                self.assertNotIn(name, Syntax().types)
                result = self.compile(f'def work():\n    value = {name}[{arguments}](1,2)\n')
                self.assertTrue(result.diagnostics)
                self.assertIn('callable does not declare template parameters', str(result.diagnostics[0]))

    def test_fixed_array_type_requires_allocate_interface(self):
        result = self.compile('def work(value: array[int,2]):\n    return\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        result = self.compile('def work():\n    value = array[int,2](1,2)\n')
        self.assertTrue(result.diagnostics)

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
        result = self.compile('def checked(x: int) -> int with {prefix x > 0, fix x > 0, suffix x > 0,}:\n    return x\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        assertions = [o for body in result.program.bodies for block in body.blocks for o in block.operations if o.kind == 'assert']
        self.assertGreaterEqual(len(assertions), 3)

    def test_compiler_fixtures_stay_scoped_and_use_separate_owners(self):
        from py_compiler.syntax.recognition import Syntax
        self.assertNotIn('accept', Syntax().block_providers)
        self.assertNotIn('reject', Syntax().block_providers)
        result = self.compile('test with compiler "fixtures":\n    accept:\n        x = 1\n    reject:\n        x: bool = 1\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_recursive_async_requires_a_complete_signature(self):
        result = self.compile('def work():\n    pending = async work() with self\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('recursive call requires a complete callable signature', str(result.diagnostics[0]))

    def test_lambda_is_available_without_an_import(self):
        result = self.compile('test:\n    fn = lambda x: x + 1\n    assert(fn(2) == 3)\n')
        self.assertFalse(result.diagnostics, result.diagnostics)
