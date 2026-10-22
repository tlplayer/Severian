"""Generic templates bind semantic arguments before compiling a realization."""
from py_compiler.syntax.block.record import implements
from dataclasses import dataclass, replace
import re
from py_compiler.syntax.generic.definition import TypeDefinition


class GenericType(TypeDefinition):
    def __init__(self):
        super().__init__('generic')


def split_arguments(text, separator=','):
    depth, start, result = 0, 0, []
    for index, character in enumerate(text):
        depth += (character in '[(') - (character in '])')
        if character == separator and depth == 0:
            result.append(text[start:index])
            start = index + 1
    result.append(text[start:])
    if depth or any(not item for item in result):
        raise ValueError('invalid generic argument list')
    return tuple(result)


@dataclass(frozen=True)
class NumberArgument:
    value: int

    @property
    def name(self):
        return str(self.value)


def named_arguments(parameters, arguments, context=None):
    names = [p.name for p in parameters]
    if len(arguments) > len(names):
        raise ValueError('too many generic arguments')
    result = {}
    for index, spelling in enumerate(arguments):
        depth, separator = 0, None
        for position, character in enumerate(spelling):
            if context is not None and spelling in context.types:
                break
            if character in ':=' and depth == 0:
                separator = position
                break
            depth += (character in '[(') - (character in '])')
        key, value = (spelling[:separator], spelling[separator+1:]) if separator is not None else (names[index], spelling)
        if key not in names or key in result or not value:
            raise ValueError('unknown or duplicate generic argument ' + key)
        result[key] = value
    return result


def template_argument(spelling, context):
    if spelling.startswith('tuple(') or spelling.startswith('(') and not re.search(r'[A-Za-z_]', spelling.replace('None','')):
        from py_compiler.syntax.complex.tensor import ShapeArgument, parse_shape
        return ShapeArgument(parse_shape(spelling))
    candidate = context.declaration_name(spelling, context.scope)
    if '[' not in spelling and candidate in context.record_templates:
        return context.record_templates[candidate]
    return NumberArgument(int(spelling)) if spelling.isdigit() else context.type(spelling)


@dataclass(frozen=True)
class TemplateParameter:
    name: str
    default: str | None = None
    constraint: str | None = None


def template_header(items):
    """Consume the optional [parameters] immediately following a declaration name."""
    if len(items) < 2 or items[1].text != '[':
        return (), items
    depth, groups, group, end = 0, [], [], None
    for index in range(2, len(items)):
        token = items[index]
        if token.text == ']' and depth == 0:
            if group:
                groups.append(group)
            end = index
            break
        if token.text == ',' and depth == 0:
            groups.append(group)
            group = []
        else:
            group.append(token)
            depth += (token.text == '[') - (token.text == ']')
    if end is None or not groups:
        raise ValueError('invalid template parameter list')
    result = []
    for tokens in groups:
        if not tokens or tokens[0].kind != 'IDENTIFIER' or any(p.name == tokens[0].text for p in result):
            raise ValueError('invalid or duplicate template parameter')
        words = [token.text for token in tokens]
        equal = words.index('=') if '=' in words else len(words)
        constraint = None
        if equal > 1:
            if words[1] != ':' or equal == 2:
                raise ValueError('expected template parameter, constraint, or default')
            constraint = ''.join(words[2:equal])
        default = ''.join(words[equal + 1:]) if equal < len(words) else None
        if equal < len(words) and not default:
            raise ValueError('template default is missing')
        result.append(TemplateParameter(words[0], default, constraint))
    return tuple(result), [items[0], *items[end + 1:]]


def substitute(annotation, types):
    if annotation is None:
        return None
    return re.sub(r'(?<![\w.])[A-Za-z_]\w*\b', lambda m: types[m[0]].name if m[0] in types else m[0], annotation)


def check_constraints(parameters, bindings, context):
    for parameter in parameters:
        if not parameter.constraint:
            continue
        actual = bindings.get(parameter.name)
        if actual is None:
            raise ValueError(f'unbound constrained parameter {parameter.name}')
        for requirement in parameter.constraint.split('+'):
            required = substitute(requirement, bindings)
            if isinstance(actual, NumberArgument):
                if required not in ('int', 'usize', 'N') or actual.value < 0:
                    raise ValueError(f'{parameter.name} does not satisfy {required}')
                continue
            contract = context.type(required)
            from py_compiler.syntax.block.record import RecordTemplate
            if isinstance(actual, RecordTemplate):
                actual = realize_provider(actual, contract, context)
            if actual == contract:
                continue
            if callable(getattr(contract, 'accepts_type', None)) and contract.accepts_type(actual):
                continue
            declaration = context.by_name.get(actual.name)
            if contract.name == 'Error' and declaration is not None and implements(declaration, contract.name, context):
                continue
            from py_compiler.syntax.complex.collection import primitive_contract
            if contract.family == 'trait' and primitive_contract(actual, contract, context):
                continue
            if contract.family != 'trait' or declaration is None or not implements(declaration, contract.name, context):
                raise ValueError(f'{actual.name} does not satisfy {required}')
            context.provider_by_name[contract.name].satisfy(context.by_name[contract.name], declaration, context)


def realize_provider(provider, contract, context):
    """Infer a provider's own parameters from the contract it declares.

    Slot[T]: C[int,T] binds Slot's T from C's value argument, not C's key.
    Generic provider arity is independent of its implemented trait's arity.
    """
    from py_compiler.syntax.block.record import record_header
    from types import SimpleNamespace
    parameters = {p.name for p in provider.parameters}
    possibilities = []
    previous = context.scope
    context.scope = provider.scope
    try:
        pending = [(pattern, provider.scope) for pattern in record_header(provider.node.header)[2]]
        visited = set()
        while pending:
            pattern, scope = pending.pop()
            if (pattern, scope) in visited:
                continue
            visited.add((pattern, scope))
            context.scope = scope
            base = pattern.split('[', 1)[0]
            qualified = context.declaration_name(base, scope)
            declaration = context.record_templates.get(qualified)
            identity = declaration.name if declaration is not None else qualified
            terms = split_arguments(pattern.split('[', 1)[1][:-1]) if '[' in pattern else ()
            if identity != (getattr(contract.declaration, 'template_name', '') or contract.name):
                if declaration is not None and declaration.provider.spelling == 'trait':
                    supplied = named_arguments(declaration.parameters, terms, context)
                    symbolic = {}
                    for parameter in declaration.parameters:
                        term = supplied.get(parameter.name, parameter.default)
                        if term is None:
                            raise ValueError('missing provider trait argument ' + parameter.name)
                        term = substitute(term, symbolic)
                        # Concrete arguments retain their source scope while
                        # symbolic provider parameters pass through trait layers.
                        if not any(re.search(r'(?<![\w.])' + re.escape(p) + r'\b', term) for p in parameters):
                            term = template_argument(term, context).name
                        symbolic[parameter.name] = SimpleNamespace(name=term)
                    parents = list(record_header(declaration.node.header)[2])
                    parents.extend(''.join(t.text for t in child.tokens)
                        for child in declaration.node.children if not child.provider and child.tokens
                        and all(t.text not in (':', '=', ':=') for t in child.tokens))
                    pending.extend((substitute(parent, symbolic), declaration.scope) for parent in parents)
                elif qualified in context.by_name:
                    pending.extend((parent, scope) for parent in context.by_name[qualified].traits)
                continue
            inferred = {}

            def bind(pattern, actual):
                if pattern in parameters:
                    if pattern in inferred and inferred[pattern] != actual:
                        raise ValueError('conflicting provider type arguments for ' + pattern)
                    inferred[pattern] = actual
                    return
                if '[' in pattern and pattern.endswith(']'):
                    base, arguments = pattern[:-1].split('[', 1)
                    actual_base = getattr(getattr(actual, 'declaration', None), 'template_name', '') or actual.family
                    expected = context.declaration_name(base, scope)
                    template = context.record_templates.get(expected)
                    if (template.name if template is not None else expected) != actual_base:
                        raise ValueError('provider contract type pattern does not match')
                    terms = split_arguments(arguments)
                    concrete = getattr(actual, 'generic_arguments', ())
                    if len(terms) != len(concrete):
                        raise ValueError('provider contract argument count mismatch')
                    for term, argument in zip(terms, concrete):
                        bind(term, argument)
                    return
                if template_argument(pattern, context) != actual:
                    raise ValueError('provider contract has an incompatible fixed argument')

            arguments = getattr(contract, 'generic_arguments', ())
            if len(terms) != len(arguments):
                continue
            for term, argument in zip(terms, arguments):
                bind(term, argument)
            if all(p.name in inferred or p.default is not None for p in provider.parameters):
                possibilities.append(tuple(inferred[p.name].name if p.name in inferred else substitute(p.default, inferred)
                                           for p in provider.parameters))
    finally:
        context.scope = previous
    possibilities = list(dict.fromkeys(possibilities))
    if len(possibilities) != 1:
        raise ValueError('generic provider requires one inferable implementation of ' + contract.name)
    return provider.realize(possibilities[0], context)


def type_parameters(entry):
    words = set(re.findall(r'\b[A-Za-z_]\w*\b', ' '.join(t or '' for _, t in entry.parameters) + ' ' + (entry.result or '')))
    typed = {p.name for p in entry.templates if p.name in words}
    typed.update(p.name for p in entry.templates if p.constraint)
    constraint_words = set(re.findall(r'\b[A-Za-z_]\w*\b', ' '.join(p.constraint or '' for p in entry.templates)))
    typed.update(p.name for p in entry.templates if p.name in constraint_words)
    changed = True
    while changed:
        previous = set(typed)
        for parameter in entry.templates:
            if parameter.name in typed and parameter.default:
                names = set(re.findall(r'\b[A-Za-z_]\w*\b', parameter.default))
                typed.update(p.name for p in entry.templates if p.name in names)
        changed = typed != previous
    return typed


def explicit_bindings(entry, explicit, context):
    if len(explicit) > len(entry.templates):
        raise ValueError('too many template arguments')
    types, functions = {}, {}
    typed = type_parameters(entry)
    for parameter, spelling in zip(entry.templates, explicit):
        if parameter.name in typed:
            types[parameter.name] = template_argument(spelling, context)
        else:
            functions[parameter.name] = resolve_callable(spelling, context, context.scope)
    return types, functions


def resolve_callable(name, context, scope):
    functions = context.lookup_functions(name, scope)
    if len(functions) != 1:
        raise ValueError(f'callable template argument {name!r} must resolve to exactly one declaration')
    return functions[0]


def realize(entry, explicit, values, context):
    types, functions = explicit_bindings(entry, explicit, context)
    typed = type_parameters(entry)
    def infer(pattern, actual):
        if pattern in typed:
            if pattern in types and types[pattern] != actual:
                raise ValueError(f'conflicting type arguments for {pattern}')
            types[pattern] = actual
        elif pattern and '[' in pattern and pattern.endswith(']'):
            base, inner = pattern[:-1].split('[', 1)
            actual_base = getattr(getattr(actual, 'declaration', None), 'template_name', '')
            if getattr(actual, 'family', None) != base and actual_base.rsplit('.', 1)[-1] != base.rsplit('.', 1)[-1]:
                raise ValueError('generic container argument does not match its type pattern')
            terms = split_arguments(inner)
            actual_arguments = getattr(actual, 'generic_arguments', None)
            if actual_arguments is None:
                actual_arguments = (actual.element,) if hasattr(actual, 'element') else ()
            if len(terms) > len(actual_arguments):
                raise ValueError('generic container argument count mismatch')
            for term, argument in zip(terms, actual_arguments):
                infer(term, argument)
        elif pattern and pattern.isdigit() and getattr(actual, 'name', None) != pattern:
            raise ValueError('generic element count mismatch')
    for (_, pattern), value in zip(entry.parameters, values):
        infer(pattern, value.type)
    for parameter in entry.templates:
        actual = types.get(parameter.name)
        if actual is not None and parameter.constraint:
            from py_compiler.syntax.complex.collection import indexed_arguments
            indexed = indexed_arguments(actual, context)
            if indexed is not None:
                for requirement in parameter.constraint.split('+'):
                    if '[' not in requirement or not requirement.endswith(']'):
                        continue
                    base, inner = requirement[:-1].split('[', 1)
                    leaf = base.rsplit('.', 1)[-1]
                    arguments = indexed if leaf in ('C', 'Keyed', 'Indexable') else indexed[1:] if leaf in ('Iterable', 'Sequence') else ()
                    patterns = split_arguments(inner)
                    if len(arguments) == len(patterns):
                        for pattern, argument in zip(patterns, arguments):
                            infer(pattern, argument)
        declaration = getattr(actual, 'declaration', None)
        if declaration is not None and parameter.constraint:
            pending = list(getattr(declaration, 'traits', ()))
            visited = set()
            while pending:
                name = pending.pop()
                if name in visited:
                    continue
                visited.add(name)
                contract = context.type(name)
                for requirement in parameter.constraint.split('+'):
                    base = requirement.split('[', 1)[0].rsplit('.', 1)[-1]
                    if getattr(contract.declaration, 'template_name', '').rsplit('.', 1)[-1] == base:
                        infer(requirement, contract)
                pending.extend(getattr(contract.declaration, 'traits', ()))
    previous_scope, previous_types = context.scope, context.template_types
    context.scope = entry.scope
    context.template_types = dict(types)
    try:
        for parameter in entry.templates:
            if parameter.name in types or parameter.name in functions:
                continue
            if parameter.default is None:
                raise ValueError(f'cannot infer template argument {parameter.name}')
            if parameter.name in typed:
                types[parameter.name] = template_argument(substitute(parameter.default, types), context)
                context.template_types[parameter.name] = types[parameter.name]
            else:
                functions[parameter.name] = functions.get(parameter.default) or resolve_callable(parameter.default, context, entry.scope)
    finally:
        context.scope, context.template_types = previous_scope, previous_types
    previous_scope = context.scope
    context.scope = entry.scope
    try:
        check_constraints(entry.templates, types, context)
    finally:
        context.scope = previous_scope
    key = tuple((p.name, 'type', types[p.name].name) if p.name in types else
                (p.name, 'callable', functions[p.name].symbol) for p in entry.templates)
    from py_compiler.hir.thir.src.lib import type_identity, callable_identity
    cache_key = (entry.symbol, tuple((p.name, 'type', type_identity(types[p.name])) if p.name in types else
                                   (p.name, 'callable', callable_identity(functions[p.name])) for p in entry.templates))
    if cache_key in context.realizations:
        return context.realizations[cache_key]
    result = replace(entry, parameters=tuple((n, substitute(t, types)) for n, t in entry.parameters),
                     result=substitute(entry.result, types), templates=(), type_arguments={**entry.type_arguments, **types},
                     callable_arguments=functions, realization_key=key, body=None, compiling=False,
                     parameter_ownership=dict(entry.parameter_ownership),
                     infer_result_ownership=entry.result in types and entry.result_ownership == 'own')
    context.realizations[cache_key] = result
    context.compilers[id(result)] = context.compilers[id(entry)]
    for type_ in types.values():
        if not isinstance(type_, NumberArgument):
            context.types[type_.name] = type_
    return result


import unittest


class GenericRealizationTests(unittest.TestCase):
    def test_nested_generic_optional_results_preserve_bracket_boundaries(self):
        self.valid('''class Box[T]:
    value: T
    def optional(self: view Self) -> T | None:
        return copy value
test:
    nested = Box[Box[int]](Box[int](1))
    result = nested.optional()
    assert(result.has_value and result.value.value == 1)
    mixed: Box[int | bool] | None = Box[int | bool](true)
    assert(mixed.has_value and mixed.value.value == true)
''')

    identity = 'def identity[T = int](value: T) -> T:\n    return value\n'

    def test_provider_infers_value_argument_through_inherited_traits(self):
        self.valid('''trait C[K,V]:
    def get(self: view Self, key: K) -> V
trait Indexed[T]: C[int,T]
trait Storage[T]: Indexed[T]
class Slot[T]: Storage[T]
    value: T
    def get(self: view Self, key: int) -> T:
        return copy value
class Collection[T, C:C[int,T]]:
    storage: C[T]
    def Collection(value: T):
        storage = C[T](value)
    def get() -> T:
        return storage.get(0)
def work() -> string:
    values = Collection[T:string,C:Slot]("owned")
    result = values.get()
    drop(values)
    return result
''')

    def test_scoped_type_identity_is_not_a_named_generic_argument(self):
        self.valid('''trait C[T]:
    def get(self: view Self) -> T
class Box[T]: C[T]
    value: T
    def get(self: view Self) -> T:
        return copy value
def work() -> int:
    class Local:
        value: int
    box = Box[Local](Local(42))
    return box.get().value
''')

    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('generics.sev', text, Syntax())

    def valid(self, text):
        result = self.compile(text)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result.program

    def test_union_constraint_accepts_each_member_and_rejects_other_types(self):
        source = ('class Buffer[T: u8 | u32 = u8]:\n    value: T\n'
                  'def work():\n    byte = Buffer(u8(1))\n    wide = Buffer[u32](u32(955))\n')
        self.valid(source)
        result = self.compile(source + 'def invalid():\n    value = Buffer[u16](u16(1))\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('does not satisfy', str(result.diagnostics))

    def test_integer_string_object_and_dynamic_realizations(self):
        program = self.valid(self.identity + 'class Point:\n    x: int\n    y: int\ndef work():\n    integer = identity(42)\n    text = identity("hello")\n    original = Point(10, 20)\n    point = identity(original)\n    value: dynamic = 42\n    erased = identity(value)\n')
        bodies = [b for b in program.bodies if b.declaration == 'identity']
        self.assertEqual({b.result_type.name for b in bodies}, {'i64', 'string', 'Point', 'dynamic'})
        self.assertEqual(len({b.name for b in bodies}), 4)
        for body in bodies:
            scalar = body.result_type.name == 'i64'
            self.assertEqual(body.result_ownership, 'own' if scalar else 'view')
            self.assertEqual(body.result_sources, () if scalar else (0,))
        from py_compiler.mlir.src.lib import lower, render
        ir = render(lower(program))
        for representation in ('i64', 'memref<?xi8>', '!llvm.ptr', '!llvm.struct<(ptr, ptr)>'):
            self.assertIn(representation, ir)

    def test_defaults_inference_explicit_types_and_cache(self):
        program = self.valid(self.identity + 'def work():\n    a = identity(1)\n    b = identity(2)\n    c = identity[i32](3)\n    text = identity("hi")\n')
        bodies = [b for b in program.bodies if b.declaration == 'identity']
        self.assertEqual(sorted(b.result_type.name for b in bodies), ['i32', 'i64', 'string'])

    def test_default_type_used_when_arguments_cannot_infer_it(self):
        program = self.valid('def reserve[T = int](count: int) -> array[T]:\n    unsafe:\n        return allocate[T](count)\ndef work():\n    a = reserve(4)\n    b = reserve[i32](4)\n')
        self.assertEqual({b.result_type.name for b in program.bodies if b.declaration == 'reserve'}, {'array[i64]', 'array[i32]'})

    def test_callable_defaults_and_explicit_callable_identity(self):
        source = ('def some_function_default[T](value: T) -> T:\n    return value\n'
                  'def increment(value: int) -> int:\n    return value + 1\n'
                  'def apply[T = int, F = some_function_default](value: T) -> T:\n    return F(value)\n'
                  'def work():\n    a = apply(10)\n    b = apply[int, increment](10)\n    c = apply("hello")\n')
        program = self.valid(source)
        bodies = [b for b in program.bodies if b.declaration == 'apply']
        self.assertEqual(len(bodies), 3)
        self.assertEqual(len({b.name for b in bodies}), 3)

    def test_view_result_retains_source_lifetime(self):
        source = self.identity + 'def work():\n    original = "hello"\n    result = identity(original)\n    drop original\n    again = identity(result)\n'
        result = self.compile(source)
        self.assertTrue(result.diagnostics)
        self.assertIn('moved or dropped', str(result.diagnostics[0]))

    def test_local_object_view_cannot_escape_through_generic_call(self):
        result = self.compile(self.identity + 'class Point:\n    x: int\ndef bad() -> Point:\n    original = Point(1)\n    return identity(original)\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('borrowed result by value', str(result.diagnostics[0]))

    def test_generic_view_rejects_escape_and_use_after_owner_destruction(self):
        prefix = self.identity + 'class Point:\n    x: int\n'
        for body, diagnostic in (
            ('def bad() -> view Point:\n    point = Point(1)\n    return identity(point)\n', 'escapes'),
            ('def bad():\n    point = Point(1)\n    alias = identity(point)\n    drop point\n    result = alias.x\n', 'moved or dropped'),
        ):
            with self.subTest(diagnostic=diagnostic):
                result = self.compile(prefix + body)
                self.assertTrue(result.diagnostics)
                self.assertIn(diagnostic, str(result.diagnostics))

    def test_c_generic_view_and_owned_results_keep_distinct_contracts(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = self.identity + '''class Point:
    x: int
def duplicate[T](value: copy T) -> T:
    return value
def make[T](number: int) -> T:
    return Point(number)
def repeat[T](value: T, count: int) -> T:
    if count == 0:
        return value
    return repeat(value, count - 1)
test "generic results preserve views and independent owners":
    original = Point(4)
    alias = identity(original)
    repeated = repeat(original, 2)
    independent = duplicate(original)
    created = make[Point](42)
    original.x = 9
    assert(alias.x == 9)
    assert(repeated.x == 9)
    assert(independent.x == 4)
    assert(created.x == 42)
'''
        with tempfile.TemporaryDirectory(prefix='sev-generic-results-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'generic-results', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_conflicting_inference_and_invalid_callable_contract(self):
        cases = (
            'def same[T](a: T, b: T) -> T:\n    return a\ndef work():\n    x = same(1, "bad")\n',
            'def wrong(value: int) -> string:\n    return "bad"\ndef apply[T = int, F = wrong](value: T) -> T:\n    return F(value)\ndef work():\n    x = apply(1)\n',
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertTrue(self.compile(source).diagnostics)

    def test_symbols_include_types_and_callable_arguments_deterministically(self):
        source = (self.identity +
                  'def first(value: int) -> int:\n    return value + 1\n'
                  'def second(value: int) -> int:\n    return value + 2\n'
                  'def apply[T = int, F = first](value: T) -> T:\n    return F(value)\n'
                  'def work():\n    a = identity(1)\n    b = identity("hello")\n    c = apply[int, first](1)\n    d = apply[int, second](1)\n')
        one, two = self.valid(source), self.valid(source)
        self.assertEqual([b.name for b in one.bodies], [b.name for b in two.bodies])
        specialized = [b for b in one.bodies if b.declaration in ('identity', 'apply')]
        self.assertEqual(len(specialized), 4)
        self.assertEqual(len({b.name for b in specialized}), 4)
        self.assertTrue(any(b.name.startswith('__sev_fn_identity[i64]::') for b in specialized))
        self.assertTrue(any(b.name.startswith('__sev_fn_identity[string]::') for b in specialized))
        from py_compiler.mlir.src.lib import lower, render, symbol_ref
        text = render(lower(one))
        for body in specialized:
            self.assertIn('func.func ' + symbol_ref(body.name) + '(', text)
            self.assertIn('func.call ' + symbol_ref(body.name) + '(', text)
        symbols = {b.name for b in one.bodies}
        for body in one.bodies:
            for block in body.blocks:
                for operation in block.operations:
                    if operation.kind == 'call':
                        self.assertIn(operation.payload.symbol, symbols)

    def test_dependent_defaults_and_recursive_realization(self):
        self.valid('def make[T = int, U = T]() -> U:\n    return U(1)\ndef work():\n    x = make()\n')
        program = self.valid('def repeat[T](value: T, n: int) -> T:\n    if n == 0:\n        return value\n    return repeat[T](value, n - 1)\ndef work():\n    x = repeat(4, 2)\n')
        self.assertEqual(len([b for b in program.bodies if b.declaration == 'repeat']), 1)

    def test_realizations_verify_as_mlir(self):
        import shutil
        from py_compiler.mlir.src.lib import lower, render, verify_c, verifier_path
        if not shutil.which(verifier_path()):
            self.skipTest('install mlir-opt to verify generic realizations')
        program = self.valid(self.identity + 'class Point:\n    x: int\ndef work():\n    a = identity(1)\n    b = identity("hello")\n    p = Point(2)\n    c = identity(p)\n    value: dynamic = 3\n    d = identity(value)\n')
        verify_c(render(lower(program)))

    def test_multiple_type_and_number_parameters(self):
        source = ('class Counted[T,N]:\n    value: T\n'
                  'def first[T1,N1,T2,N2](a: Counted[T1,N1], b: Counted[T2,N2]) -> T1:\n    return a.value\n'
                  'def work() -> int:\n    a = Counted[int,2](1)\n    b = Counted[i32,3](3)\n    return first(a,b)\n')
        program = self.valid(source)
        body = next(body for body in program.bodies if body.declaration == 'first')
        self.assertIn('[i64,2,i32,3]', body.name)

    def test_number_template_default(self):
        program = self.valid('def make[T = int,N: int = 2]() -> T:\n    return T(1)\ndef work():\n    a = make()\n')
        body = next(body for body in program.bodies if body.declaration == 'make')
        self.assertIn('[i64,2]', body.name)


def record_clone(node, bindings, suffix):
    """Keep source spans while giving each realization distinct declaration IDs."""
    from dataclasses import replace
    local = bindings
    if node.provider and node.provider.spelling == 'def':
        parameters, _ = template_header(list(node.header))
        local = {name: value for name, value in bindings.items() if name not in {p.name for p in parameters}}
    def token(value, previous=None):
        text = substitute(value.text, local) if value.kind == 'IDENTIFIER' and getattr(previous, 'text', None) != '.' else value.text
        return value.with_text(text, kind='NUMBER' if text.isdigit() else value.kind)
    def tokens(values):
        return tuple(token(value, values[index-1] if index else None) if hasattr(value, 'text') else value
                     for index, value in enumerate(values))
    return replace(node, identity=node.identity + suffix, parent=node.parent + suffix,
                   tokens=tokens(node.tokens), header=tokens(node.header),
                   children=[record_clone(child, local, suffix) for child in node.children])


@dataclass
class GenericAlias:
    name: str
    parameters: tuple
    target: str
    arguments: tuple
    scope: tuple
    node: object

    def realize(self, arguments, context):
        supplied = named_arguments(self.parameters, arguments, context)
        if set(supplied) != {p.name for p in self.parameters}:
            raise ValueError('generic alias requires all arguments')
        bindings = {name:template_argument(value,context) for name,value in supplied.items()}
        resolved = tuple(substitute(value,bindings) for value in self.arguments)
        template = context.record_templates[context.declaration_name(self.target,self.scope)]
        return template.realize(resolved,context)


@dataclass(frozen=True)
class CallableAlias:
    target: str
    scope: tuple
    provider: str | None = None
    prelude: bool = False

    def resolve(self, context):
        target = self.target
        if self.provider:
            from py_compiler.syntax.prelude import require_library
            namespace = require_library(context, self.provider)
            if namespace is None:
                raise ValueError('unavailable callable provider ' + self.provider)
            target = namespace + '.' + target
        entries = context.lookup_functions(target, self.scope)
        if not entries:
            raise ValueError('unresolved callable alias target ' + target)
        return entries


def declare_alias(node, context):
    text = ''.join(t.text for t in node.tokens)
    position = next(i for i,t in enumerate(node.tokens) if t.text == 'as')
    left = ''.join(t.text for t in node.tokens[:position])
    right = ''.join(t.text for t in node.tokens[position+1:])
    if left.endswith('()') and right.endswith('()'):
        target, name = left[:-2], right[:-2]
        if not name.isidentifier() or not all(part.isidentifier() for part in target.split('.')):
            raise ValueError('callable alias requires callable names')
        qualified = '.'.join((*context.scope, name))
        if qualified in context.functions or qualified in context.function_aliases or qualified in context.by_name:
            raise ValueError('duplicate callable alias ' + name)
        binding = context.function_aliases.get(target)
        if binding is not None and binding.prelude:
            if context.scope:
                raise ValueError('prelude aliases must be renamed at module scope')
            context.function_aliases[qualified] = binding
            del context.function_aliases[target]
        else:
            context.check_prelude_conflict(name)
            context.function_aliases[qualified] = CallableAlias(target, context.scope)
        return
    if '[' not in left or '[' not in right or not left.endswith(']') or not right.endswith(']'):
        raise ValueError('generic alias requires a template and named parameters')
    target, inner = left[:-1].split('[',1)
    name, parameters = right[:-1].split('[',1)
    parameters = tuple(TemplateParameter(p) for p in split_arguments(parameters))
    qualified = '.'.join((*context.scope,name))
    if qualified in context.record_templates or qualified in context.types:
        raise ValueError('duplicate generic alias ' + name)
    context.record_templates[qualified] = GenericAlias(qualified,parameters,target,split_arguments(inner),context.scope,node)


class CallableAliasTests(unittest.TestCase):
    def test_alias_preserves_overloads_and_supports_forward_targets(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('aliases.sev',
            '__value() as value()\n'
            'def __value(x: int) -> int:\n    return x + 1\n'
            'def __value(x: bool) -> bool:\n    return not x\n'
            'test:\n    assert(value(2) == 3)\n    assert(value(false))\n', Syntax())
        self.assertFalse(result.diagnostics, result.diagnostics)
        self.assertEqual(result.aliases, ({'name': 'value', 'target': '__value'},))
        self.assertFalse(any(body.declaration == 'value' for body in result.program.bodies))

    def test_generic_alias_round_trips_through_imported_interface(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        producer = compile_source('provider.sev',
            'def __identity[T](value: T) -> T:\n    return value\n'
            '__identity() as identity()\n', Syntax())
        self.assertFalse(producer.diagnostics, producer.diagnostics)
        package = {'package': {'name': 'provider'}, 'objects': ['provider.o'],
                   'exports': [], 'types': [], 'templates': producer.templates, 'aliases': producer.aliases}
        result = compile_source('consumer.sev',
            'from provider import identity\ntest:\n    assert(identity(42) == 42)\n',
            Syntax(), c_packages={'provider': package})
        self.assertFalse(result.diagnostics, result.diagnostics)

    def test_cycles_and_duplicate_names_are_rejected(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for source, diagnostic in (
            ('b() as a()\na() as b()\ntest:\n    a()\n', 'cyclic callable alias'),
            ('def a() -> int:\n    return 1\na() as b()\ndef b() -> int:\n    return 2\n',
             'conflicts with alias'),
        ):
            with self.subTest(diagnostic=diagnostic):
                result = compile_source('bad-alias.sev', source, Syntax())
                self.assertTrue(result.diagnostics)
                self.assertIn(diagnostic, str(result.diagnostics[0]))


def infer_empty_constructor(constructor, binding, nodes, cfg, env):
    """Constrain an empty generic constructor from subsequent method arguments."""
    from py_compiler.syntax.function.contracts import signature
    from py_compiler.syntax.grammar.expression import expression
    from py_compiler.syntax.generic.place import expression_type
    context = cfg.context
    template = context.record_templates.get(context.declaration_name(constructor, cfg.declaration_scope))
    if template is None or not hasattr(template, 'node'):
        return None
    parameters = {p.name for p in template.parameters}
    methods = {}
    for child in template.node.children:
        if getattr(child.provider, 'spelling', '') == 'def':
            method = signature(child, template.name)
            methods[method.name.rsplit('.', 1)[-1]] = method.parameters
    inferred = {}
    def inspect(children):
        for node in children:
            tokens = node.tokens
            for index in range(len(tokens) - 3):
                if tokens[index].text != binding or tokens[index + 1].text != '.' or tokens[index + 3].text != '(':
                    continue
                method = methods.get(tokens[index + 2].text)
                if method is None:
                    continue
                depth, end = 1, index + 4
                while end < len(tokens) and depth:
                    depth += (tokens[end].text == '(') - (tokens[end].text == ')')
                    end += 1
                if depth:
                    continue
                call = expression(tokens[index:end])
                for (_, annotation), argument in zip(method, call.operands[1:]):
                    if annotation not in parameters:
                        continue
                    actual = expression_type(argument, cfg, env)
                    if annotation in inferred and inferred[annotation] != actual.name:
                        raise ValueError('conflicting constructor use inference for ' + binding)
                    inferred[annotation] = actual.name
            inspect(node.children)
    inspect(nodes)
    arguments = tuple(inferred.get(p.name, p.default) for p in template.parameters)
    if any(argument is None for argument in arguments):
        return None
    return template.realize(arguments, context)
