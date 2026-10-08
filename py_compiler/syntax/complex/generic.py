"""Generic templates bind semantic arguments before compiling a realization."""
from dataclasses import dataclass, replace
import re
from py_compiler.syntax.generic.definition import TypeDefinition


class GenericType(TypeDefinition):
    def __init__(self):
        super().__init__('generic')


def split_arguments(text):
    depth, start, result = 0, 0, []
    for index, character in enumerate(text):
        depth += (character == '[') - (character == ']')
        if character == ',' and depth == 0:
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


def template_argument(spelling, context):
    return NumberArgument(int(spelling)) if spelling.isdigit() else context.type(spelling)


@dataclass(frozen=True)
class TemplateParameter:
    name: str
    default: str | None = None


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
        if len(tokens) > 1 and (len(tokens) < 3 or tokens[1].text != '='):
            raise ValueError('expected template parameter or parameter = default')
        default = ''.join(t.text for t in tokens[2:]) if len(tokens) > 1 else None
        result.append(TemplateParameter(tokens[0].text, default))
    return tuple(result), [items[0], *items[end + 1:]]


def substitute(annotation, types):
    if annotation is None:
        return None
    return re.sub(r'\b[A-Za-z_]\w*\b', lambda m: types[m[0]].name if m[0] in types else m[0], annotation)


def type_parameters(entry):
    words = set(re.findall(r'\b[A-Za-z_]\w*\b', ' '.join(t or '' for _, t in entry.parameters) + ' ' + (entry.result or '')))
    typed = {p.name for p in entry.templates if p.name in words}
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
            if getattr(actual, 'family', None) != base:
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
    key = tuple((p.name, 'type', types[p.name].name) if p.name in types else
                (p.name, 'callable', functions[p.name].symbol) for p in entry.templates)
    cache_key = (entry.symbol, key)
    if cache_key in context.realizations:
        return context.realizations[cache_key]
    result = replace(entry, parameters=tuple((n, substitute(t, types)) for n, t in entry.parameters),
                     result=substitute(entry.result, types), templates=(), type_arguments=types,
                     callable_arguments=functions, realization_key=key, body=None, compiling=False)
    context.realizations[cache_key] = result
    context.compilers[id(result)] = context.compilers[id(entry)]
    for type_ in types.values():
        if not isinstance(type_, NumberArgument):
            context.types[type_.name] = type_
    return result


import unittest


class GenericRealizationTests(unittest.TestCase):
    identity = 'def identity[T = int](value: T) -> T:\n    return value\n'

    def compile(self, text):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('generics.sev', text, Syntax())

    def valid(self, text):
        result = self.compile(text)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        return result.program

    def test_integer_string_object_and_dynamic_realizations(self):
        program = self.valid(self.identity + 'class Point:\n    x: int\n    y: int\ndef work():\n    integer = identity(42)\n    text = identity("hello")\n    original = Point(10, 20)\n    point = identity(original)\n    value: dynamic = 42\n    erased = identity(value)\n')
        bodies = [b for b in program.bodies if b.declaration == 'identity']
        self.assertEqual({b.result_type.name for b in bodies}, {'i64', 'string', 'Point', 'dynamic'})
        self.assertEqual(len({b.name for b in bodies}), 4)
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
        self.assertIn('storage belongs to this callable', str(result.diagnostics[0]))

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
        from py_compiler.mlir.src.lib import lower, render, verify_native, verifier_path
        if not shutil.which(verifier_path()):
            self.skipTest('install mlir-opt to verify generic realizations')
        program = self.valid(self.identity + 'class Point:\n    x: int\ndef work():\n    a = identity(1)\n    b = identity("hello")\n    p = Point(2)\n    c = identity(p)\n    value: dynamic = 3\n    d = identity(value)\n')
        verify_native(render(lower(program)))

    def test_multiple_type_and_number_parameters(self):
        source = ('def first[T1,N1,T2,N2](a: array[T1,N1], b: array[T2,N2]) -> T1:\n    return a[0]\n'
                  'def work() -> int:\n    a = array[int,2](1,2)\n    b = array[i32,3](3,4,5)\n    return first(a,b)\n')
        program = self.valid(source)
        body = next(body for body in program.bodies if body.declaration == 'first')
        self.assertIn('[i64,2,i32,3]', body.name)

    def test_number_template_default(self):
        self.valid('def make[T = int,N = 2]() -> array[T,N]:\n    return array[T,N](1,2)\ndef work():\n    a = make()\n')
