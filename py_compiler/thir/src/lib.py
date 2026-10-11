"""Materialize inferred callable contracts as typed views of HIR submodules.

The bootstrap's semantic providers still use Builder for inference and execution
expansion together. This pass captures their resolved output once; MIR registers
that output without recompiling bodies or changing the source partition.
"""
from dataclasses import dataclass
from hashlib import sha256
import json


def identity(parts):
    return sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def type_identity(type_):
    from py_compiler.syntax.complex.generic import NumberArgument
    if isinstance(type_, NumberArgument):
        return identity(('number', type_.value))
    if type_ is None or getattr(type_, 'family', 'unresolved') == 'unresolved':
        raise ValueError('THIR requires a resolved type')
    declaration = getattr(type_, 'declaration', None)
    arguments = getattr(type_, 'generic_arguments', ())
    if not arguments and hasattr(type_, 'element'):
        arguments = (type_.element,)
    variants = getattr(type_, 'variants', ())
    return identity(('type', type(type_).__module__, type(type_).__qualname__, type_.name,
                     getattr(declaration, 'identity', None),
                     tuple(type_identity(a) for a in arguments),
                     tuple(type_identity(v) for v in variants if not isinstance(v, str))))


def callable_identity(entry):
    return identity(('callable', entry.node.identity, entry.name,
                     tuple(sorted((name, type_identity(value)) for name, value in entry.type_arguments.items())),
                     tuple(sorted((name, callable_identity(value)) for name, value in entry.callable_arguments.items()))))


@dataclass(frozen=True)
class TypedCallable:
    identity: str
    definition: str
    name: str
    source: str
    parameters: tuple
    result: object
    type_arguments: tuple
    callable_arguments: tuple
    body: object


@dataclass(frozen=True)
class TypedSubmodule:
    identity: str
    source: str
    declarations: tuple
    substitution: tuple
    callables: tuple
    dependencies: tuple = ()
    constants: tuple = ()


@dataclass(frozen=True)
class TypedModule:
    identity: str
    interface: str
    source: object
    submodules: tuple
    bodies: tuple
    declarations: tuple


def prepare_literals(source, program, interface=None):
    owners = {member: sub for sub in program.submodules for member in sub.declarations}
    groups = []
    for constant in program.constants:
        source_submodule = owners[constant.identity]
        substitution = (('value-type', type_identity(constant.type)),)
        key = identity(('typed-submodule', source_submodule.identity, substitution))
        groups.append(TypedSubmodule(key, source_submodule.identity, source_submodule.declarations,
                                     substitution, (), (), (constant,)))
    return TypedModule(source.identity, interface or source.path, source, tuple(groups), (), program.declarations)


def prepare(module, interface=None):
    # Keep one semantic expansion: calling MIR lowering here would recursively
    # re-enter this pass. resolve_bodies is the existing provider/inference engine.
    from py_compiler.mir.lowering import resolve_bodies
    bodies, declarations = resolve_bodies(module)
    context = module.context
    owners = {member: sub for sub in module.submodules for member in sub.declarations}
    nodes = {}
    def collect(node):
        nodes[node.identity] = node
        for child in node.children:
            collect(child)
    collect(module.root)

    def source_owner(node):
        if node.identity in owners:
            return owners[node.identity]
        # Generic record clones retain their original source span and provider.
        candidates = [original for original in nodes.values()
                      if original.span == node.span and original.provider is node.provider]
        if len(candidates) != 1:
            raise ValueError('THIR cannot locate the source partition for a generated declaration')
        return owners[candidates[0].identity]

    entries = {id(entry.body): entry for entry in (*context.callables, *context.realizations.values())
               if entry.body is not None}
    groups, body_owners = {}, {}
    for body in bodies:
        entry = entries.get(id(body))
        node = entry.node if entry else module.root
        source = source_owner(node)
        type_arguments = tuple(sorted(entry.type_arguments.items())) if entry else ()
        callable_arguments = tuple(sorted((name, callable_identity(value))
                                         for name, value in entry.callable_arguments.items())) if entry else ()
        substitution = (tuple((name, type_identity(value)) for name, value in type_arguments), callable_arguments)
        key = identity(('typed-submodule', source.identity, substitution))
        parameters = tuple(value.type for value in body.blocks[0].parameters)
        # Include every value, including inferred locals and branch parameters.
        for type_ in (*parameters, body.result_type, *(binding.type for binding in body.bindings)):
            type_identity(type_)
        for block in body.blocks:
            for value in (*block.parameters, *(op.result for op in block.operations if op.result is not None),
                          *(v for op in block.operations for v in op.operands)):
                type_identity(value.type)
        callable_key = callable_identity(entry) if entry else identity(('initializer', module.identity))
        typed = TypedCallable(callable_key, node.identity, body.declaration, body.source, parameters,
                              body.result_type, type_arguments, callable_arguments, body)
        if body.name in body_owners:
            raise ValueError('THIR has duplicate callable symbols')
        body_owners[body.name] = key
        group = groups.setdefault(key, [source, substitution, []])
        group[2].append(typed)
    result = []
    from py_compiler.syntax.complex.object import Invoke
    for key, (source, substitution, callables) in groups.items():
        dependencies = set()
        for callable_ in callables:
            for block in callable_.body.blocks:
                for operation in block.operations:
                    if isinstance(operation.payload, Invoke):
                        target = body_owners.get(operation.payload.symbol)
                        if target is None:
                            raise ValueError('THIR call has no typed realization: ' + operation.payload.symbol)
                        dependencies.add(target)
        result.append(TypedSubmodule(key, source.identity, source.declarations, substitution,
                                     tuple(callables), tuple(sorted(dependencies))))
    return TypedModule(module.identity, interface or module.source.path, module.source,
                       tuple(result), bodies, declarations)


import unittest


class ThirTests(unittest.TestCase):
    source = ('def simple_hash[T](value: T) -> int:\n    return int(value)\n'
              'def hash[T, F=simple_hash](hashable: T) -> int:\n    return F(hashable)\n'
              'test:\n'
              '    assert hash[int](42) == simple_hash(42)\n'
              '    assert hash[float](3.14) == simple_hash(3.14)\n'
              "    assert hash[char]('a') == simple_hash('a')\n"
              '    assert hash[int, simple_hash](42) == hash(42)\n')

    def compile(self, text=None, **options):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('src/hash/mod.sev', text or self.source, Syntax(include_tests=True), **options)

    def test_hash_source_materializes_three_submodules_before_mir(self):
        result = self.compile(stop_after='thir')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertFalse(result.program.bodies)
        self.assertFalse(result.program.mir_modules)
        typed = result.program.typed_modules[0]
        hashes = [(sub, call) for sub in typed.submodules for call in sub.callables if call.name == 'hash']
        self.assertEqual(len(hashes), 3)
        self.assertEqual({call.parameters[0].name for _, call in hashes}, {'i64', 'f64', 'char'})
        self.assertEqual(len({sub.source for sub, _ in hashes}), 1)
        self.assertEqual(len({sub.identity for sub, _ in hashes}), 3)
        self.assertTrue(all(call.result.name == 'i64' for _, call in hashes))
        self.assertTrue(all(sub.dependencies for sub, _ in hashes))

    def test_i32_hash_is_distinct_from_i64_and_inferred_calls_reuse_it(self):
        source = self.source.split('test:', 1)[0] + (
            'test:\n    small: i32 = 42\n'
            '    assert hash[i32](42) == simple_hash(small)\n'
            '    assert hash(small) == hash[i32](42)\n'
            '    assert hash(42) == simple_hash(42)\n')
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        typed = result.program.typed_modules[0]
        hashes = [(sub, call) for sub in typed.submodules for call in sub.callables if call.name == 'hash']
        self.assertEqual(len(hashes), 2)
        self.assertEqual({call.parameters[0].name for _, call in hashes}, {'i32', 'i64'})
        self.assertEqual(len({sub.identity for sub, _ in hashes}), 2)
        self.assertTrue(all(call.result.name == 'i64' for _, call in hashes))
        i32_submodule = next(sub for sub, call in hashes if call.parameters[0].name == 'i32')
        dependencies = [call for sub in typed.submodules if sub.identity in i32_submodule.dependencies
                        for call in sub.callables]
        self.assertTrue(any(call.name == 'simple_hash' and call.parameters[0].name == 'i32'
                            for call in dependencies))

    def test_failed_inference_publishes_no_typed_submodule(self):
        for text in ('def bad(value):\n    return 1\n',
                     'def same[T](a: T, b: T) -> T:\n    return a\ndef work():\n    x = same(1, "bad")\n',
                     'def wrong(value: int) -> string:\n    return "bad"\n'
                     'def hash[T,F=wrong](value: T) -> int:\n    return F(value)\ndef work():\n    x = hash(1)\n'):
            with self.subTest(source=text):
                result = self.compile(text, stop_after='thir')
                self.assertTrue(result.diagnostics)
                self.assertFalse(result.program.typed_modules)

    def test_literal_submodules_and_test_root_selection(self):
        result = self.compile('answer: int = 42\n', stop_after='thir')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertEqual(result.program.typed_modules[0].submodules[0].constants[0].type.name, 'i64')
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        production = compile_source('hash.sev', self.source, Syntax(include_tests=False), stop_after='thir')
        self.assertFalse(production.diagnostics, str(production.diagnostics))
        self.assertFalse(any(c.name == 'hash' for module in production.program.typed_modules
                             for sub in module.submodules for c in sub.callables))

    def test_recursive_realization_and_distinct_callable_arguments(self):
        text = ('def first(value: int) -> int:\n    return value\n'
                'def second(value: int) -> int:\n    return value + 1\n'
                'def hash[T,F=first](value: T) -> int:\n    return F(value)\n'
                'def repeat[T](value: T, count: int) -> T:\n'
                '    if count == 0:\n        return value\n    return repeat[T](value, count - 1)\n'
                'def work():\n    a = hash(1)\n    b = hash[int,second](1)\n    c = repeat(1,2)\n')
        result = self.compile(text, stop_after='thir')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        submodules = result.program.typed_modules[0].submodules
        self.assertEqual(sum(c.name == 'hash' for s in submodules for c in s.callables), 2)
        recursive = next(s for s in submodules if any(c.name == 'repeat' for c in s.callables))
        self.assertIn(recursive.identity, recursive.dependencies)
