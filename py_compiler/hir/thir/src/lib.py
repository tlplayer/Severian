"""Publish checked, immutable typed bodies in their HIR source partitions."""
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
    from py_compiler.hir.thir.src.checking import check_module
    drafts, declarations = check_module(module)
    frozen = {id(body): body.freeze() for body in drafts}
    bodies = tuple(frozen[id(body)] for body in drafts)
    context = module.context
    owners = {member: sub for sub in module.submodules for member in sub.declarations}
    nodes = {}
    def collect(node):
        nodes[node.identity] = node
        for child in node.children:
            collect(child)
    collect(module.root)
    origins = {}
    for node in nodes.values():
        origins.setdefault((node.span, id(node.provider)), []).append(node)

    def source_owner(node):
        imported = getattr(context, 'imported_template_origins', {}).get(node.span.source)
        if imported is not None:
            return owners[imported]
        if node.identity in owners:
            return owners[node.identity]
        # Generic record clones retain their original source span and provider.
        candidates = origins.get((node.span, id(node.provider)), ())
        if len(candidates) != 1:
            raise ValueError('THIR cannot locate the source partition for a generated declaration')
        return owners[candidates[0].identity]

    entries = {id(frozen[id(entry.body)]): entry for entry in (*context.callables, *context.realizations.values())
               if entry.body is not None and id(entry.body) in frozen}
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
                    for symbol, _, _ in getattr(operation.payload, 'symbol_contracts', ()):
                        target = body_owners.get(symbol)
                        if target is None:
                            raise ValueError('THIR callback has no typed realization: ' + symbol)
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

    # Behavior fixtures deliberately avoid generated symbols, body shapes and
    # pass implementation details, so the HIR/THIR boundary can change freely.
    point = 'class Point:\n    x: int\n'

    def native_behavior(self, source):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.testing import test_package
        typed = self.compile(source, stop_after='thir')
        self.assertFalse(typed.diagnostics, str(typed.diagnostics))
        with tempfile.TemporaryDirectory(prefix='sev-typed-behavior-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({
                'package': {'name': 'typed-behavior', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_native_inferred_views_and_owned_results_in_declaration_orders(self):
        declarations = (
            'def identity[T](value: T) -> T:\n    return value\n',
            'def duplicate[T](value: copy T) -> T:\n    return value\n',
            'def make[T](number: int) -> T:\n    return Point(number)\n',
        )
        checks = '''test "inferred results preserve aliasing and ownership":
    original = Point(4)
    alias = identity(original)
    independent = duplicate(original)
    created = make[Point](42)
    original.x = 9
    assert(alias.x == 9)
    assert(independent.x == 4)
    assert(created.x == 42)
    drop independent
    drop created
    assert(original.x == 9)
'''
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                ordered = declarations[::-1] if reverse else declarations
                self.native_behavior(self.point + ''.join(ordered) + checks)

    def test_native_return_branch_order_preserves_both_borrow_sources(self):
        # TDD: choosing between sources must not require one statically known owner.
        branches = (
            '    if take_left:\n        return left\n    return right\n',
            '    if not take_left:\n        return right\n    return left\n',
        )
        checks = '''test "either selected parameter remains a view":
    left = Point(10)
    right = Point(20)
    chosen_left = choose(left, right, true)
    chosen_right = choose(left, right, false)
    left.x = 11
    right.x = 22
    assert(chosen_left.x == 11)
    assert(chosen_right.x == 22)
'''
        for branch in branches:
            with self.subTest(branch=branch):
                self.native_behavior(self.point +
                    'def choose[T](left: T, right: T, take_left: bool) -> T:\n' + branch + checks)

    def test_either_returned_parameter_must_outlive_result(self):
        declaration = ('def choose[T](left: T, right: T, take_left: bool) -> T:\n'
                       '    if take_left:\n        return left\n    return right\n')
        for owner, flag in (('left', 'true'), ('right', 'false')):
            prefix = (self.point + declaration + 'def observe():\n'
                      '    left = Point(1)\n    right = Point(2)\n'
                      f'    selected = choose(left, right, {flag})\n')
            use = '    observed = selected.x\n'
            for stage in ('thir', 'mir'):
                with self.subTest(owner=owner, stage=stage):
                    # A rejection alone could pass because choosing a view is
                    # unsupported. Require the same use with live owners first.
                    valid = self.compile(prefix + use, stop_after=stage)
                    self.assertFalse(valid.diagnostics, str(valid.diagnostics))
                    result = self.compile(prefix + f'    drop {owner}\n' + use, stop_after=stage)
                    self.assertTrue(result.diagnostics)
                    self.assertRegex(str(result.diagnostics), 'moved or dropped|borrow|lifetime')

    def test_native_mutual_recursion_preserves_views_in_all_orders(self):
        # TDD: discovering a recursive return before the base return must not
        # commit the recursive callable to an owned-result contract.
        branches = (
            '    if count == 0:\n        return value\n'
            '    return second(value, count - 1)\n',
            '    if count > 0:\n        return second(value, count - 1)\n'
            '    return value\n',
        )
        checks = '''test "mutually recursive results retain the caller's owner":
    original = Point(4)
    from_first = first(original, 4)
    from_second = second(original, 4)
    original.x = 19
    assert(from_first.x == 19)
    assert(from_second.x == 19)
'''
        for branch in branches:
            declarations = (
                'def first[T](value: T, count: int) -> T:\n' + branch,
                'def second[T](value: T, count: int) -> T:\n    return first(value, count)\n',
            )
            for reverse in (False, True):
                with self.subTest(branch=branch, reverse=reverse):
                    ordered = declarations[::-1] if reverse else declarations
                    self.native_behavior(self.point + ''.join(ordered) + checks)

    def test_mixed_owned_and_view_results_rejected_in_both_branch_orders(self):
        branches = (
            '    if fresh:\n        return Point(7)\n    return value\n',
            '    if not fresh:\n        return value\n    return Point(7)\n',
        )
        for branch in branches:
            source = (self.point + 'def choose[T](value: T, fresh: bool) -> T:\n' + branch +
                      'def invalid():\n    original = Point(1)\n    result = choose(original, true)\n')
            for stage in ('thir', 'mir'):
                with self.subTest(branch=branch, stage=stage):
                    result = self.compile(source, stop_after=stage)
                    self.assertTrue(result.diagnostics)
                    self.assertIn('ownership contract', str(result.diagnostics))

    def test_invalid_semantics_report_diagnostics_at_type_checking(self):
        cases = {
            'unknown symbol': ('def invalid() -> int:\n    return missing\n', 'unknown binding'),
            'conflicting type arguments': (
                'def same[T](a: T, b: T) -> T:\n    return a\n'
                'def invalid():\n    result = same(1, "text")\n', 'conflicting type arguments'),
            'wrong result type': (
                'def invalid() -> int:\n    return "text"\n', 'does not satisfy'),
            'missing parameter type': (
                'def invalid(value):\n    return value\n', 'requires an annotation'),
            'invalid conversion': (self.point +
                'def invalid():\n    value = int(Point(1))\n', 'no declared conversion'),
            'unresolved recursive type argument': (
                'def recurse[T]() -> T:\n    return recurse[T]()\n'
                'def invalid():\n    value = recurse()\n', 'cannot infer template argument'),
        }
        for reason, (source, diagnostic) in cases.items():
            with self.subTest(reason=reason):
                result = self.compile(source, stop_after='thir')
                self.assertTrue(result.diagnostics, reason)
                self.assertIn(diagnostic, str(result.diagnostics))
                self.assertTrue(all(d.source.text == source for d in result.diagnostics))
                self.assertFalse(result.program.typed_modules)

    def test_native_inferred_types_conversions_and_callable_selection(self):
        self.native_behavior('''def convert[T](value: T) -> int:
    return int(value)
def increment(value: int) -> int:
    return value + 1
def apply[T, F=convert](value: T) -> int:
    return F(value)
test "types and callable arguments determine behavior":
    small: i32 = 7
    assert(apply(small) == 7)
    assert(apply(3.75) == 3)
    assert(apply[int, increment](7) == 8)
    assert(apply(7) == 7)
    assert(apply[i32](small) == 7)
''')

    def test_native_shared_generic_dependencies_with_increasing_callers(self):
        # Linear-sized fixtures exercise shared dependencies without asserting
        # a cache implementation or using noisy wall-clock performance limits.
        declarations = (
            'def identity[T](value: T) -> T:\n    return value\n'
            'def increment(value: int) -> int:\n    return value + 1\n'
            'def apply[T, F=identity](value: T) -> T:\n    return F(value)\n'
        )
        for count in (4, 16):
            helpers = [f'def caller_{i}(value: int) -> int:\n'
                       f'    return apply(value) + apply[int, increment]({i})\n'
                       for i in range(count)]
            checks = [f'    assert(caller_{i}(10) == {11 + i})' for i in range(count)]
            with self.subTest(callers=count):
                self.native_behavior(''.join(reversed(helpers)) + declarations +
                    'test "shared generic dependencies select the right callable":\n' +
                    '\n'.join(checks) + '\n')

    def test_checked_program_can_be_lowered_repeatedly_without_changing_meaning(self):
        from dataclasses import replace
        from py_compiler.mir.registration import register
        from py_compiler.mlir.src.lib import lower, render
        source = '''def identity[T](value: T) -> T:
    return value
def work(flag: bool) -> int:
    values = array[int, 2]()
    values[0] = 7
    values[1] = 11
    if flag:
        return identity(values[0])
    return identity(values[1])
'''
        checked = self.compile(source, stop_after='thir')
        self.assertFalse(checked.diagnostics, str(checked.diagnostics))
        direct = self.compile(source)
        self.assertFalse(direct.diagnostics, str(direct.diagnostics))
        expected = render(lower(direct.program))
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                modules = tuple(register(module) for module in checked.program.typed_modules)
                resumed = replace(checked.program, stage='mir', mir_modules=modules,
                                  bodies=tuple(body for module in modules for body in module.bodies))
                self.assertEqual(render(lower(resumed)), expected)

    def test_type_errors_are_reported_at_the_type_checking_boundary(self):
        for source, diagnostic in (
            ('def wrong() -> int:\n    return "text"\n', 'does not satisfy'),
            ('def wrong() -> int:\n    return missing\n', 'unknown binding'),
            ('class Point:\n    x: int\ndef wrong():\n    result = int(Point(1))\n', 'no declared conversion'),
        ):
            with self.subTest(source=source):
                result = self.compile(source, stop_after='thir')
                self.assertTrue(result.diagnostics)
                self.assertEqual(result.diagnostics[0].stage, 'THIR')
                self.assertIn(diagnostic, str(result.diagnostics[0]))
