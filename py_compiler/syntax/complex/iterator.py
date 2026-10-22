"""Suspended iterators own independent frames and borrow their captured sources."""
from dataclasses import dataclass
from py_compiler.syntax.generic.definition import TypeDefinition
from py_compiler.syntax.generic.atom import Atom


class IteratorType(TypeDefinition):
    def __init__(self):
        super().__init__('iterator')

    def specialize(self, element):
        return Iterator(element)


@dataclass(frozen=True)
class Iterator:
    element: object
    family: str = 'iterator'
    binding_ownership: str = 'view'
    parameter_ownership: str = 'borrow'
    mlir: str = '!llvm.struct<(!llvm.ptr, !llvm.ptr, !llvm.ptr)>'

    def grammars(self):
        return ()

    def require_context(self, cfg):
        pass

    @property
    def name(self):
        return 'iterator[' + self.element.name + ']'

    def result(self, context):
        from py_compiler.syntax.block.union import UnionType
        return UnionType((self.element, context.type('None'))).with_layout(context)

    def release(self, value, cfg, span):
        cfg.effect('iterator-destroy', (value,), IteratorCall('destroy'), span)

    def copy_value(self, value, cfg, span):
        raise ValueError('iterator frames are unique owners; move the iterator or create another iterator(source)')

    def call_method(self, receiver, method, arguments, cfg, env, span, binding):
        if method != 'next' or arguments:
            raise ValueError('iterator supports next() with no arguments')
        if binding is None or binding.ownership == 'view':
            raise ValueError('next requires an owning iterator or a mutable borrow')
        from py_compiler.syntax.generic.storage import BindingPlace
        with BindingPlace(binding, rebind=False).capture(cfg, 'borrow'):
            result = cfg.emit('iterator-next', self.result(cfg.context), (receiver,), IteratorCall('next'), span)
        cfg.call_result_owners[result.identity] = binding
        cfg.call_result_modes[result.identity] = 'view'
        index = cfg.parameter_bindings.get(binding.identity)
        cfg.value_sources[result.identity] = {index} if index is not None else set()
        from py_compiler.syntax.generic.storage import contains_borrowed_fields
        if contains_borrowed_fields(self.element, cfg.context):
            cfg.value_dependencies[result.identity] = {cfg.flow.resolve(binding.identity)}
        if result.type.release and index is None:
            cfg.stack_values.add(result.identity)
        return result


class Next:
    def call(self, type_, arguments, cfg, env, span):
        if type_ is not None or len(arguments) != 1:
            raise ValueError('next requires one iterator')
        from py_compiler.syntax.generic.storage import root_binding
        binding = root_binding(arguments[0], cfg, env)
        value = cfg.expr(arguments[0], env)
        if value.type.family != 'iterator':
            raise ValueError('next requires an iterator')
        return value.type.call_method(value, 'next', (), cfg, env, span, binding)


@dataclass(frozen=True)
class IteratorCall:
    kind: str

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type if operation.result else None,
                    ('move' if self.kind == 'destroy' else 'borrow',), ('call',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        value, tag = operation.operands[0], temporary_suffix(operation)
        result = name(operation.result) + ' = ' if operation.result else ''
        output = operation.result.type.mlir if operation.result else '()'
        return [f'%frame_{tag} = llvm.extractvalue {name(value)}[0] : {value.type.mlir}',
                f'%resume_{tag} = llvm.extractvalue {name(value)}[{2 if self.kind == "destroy" else 1}] : {value.type.mlir}',
                f'{result}llvm.call %resume_{tag}(%frame_{tag}) : !llvm.ptr, (!llvm.ptr) -> {output}']


import unittest


class IteratorTests(unittest.TestCase):
    source = '''calls = 0
class Items:
    count: int
    grammar iterate["iterator", "(", self, ")"]() -> iterator[int]:
        scratch = array[int,2]()
        index = 0
        while index < self.count:
            module.calls += 1
            scratch[0] = index * 10
            yield scratch[0]
            index += 1

test "independent lazy frames retain local values":
    values = Items(3)
    first = iterator(values)
    second = iterator(values)
    assert(module.calls == 0)
    assert(next(first).value == 0)
    assert(module.calls == 1)
    assert(next(first).value == 10)
    assert(next(second).value == 0)
    assert(module.calls == 3)
    drop second
    assert(next(first).value == 20)
    assert(not next(first).has_value)
    assert(not next(first).has_value)
    assert(module.calls == 4)
    assert(next(first) == None)

test "stored cursors also support for and early break":
    values = Items(4)
    cursor = iterator(values)
    assert(next(cursor).value == 0)
    total = 0
    for value in cursor:
        total += value
        if value == 20:
            break
    assert(total == 30)
    assert(next(cursor).value == 30)
    assert(next(cursor) == None)

test "early destruction frees suspended local owners":
    values = Items(10)
    cursor = iterator(values)
    assert(next(cursor).has_value)
    drop cursor

test "unstarted frame destruction runs no producer code":
    before = module.calls
    values = Items(1)
    cursor = iterator(values)
    drop cursor
    assert(module.calls == before)
'''

    def test_c_suspension_exhaustion_and_early_destruction(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        with tempfile.TemporaryDirectory(prefix='sev-iterator-frames-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(self.source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'iterator-frames', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_captured_sources_and_yielded_views_cannot_outlive_owners(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        base = self.source.split('test "independent')[0]
        for statement in ('drop values', 'values.count = 0'):
            result = compile_source('iterator-lifetime.sev', base +
                'test:\n    values = Items(2)\n    cursor = iterator(values)\n    ' + statement + '\n    next(cursor)\n', Syntax())
            self.assertTrue(result.diagnostics)
            self.assertIn('borrow', str(result.diagnostics))
        result = compile_source('iterator-escape.sev', base +
            'def bad() -> iterator[int]:\n    values = Items(2)\n    cursor = iterator(values)\n    return cursor\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('borrow of local storage', str(result.diagnostics))
        result = compile_source('iterator-view.sev', base +
            'test:\n    values = Items(2)\n    cursor = iterator(values)\n    item = next(cursor)\n    next(cursor)\n    assert(item.value == 0)\n', Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('borrow', str(result.diagnostics))

    def test_c_owned_yields_and_returned_frame_lifetimes(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''class Buffer:
    values: array[int]
class Items:
    count: int
    grammar iterate["iterator", "(", self, ")"]() -> iterator[Buffer]:
        buffer = Buffer(array[int,1]())
        buffer.values[0] = self.count
        yield buffer
        buffer.values[0] = 7
        yield buffer

def cursor_for(source: Items) -> iterator[Buffer]:
    cursor = iterator(source)
    return cursor

test "copied yielded records own independent storage":
    source = Items(42)
    cursor = cursor_for(source)
    saved = copy next(cursor).value
    assert(saved.values[0] == 42)
    assert(next(cursor).value.values[0] == 7)
    drop cursor
    assert(saved.values[0] == 42)

test with compiler "returned iterator retains its captured parameter":
    reject "borrow":
        class Items:
            value: int
            grammar iterate["iterator", "(", self, ")"]() -> iterator[int]:
                yield self.value
        def cursor_for(source: Items) -> iterator[int]:
            cursor = iterator(source)
            return cursor
        def invalid():
            source = Items(1)
            cursor = cursor_for(source)
            drop source
            next(cursor)
'''
        with tempfile.TemporaryDirectory(prefix='sev-owned-iterator-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'owned-iterator', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_argument_evaluation_cannot_invalidate_an_earlier_yield(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        source = '''class Item:
    value: int
    grammar iterate["iterator", "(", self, ")"]() -> iterator[Item]:
        yield self
        yield self
def combine(left: Item, right: Item) -> int:
    return left.value + right.value
test:
    item = Item(2)
    cursor = iterator(item)
    result = combine(next(cursor).value, next(cursor).value)
'''
        result = compile_source('iterator-arguments.sev', source, Syntax())
        self.assertTrue(result.diagnostics)
        self.assertIn('borrow', str(result.diagnostics))
