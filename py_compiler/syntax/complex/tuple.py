"""tuple[T,N] is homogeneous fixed-size storage, lowered directly to memref."""
from dataclasses import dataclass
from py_compiler.syntax.complex.tensor import Tensor, TensorType, TensorOperation, TensorPlace


class TupleType(Tensor):
    def __init__(self):
        super().__init__('tuple', 'tuple')

    def specialize(self, element, count):
        return TupleValue(element, shape=(count,))

    def specialize_arguments(self, arguments, context):
        if len(arguments) == 2 and arguments[1].isdigit():
            return self.specialize(context.type(arguments[0]), int(arguments[1]))
        if not arguments:
            raise ValueError('tuple requires member types or an element type and count')
        return product_type(tuple(context.type(t) for t in arguments), context)

    def construct_expression(self, argument, arguments, cfg, env, span, expected=None):
        from py_compiler.syntax.grammar.expression import type_spelling
        from py_compiler.syntax.function.calls import literal_value
        terms = argument.operands if argument and argument.kind == 'template_arguments' else (argument,) if argument else ()
        target = self.specialize_arguments(tuple(type_spelling(t) for t in terms), cfg.context) if terms else expected
        if getattr(target, 'family', None) == 'union':
            candidates = [member for member in target.variants if isinstance(member, (ProductType, TupleValue))
                          and (len(member.generic_arguments) if isinstance(member, ProductType) else member.count) == len(arguments)]
            target = candidates[0] if len(candidates) == 1 else None
        if isinstance(target, ProductType):
            return construct_product(target, arguments, cfg, env, span)
        if target is None and not arguments:
            raise ValueError('empty tuple requires an explicit element type: tuple[T,0]()')
        if target is not None and (not isinstance(target, TupleValue) or len(arguments) != target.count):
            raise ValueError('tuple requires exactly N initializer values')
        values = []
        for argument in arguments:
            values.append(cfg.expr(argument, env, target.element if target else None))
        if target is None:
            if len({v.type for v in values}) > 1 or any(v.type.family not in ('integer', 'float', 'bool', 'char', 'byte') for v in values):
                return construct_product(product_type(tuple(v.type for v in values), cfg.context), arguments, cfg, env, span, values)
            target = self.specialize(values[0].type, len(values))
        if any(value.type != target.element for value in values):
            raise ValueError('tuple initializer values must share element type T')
        count = literal_value(target.count, cfg.context.type('index'), cfg, span)
        result = cfg.emit('allocate-tuple', target, (count,), TensorOperation('allocate'), span)
        cfg.owned_results.add(result.identity)
        for index, value in enumerate(values):
            position = literal_value(index, cfg.context.type('index'), cfg, span)
            TensorPlace(result, (position,), target.element).write(cfg, value, span)
        return result


@dataclass(frozen=True)
class TupleValue(TensorType):
    family: str = 'tuple'

    def binary_operation(self, node, left, cfg, env):
        from py_compiler.syntax.complex.collection import primitive_method
        from py_compiler.syntax.generic.owned import select
        from py_compiler.syntax.generic.storage import root_binding
        if node.token.text in ('==', '!='):
            result = primitive_method(left, 'equal', (node.operands[1],), cfg, env, node.token.span,
                                      root_binding(node.operands[0], cfg, env))
            return result if node.token.text == '==' else select(result.type, 'F.unary', 'not').expand(cfg, result, node.token.span)
        return super().binary_operation(node, left, cfg, env)

    @property
    def name(self):
        return f'tuple[{self.element.name},{self.count}]'

    @property
    def generic_arguments(self):
        from py_compiler.syntax.complex.generic import NumberArgument
        return self.element, NumberArgument(self.count)


from py_compiler.syntax.complex.object import ObjectType


class ProductType(ObjectType):
    def grammars(self):
        from py_compiler.syntax.complex.collection import CollectionCondition
        return (*super().grammars(), CollectionCondition(self))

    def indexed_type(self, index):
        if index.kind != 'literal' or not index.token.text.isdigit():
            raise ValueError('heterogeneous tuple index must be a constant integer')
        position = int(index.token.text)
        if position >= len(self.generic_arguments):
            raise ValueError('tuple index outside member count')
        return self.generic_arguments[position]

    def element_place(self, receiver, index, cfg, env, span):
        self.indexed_type(index)
        return cfg.context.provider_by_name[self.name].field_place(receiver, 'item' + index.token.text, cfg, span)

    def length(self, value, cfg, span):
        from py_compiler.syntax.function.calls import literal_value
        return literal_value(len(self.generic_arguments), cfg.context.type('int'), cfg, span)


def product_type(types, context):
    from py_compiler.syntax.block.record import TypeDeclaration
    from py_compiler.syntax.generic.storage import owns_storage
    from importlib import import_module
    name = 'tuple[' + ','.join(t.name for t in types) + ']'
    if name in context.types:
        return context.types[name]
    declaration = TypeDeclaration('product:' + name, name, 'class',
        tuple(('item' + str(i), t.name) for i,t in enumerate(types)), (), (), context.source.span(0,0),
        defaults=(None,) * len(types), template_name='tuple', arguments=types)
    context.by_name[name] = declaration
    context.provider_by_name[name] = import_module('py_compiler.syntax.block.class').Class()
    context.types[name] = ProductType(name, 'record', '!llvm.ptr', declaration=declaration,
                                    owns_fields=any(owns_storage(t) for t in types))
    return context.types[name]


def construct_product(target, arguments, cfg, env, span, values=None):
    from py_compiler.syntax.complex.object import Allocate
    from py_compiler.syntax.generic.storage import FieldPlace, copy_storage, stored_type
    if len(arguments) != len(target.generic_arguments):
        raise ValueError('tuple requires exactly one value per member type')
    if values is None:
        values = [cfg.expr(term, env, type_) for term,type_ in zip(arguments,target.generic_arguments)]
    layout = cfg.context.provider_by_name[target.name].layout(target.declaration, cfg.context)
    result = cfg.emit('allocate', target, (), Allocate(layout), span)
    cfg.stack_values.add(result.identity)
    for index,(value,type_) in enumerate(zip(values,target.generic_arguments)):
        if value.type != type_:
            raise ValueError('tuple member does not satisfy its concrete type')
        FieldPlace(result,index,layout,type_,stored_type(type_,cfg.context)).write(cfg,copy_storage(value,cfg,span),span)
    from py_compiler.syntax.generic.storage import root_binding, release_storage
    for term,value in zip(arguments,values):
        if value.identity in cfg.owned_results and root_binding(term,cfg,env) is None:
            release_storage(value,cfg,span)
    cfg.owned_results.add(result.identity)
    return result


import unittest


class TupleTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('tuple.sev', source, Syntax(include_tests=True))

    def test_static_homogeneous_storage_and_dynamic_indexing(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('def make() -> tuple[int,3]:\n    return tuple[int,3](1,2,3)\ntest "tuple":\n    values = make()\n    index = 1\n    values[index] = 9\n    assert(values[index] == 9 and len(values) == 3)\n    copied = values.copy()\n    copied[0] = 5\n    assert(values[0] == 1)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        output = render(lower(result.program))
        self.assertIn('memref<3xi64>', output)
        self.assertNotIn('sev_vector_', output)

    def test_initializer_count_and_type_are_checked(self):
        for expression in ('tuple[int,2](1)', 'tuple[int,2](1,2,3)', 'tuple[int,2](1,true)', 'tuple[int,-1]()'):
            with self.subTest(expression=expression):
                self.assertTrue(self.compile('def work():\n    value = ' + expression + '\n').diagnostics)

    def test_heterogeneous_literal_uses_member_types(self):
        result = self.compile('test "members":\n    pair = (1, true)\n    assert(pair[0] == 1 and pair[1])\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_zero_length_tuple_has_static_storage(self):
        result = self.compile('test "empty":\n    value = tuple[u8,0]()\n    assert(len(value) == 0)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
