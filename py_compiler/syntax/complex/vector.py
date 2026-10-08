"""vector[T,N]: N is initial capacity; append grows the owning buffer."""
from dataclasses import dataclass
from py_compiler.syntax.complex.array import Array
from py_compiler.syntax.complex.native_collection import scalar, constant, emit, require_mutable


@dataclass(frozen=True)
class Vector(Array):
    name: str = 'vector'
    family: str = 'vector'
    dictionary_kind = 1

    def specialize(self, element):
        return VectorType(element)

    def specialize_arguments(self, arguments, context):
        if len(arguments) == 2 and arguments[1] in context.template_types:
            arguments = (arguments[0], context.template_types[arguments[1]].name)
        if len(arguments) not in (1, 2) or (len(arguments) == 2 and not arguments[1].isdigit()):
            raise ValueError('vector expects T and optional nonnegative N')
        return VectorType(context.type(arguments[0]), int(arguments[1]) if len(arguments) == 2 else 0)

    def construct(self, explicit, arguments, cfg, env, span, expected=None):
        target = self.specialize_arguments(explicit, cfg.context) if explicit else expected
        if target is not None and not isinstance(target, VectorType):
            raise ValueError('vector constructor requires a vector result type')
        values = []
        for argument in arguments:
            values.append(cfg.expr(argument, env, target.element if target else (values[0].type if values else None)))
        if target is None:
            if not values:
                raise ValueError('empty vector requires T')
            target = VectorType(values[0].type, len(values))
        if any(value.type != target.element for value in values):
            raise ValueError('vector values must satisfy T')
        result = emit('sev_vector_new', target, (constant(max(target.count, len(values)), cfg, span),), cfg, span)
        for value in values:
            emit('sev_vector_push', None, (result, value), cfg, span, mutation=True)
        return result


@dataclass(frozen=True)
class VectorType:
    element: object
    count: int = 0
    family: str = 'vector'
    mlir: str = '!llvm.ptr'
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'
    bits: int = 0
    signed: bool = False

    def __post_init__(self):
        scalar(self.element)
        if not 0 <= self.count <= (1 << 63) - 1:
            raise ValueError('vector N is outside supported capacity range')

    @property
    def name(self):
        return f'vector[{self.element.name},{self.count}]'

    def grammars(self):
        return ()

    @property
    def generic_arguments(self):
        from py_compiler.syntax.complex.generic import NumberArgument
        return (self.element, NumberArgument(self.count))

    def release(self, value, cfg, span):
        emit('sev_vector_drop', None, (value,), cfg, span, release=True)

    def copy_value(self, value, cfg, span):
        return emit('sev_vector_copy', self, (value,), cfg, span)

    def element_place(self, receiver, index, cfg, env, span):
        return VectorPlace(receiver, cfg.expr(index, env, cfg.context.type('int')), self.element)

    def call_method(self, receiver, method, arguments, cfg, env, span, binding):
        if method == 'append' and len(arguments) == 1:
            require_mutable(binding)
            value = cfg.expr(arguments[0], env, self.element)
            return emit('sev_vector_push', None, (receiver, value), cfg, span, mutation=True)
        if method == 'len' and not arguments:
            return emit('sev_vector_len', cfg.context.type('int'), (receiver,), cfg, span)
        raise ValueError('vector supports append(value), len(), and indexed access')


@dataclass(frozen=True)
class VectorPlace:
    receiver: object
    index: object
    type: object

    def read(self, cfg, span):
        return emit('sev_vector_get', self.type, (self.receiver, self.index), cfg, span)

    def write(self, cfg, value, span):
        if value.type != self.type:
            raise ValueError('vector assignment must satisfy T')
        emit('sev_vector_set', None, (self.receiver, self.index, value), cfg, span, mutation=True)


import unittest


class VectorTests(unittest.TestCase):
    def test_growth_and_indexed_access_lower_to_runtime(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        from py_compiler.mlir.src.lib import lower, render
        result = compile_source('vector.sev', 'def work() -> int:\n    v = vector[int,1](1)\n    v.append(2)\n    v[0] = 3\n    return v[1]\n', Syntax())
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        for symbol in ('sev_vector_push', 'sev_vector_get', 'sev_vector_set', 'sev_vector_drop'):
            self.assertIn(symbol, text)

    def test_view_cannot_append(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('view.sev', 'def bad(v: vector[int,1]):\n    v.append(2)\n', Syntax())
        self.assertTrue(result.diagnostics)
