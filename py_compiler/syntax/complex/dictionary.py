"""dict[K:T1,V:T2,C]: key/value contracts independent of storage policy C."""
from dataclasses import dataclass
from py_compiler.syntax.complex.array import Array
from py_compiler.syntax.complex.native_collection import scalar, constant, emit, require_mutable


@dataclass(frozen=True)
class Dictionary(Array):
    name: str = 'dict'
    family: str = 'dict'

    def specialize_arguments(self, arguments, context):
        if len(arguments) not in (2, 3):
            raise ValueError('dict expects key type T1, value type T2, and optional C')
        key, value = (item.split(':', 1)[-1] for item in arguments[:2])
        return DictionaryType(context.type(key), context.type(value), context.type(arguments[2] if len(arguments) == 3 else 'btree'), self.name)

    def construct(self, explicit, arguments, cfg, env, span, expected=None):
        if len(arguments) % 2:
            raise ValueError('dictionary constructor requires alternating key, value arguments')
        target = self.specialize_arguments(explicit, cfg.context) if explicit else expected
        if target is not None and not isinstance(target, DictionaryType):
            raise ValueError('dictionary constructor requires a dictionary result type')
        values = []
        for index, argument in enumerate(arguments):
            type_ = (target.key if index % 2 == 0 else target.value) if target else (values[index % 2].type if index >= 2 else None)
            values.append(cfg.expr(argument, env, type_))
        if target is None:
            if not values:
                raise ValueError('empty map requires map[K,V] annotation or explicit types')
            target = DictionaryType(values[0].type, values[1].type, cfg.context.type('btree'), self.name)
        for index, value in enumerate(values):
            if value.type != (target.key if index % 2 == 0 else target.value):
                raise ValueError('dictionary entries must satisfy K and V')
        kind = 2 if target.key.family == 'string' else int(target.key.signed)
        result = emit('sev_dict_new', target, (constant(target.container.dictionary_kind, cfg, span), constant(kind, cfg, span)), cfg, span)
        for index in range(0, len(values), 2):
            target.place(result, values[index], cfg, span).write(cfg, values[index + 1], span)
        return result


@dataclass(frozen=True)
class DictionaryType:
    key: object
    value: object
    container: object
    family: str = 'dict'
    mlir: str = '!llvm.ptr'
    bits: int = 0
    signed: bool = False
    binding_ownership: str = 'view'
    parameter_ownership: str = 'view'

    def __post_init__(self):
        if self.key.family != 'string':
            scalar(self.key)
            if self.key.family == 'float':
                raise ValueError('dictionary keys require a total ordering; float keys are not supported')
        scalar(self.value)
        if not hasattr(self.container, 'dictionary_kind'):
            raise ValueError('C must implement the dictionary storage contract')

    @property
    def name(self):
        return f'{self.family}[{self.key.name},{self.value.name},{self.container.name}]'

    def grammars(self):
        return ()

    @property
    def generic_arguments(self):
        return (self.key, self.value, self.container)

    def place(self, receiver, key, cfg, span):
        padding = () if self.key.family == 'string' else (constant(0, cfg, span),)
        return DictionaryPlace(receiver, (key, *padding), self.value)

    def element_place(self, receiver, index, cfg, env, span):
        return self.place(receiver, cfg.expr(index, env, self.key), cfg, span)

    def release(self, value, cfg, span):
        emit('sev_dict_drop', None, (value,), cfg, span, release=True)

    def copy_value(self, value, cfg, span):
        return emit('sev_dict_copy', self, (value,), cfg, span)

    def call_method(self, receiver, method, arguments, cfg, env, span, binding):
        if method == 'len' and not arguments:
            return emit('sev_dict_len', cfg.context.type('int'), (receiver,), cfg, span)
        if method in ('contains', 'get') and len(arguments) == 1:
            place = self.element_place(receiver, arguments[0], cfg, env, span)
            if method == 'get':
                return place.read(cfg, span)
            return emit('sev_dict_contains', cfg.context.type('bool'), (receiver, *place.key), cfg, span)
        if method == 'set' and len(arguments) == 2:
            require_mutable(binding)
            place = self.element_place(receiver, arguments[0], cfg, env, span)
            place.write(cfg, cfg.expr(arguments[1], env, self.value), span)
            return None
        raise ValueError('dictionary supports get(key), set(key,value), contains(key), len(), and indexing')


@dataclass(frozen=True)
class DictionaryPlace:
    receiver: object
    key: tuple
    type: object

    def read(self, cfg, span):
        return emit('sev_dict_get', self.type, (self.receiver, *self.key), cfg, span)

    def write(self, cfg, value, span):
        if value.type != self.type:
            raise ValueError('dictionary assignment must satisfy V')
        emit('sev_dict_set', None, (self.receiver, *self.key, value), cfg, span, mutation=True)


import unittest


class DictionaryTests(unittest.TestCase):
    def test_storage_policy_preserves_interface(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        for container in ('btree', 'vector'):
            source = f'def work() -> int:\n    d = dict[int,int,{container}](1,10)\n    d[2] = 20\n    d.set(1,30)\n    return d.get(1)\n'
            result = compile_source('dict.sev', source, Syntax())
            self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_unknown_container_is_rejected(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('dict.sev', 'def bad():\n    d = dict[int,int,array](1,2)\n', Syntax())
        self.assertTrue(result.diagnostics)
