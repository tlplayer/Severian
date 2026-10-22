"""Callable values own snapshot captures through the existing dynamic box."""
from dataclasses import dataclass
from py_compiler.syntax.generic.definition import TypeDefinition
from py_compiler.syntax.dynamic.dynamic import DynamicType
from py_compiler.syntax.generic.atom import Atom


class FunctionType(TypeDefinition):
    def __init__(self):
        super().__init__('function')


@dataclass(frozen=True)
class CallableType(DynamicType):
    family: str = 'function'
    mlir: str = '!llvm.struct<(ptr, ptr, ptr)>'
    parameters: tuple = ()
    result: object = None

    def from_value(self, value, cfg, span):
        if value.type != self:
            raise ValueError('callable value does not satisfy ' + self.name)
        return value

    def storage_layout(self, context):
        from py_compiler.syntax.function.layout import Layout
        word = self.bits // 8
        return Layout(3 * word, word)

    def call(self, receiver, arguments, cfg, env, span):
        from types import SimpleNamespace
        from py_compiler.syntax.function.contracts import Callable
        from py_compiler.syntax.function.calls import select, finish_call_arguments, coerce
        if len(arguments) != len(self.parameters):
            raise ValueError('callable argument count mismatch')
        specification = Callable(None, ':indirect', tuple(('argument' + str(i), t.name)
            for i, t in enumerate(self.parameters)), self.result.name, ())
        specification.body = SimpleNamespace(result_sources=(), retained_result_sources=(), retained_parameter_sources=())
        entry, values = select((specification,), arguments, cfg, env)
        values = tuple(coerce(value, parameter, cfg, span) for value, parameter in zip(values, self.parameters))
        layout = cfg.context.provider_by_name[self.result.name].layout(self.result.declaration, cfg.context) if self.result.family == 'record' else ''
        operation = InvokeCallable(self, layout)
        if self.result.mlir:
            result = cfg.emit('call-indirect', self.result, (receiver, *values), operation, span)
            if self.result.binding_ownership != 'copy':
                cfg.owned_results.add(result.identity)
                cfg.value_dependencies[result.identity] = set()
                cfg.value_sources[result.identity] = set()
        else:
            cfg.effect('call-indirect', (receiver, *values), operation, span)
            result = None
        finish_call_arguments(entry, values, cfg, span)
        return result


def callable_type(parameters, result, context):
    name = '(' + ','.join(t.name for t in parameters) + ')->' + result.name
    if name not in context.types:
        context.types[name] = CallableType(name=name, bits=context.syntax.pointer_bits, parameters=tuple(parameters), result=result)
    return context.types[name]


def parse_callable_type(name, context):
    if not name.startswith('('):
        return None
    depth = 0
    for index, character in enumerate(name):
        depth += (character == '(') - (character == ')')
        if depth == 0:
            if name[index + 1:index + 3] != '->':
                return None
            from py_compiler.syntax.complex.generic import split_arguments
            parameters = tuple(context.type(item) for item in split_arguments(name[1:index])) if index > 1 else ()
            return callable_type(parameters, context.type(name[index + 3:]), context)
    return None


@dataclass(frozen=True)
class CallableCode:
    owner: object
    helper: object

    def generated_bodies(self):
        return (self.helper,)

    def auxiliary_bodies(self):
        from py_compiler.mir.lowering import lower_body
        return (lower_body(self.helper),)

    def atom(self, operation):
        return Atom(self, (self.owner,), self.owner, ('move',), (), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        from py_compiler.mlir.src.lib import symbol_ref
        result = name(operation.result)
        inputs = ', '.join((self.owner.mlir, *(t.mlir for t in self.owner.parameters)))
        output = self.helper.result_layout or self.owner.result.mlir or '()'
        signature = f'({inputs}) -> {output}'
        return [f'{result}_function = func.constant {symbol_ref(self.helper.name)} : {signature}',
                f'{result}_pointer = builtin.unrealized_conversion_cast {result}_function : {signature} to !llvm.ptr',
                f'{result} = llvm.insertvalue {result}_pointer, {name(operation.operands[0])}[2] : {self.owner.mlir}']


@dataclass(frozen=True)
class InvokeCallable:
    owner: object
    result_layout: str = ''

    def atom(self, operation):
        return Atom(self, (self.owner, *self.owner.parameters), self.owner.result if self.owner.result.mlir else None,
                    ('view',) * (len(self.owner.parameters) + 1), ('call',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name, temporary_suffix
        prefix = '%indirect_' + temporary_suffix(operation)
        inputs = ', '.join(value.type.mlir for value in operation.operands)
        result_type = self.result_layout or self.owner.result.mlir or '()'
        signature = f'({inputs}) -> {result_type}'
        lines = [f'{prefix}_pointer = llvm.extractvalue {name(operation.operands[0])}[2] : {self.owner.mlir}',
                 f'{prefix}_function = builtin.unrealized_conversion_cast {prefix}_pointer : !llvm.ptr to {signature}']
        arguments = ', '.join(name(value) for value in operation.operands)
        result = prefix + '_record' if self.result_layout else name(operation.result) if operation.result else ''
        lines.append((result + ' = ' if result else '') + f'func.call_indirect {prefix}_function({arguments}) : {signature}')
        if self.result_layout:
            lines += [f'{prefix}_one = arith.constant 1 : i64',
                      f'{name(operation.result)} = llvm.alloca {prefix}_one x {self.result_layout} : (i64) -> !llvm.ptr',
                      f'llvm.store {result}, {name(operation.result)} : {self.result_layout}, !llvm.ptr']
        return lines
