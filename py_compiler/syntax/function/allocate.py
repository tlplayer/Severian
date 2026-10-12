"""One allocation interface; independent constraints become an explicit MIR plan."""
from dataclasses import dataclass
from py_compiler.syntax.complex.array import ArrayType, MemoryOperation


PROPERTIES = ('location', 'strategy', 'ownership', 'layout', 'alignment', 'lifetime')


@dataclass(frozen=True)
class AllocationPlan:
    location: str = 'heap'
    strategy: str = 'default'
    ownership: str = 'unique'
    layout: str = 'contiguous'
    alignment: int = 0
    lifetime: str = 'infer'
    explicit: tuple = ()


def resolve_plan(type_, overrides, context):
    from py_compiler.syntax.function.layout import layout_of
    defaults = dict(getattr(getattr(type_, 'declaration', None), 'memory_defaults', ()))
    constraints = {**defaults, **{k: v for k, v in overrides.items() if v != 'infer'}}
    values = {key: value.removeprefix('memory.') if isinstance(value, str) else value
              for key, value in constraints.items()}
    for key, allowed in (('location', ('infer', 'stack', 'heap')),
                         ('strategy', ('infer', 'default', 'arena', 'pool')),
                         ('ownership', ('infer', 'unique')),
                         ('layout', ('infer', 'contiguous')),
                         ('lifetime', ('infer', 'function'))):
        if values.get(key, 'infer') not in allowed:
            raise ValueError(f'allocation {key}={values[key]} has no supported lowering contract')
    natural = layout_of(type_, context).alignment if type_.family != 'array' else layout_of(type_.element, context).alignment
    alignment = values.get('alignment', 'infer')
    alignment = natural if alignment == 'infer' else int(alignment)
    if alignment < natural or alignment & (alignment - 1):
        raise ValueError('allocation alignment must be a power of two satisfying the type alignment')
    location = values.get('location', 'infer')
    if location == 'infer':
        location = 'stack' if values.get('lifetime') == 'function' or type_.family == 'record' else 'infer'
    strategy = values.get('strategy', 'default')
    strategy = 'default' if strategy == 'infer' else strategy
    if strategy != 'default' and location == 'stack':
        raise ValueError('arena/pool strategy requires heap region storage')
    return AllocationPlan(location=location, strategy=strategy, alignment=alignment,
                          lifetime=values.get('lifetime', 'infer'), explicit=tuple(sorted(overrides.items())))


class Allocate:
    def resolve_arguments(self, argument, context):
        from py_compiler.syntax.grammar.expression import type_spelling
        if argument is None:
            raise ValueError('allocate[T] requires a concrete type')
        terms = argument.operands if argument.kind == 'template_arguments' else (argument,)
        type_ = context.type(type_spelling(terms[0]))
        overrides = {}
        for term in terms[1:]:
            if term.kind != 'template_binding' or term.token.text != '=':
                raise ValueError('allocation overrides require property=value')
            key, value = map(type_spelling, term.operands)
            if key not in PROPERTIES:
                raise ValueError(f'unknown allocation property {key!r}')
            if key in overrides:
                raise ValueError(f'duplicate allocation property {key!r}')
            overrides[key] = value
        return type_, resolve_plan(type_, overrides, context)

    def call(self, request, arguments, cfg, env, span):
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        element, plan = request
        if element.family == 'record':
            if arguments or plan.location != 'stack':
                raise ValueError('class allocation currently requires default construction in function storage')
            provider = cfg.context.provider_by_name[element.name]
            return provider.instantiate(element.declaration, (), cfg, env, span, allocation_plan=plan)
        count_hint = element.count if isinstance(element, ArrayType) else None
        if isinstance(element, ArrayType):
            element = element.element
        if len(arguments) > 1 or (count_hint is not None and arguments):
            raise ValueError('allocate[T] accepts one count, or a fixed array extent')
        result = ArrayType(element, count=count_hint, location=plan.location)
        count = (select(cfg.context.type('index'), 'F.constructor', 'construct').expand(cfg, arguments, env)
                 if arguments else literal_value(count_hint if count_hint is not None else 1, cfg.context.type('index'), cfg, span))
        from py_compiler.syntax.complex.array import record_operation
        operation = record_operation(result, 'allocate', cfg).with_plan(plan) if element.family == 'record' else MemoryOperation('allocate', plan)
        value = cfg.emit('allocate-array', result, (count,), operation, span)
        if plan.location == 'stack':
            cfg.stack_values.add(value.identity)
        return value


import unittest


class AllocationTests(unittest.TestCase):
    def compile(self, source):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('allocation-plan.sev', source, Syntax())

    def test_named_constraints_reach_mlir(self):
        from py_compiler.mlir.src.lib import lower, render
        result = self.compile('def work() -> i32:\n    a = allocate[array[i32,16],location=memory.stack,alignment=64]()\n    a[0] = 7\n    return a[0]\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        text = render(lower(result.program))
        self.assertIn('memref.alloca', text)
        self.assertIn('alignment = 64', text)
        self.assertNotIn('memref.dealloc', text)

    def test_unsupported_policies_and_stack_escape_are_rejected(self):
        for override in ('location=memory.gpu.global', 'ownership=memory.arc',
                         'strategy=memory.missing', 'layout=memory.tiled', 'alignment=3',
                         'location=memory.heap,location=memory.stack'):
            result = self.compile(f'def work():\n    a = allocate[i32,{override}]()\n')
            self.assertTrue(result.diagnostics, override)
        result = self.compile('def work() -> array[i32]:\n    a = allocate[i32,location=memory.stack]()\n    return a\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('storage belongs to this callable', str(result.diagnostics))
