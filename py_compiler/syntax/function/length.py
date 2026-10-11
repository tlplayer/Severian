"""Bounded memory length comes from its descriptor, including pointer views."""
from dataclasses import dataclass
from py_compiler.syntax.generic.atom import Atom


@dataclass(frozen=True)
class Length:
    def call(self, type_, arguments, cfg, env, span):
        if type_ is not None or len(arguments) != 1:
            raise ValueError('len requires one bounded array or pointer')
        value = cfg.expr(arguments[0], env)
        if value.type.family not in ('array', 'pointer') or not value.type.mlir.startswith('memref<'):
            raise ValueError('len requires bounded memory')
        return cfg.emit('length', cfg.context.type('int'), (value,), self, span)

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        source, result = operation.operands[0], operation.result
        return [f'%dimension_{result.identity} = arith.constant 0 : index',
                f'%length_{result.identity} = memref.dim {name(source)}, %dimension_{result.identity} : {source.type.mlir}',
                f'{name(result)} = arith.index_cast %length_{result.identity} : index to {result.type.mlir}']
