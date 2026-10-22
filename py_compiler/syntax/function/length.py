"""Bounded memory length comes from its descriptor, including pointer views."""
from dataclasses import dataclass
from py_compiler.syntax.generic.atom import Atom


@dataclass(frozen=True)
class Length:
    stride: int = 1

    def predicate_argument(self, argument, parameters, context):
        if argument.kind != 'name' or not isinstance(parameters, dict):
            return False
        annotation = parameters.get(argument.token.text)
        if not annotation:
            return False
        type_ = context.type(annotation)
        entries = context.functions.get(type_.name + '.len', ())
        if entries:
            from py_compiler.syntax.grammar.expression import expression
            from py_compiler.syntax.function.contracts import pure_predicate
            for entry in entries:
                children = entry.node.children
                if len(children) != 1 or children[0].tokens[0].text != 'return':
                    return False
                try:
                    pure_predicate(expression(children[0].tokens[1:]), {'self'} | {name for name, _ in type_.declaration.fields})
                except ValueError:
                    return False
            return True
        return type_.family in ('array', 'pointer', 'tensor', 'tuple', 'string')

    def call(self, type_, arguments, cfg, env, span):
        if type_ is not None or len(arguments) != 1:
            raise ValueError('len requires one bounded array or pointer')
        value = cfg.expr(arguments[0], env)
        entries = cfg.context.functions.get(value.type.name + '.len', ())
        if entries:
            from py_compiler.syntax.function.calls import select, invoke
            entry, operands = select(entries, (), cfg, env)
            return invoke(entry, operands, cfg, span, value)
        if callable(getattr(value.type, 'length', None)):
            return value.type.length(value, cfg, span)
        if value.type.family not in ('array', 'pointer') or not value.type.mlir.startswith('memref<'):
            raise ValueError(f'{value.type.name} does not provide len; a sized container is required')
        from py_compiler.syntax.complex.array import record_storage
        stride = record_storage(value.type, cfg.context)[0] if getattr(value.type, 'aggregate', False) else 1
        return cfg.emit('length', cfg.context.type('int'), (value,), Length(stride), span)

    def atom(self, operation):
        return Atom(self, (operation.operands[0].type,), operation.result.type, ('view',), ('read',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import name
        source, result = operation.operands[0], operation.result
        lines = [f'%dimension_{result.identity} = arith.constant 0 : index',
                 f'%length_{result.identity} = memref.dim {name(source)}, %dimension_{result.identity} : {source.type.mlir}']
        length = f'%length_{result.identity}'
        if self.stride != 1:
            lines += [f'%stride_{result.identity} = arith.constant {self.stride} : index',
                      f'%elements_{result.identity} = arith.divui {length}, %stride_{result.identity} : index']
            length = f'%elements_{result.identity}'
        if result.type.mlir == 'index':
            lines.append(f'{name(result)} = arith.addi {length}, %dimension_{result.identity} : index')
        else:
            lines.append(f'{name(result)} = arith.index_cast {length} : index to {result.type.mlir}')
        return lines
