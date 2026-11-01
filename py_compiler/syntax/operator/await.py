"""Await joins a task, transfers its result, and ends retained input lifetimes."""
from py_compiler.syntax.operator.operation import PrefixOperator


class Await(PrefixOperator):
    def __init__(self):
        super().__init__('await')

    def parse_prefix(self, token, parse, tokens, cursor):
        from py_compiler.syntax.grammar.expression import Expression
        return Expression('owned', token, (self, parse(6)))

    def lower_expression(self, node, cfg, env, expected):
        from py_compiler.syntax.generic.storage import root_binding, copy_storage
        from py_compiler.syntax.complex.error import unwrap_result
        term = node.operands[1]
        value = cfg.expr(term, env)
        if value.type.family == 'channel':
            return value.type.receive(value, cfg, node.token.span)
        if value.type.family != 'task':
            raise ValueError('await requires a task or channel')
        binding = root_binding(term, cfg, env)
        if binding is not None:
            cfg.flow.move(binding)
            cfg.flow.end_retained(binding.identity)
        value.type.join(value, cfg, node.token.span)
        result = None
        if value.type.result.mlir:
            # The task has one consumer. Transfer its payload before freeing
            # the frame; copying a resource handle then dropping the original
            # would invalidate the returned resource.
            result = value.type.read(value, cfg, node.token.span, 2, value.type.result)
            if value.type.result_ownership != 'own':
                # A returned view may point into an input snapshot in the
                # frame. Materialize it before releasing those snapshots.
                result = copy_storage(result, cfg, node.token.span)
            cfg.owned_results.add(result.identity)
        value.type.release(value, cfg, node.token.span, release_result=False)
        cfg.owned_results.discard(value.identity)
        return unwrap_result(result, cfg, node.token.span) if result is not None else None
