"""Resolve a language operation through its owning declaration."""
from py_compiler.syntax.generic.owned import select
from py_compiler.syntax.catalog import operator_bindings


def binary(cfg, node, env, expected=None):
    binding = operator_bindings(cfg.syntax).get(node.token.text)
    if binding is None:
        raise ValueError('operator has no grammar binding contract')
    if hasattr(binding, 'lower_expression'):
        return binding.lower_expression(cfg, node, env)
    left = cfg.expr(node.operands[0], env, None if binding.comparison else expected)
    grammar = select(left.type, 'F.operator', node.token.text)
    right = cfg.expr(node.operands[1], env, left.type)
    return grammar.expand(cfg, left, right, node.token.span)
