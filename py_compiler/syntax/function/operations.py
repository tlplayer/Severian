"""Resolve a language operation through its owning declaration."""
from py_compiler.syntax.generic.owned import select
from py_compiler.syntax.catalog import operator_bindings


def binary(cfg, node, env, expected=None):
    binding = operator_bindings(cfg.syntax).get(node.token.text)
    if binding is None:
        raise ValueError('operator has no grammar binding contract')
    if hasattr(binding, 'lower_expression'):
        return binding.lower_expression(cfg, node, env)
    left_node = node.operands[0]
    left_expected = None
    if expected and not binding.comparison:
        term = left_node
        while term.kind in ('unary', 'binary'):
            term = term.operands[0]
        owner = getattr(term.token.grammar, 'owner', None)
        if term.kind == 'literal' and (getattr(owner, 'family', None) == expected.family or
                                      (getattr(owner, 'family', None) == 'integer' and expected.family == 'float')):
            left_expected = expected
        elif term.kind in ('name', 'reference'):
            declaration = cfg.lookup(term, env)
            if declaration is not None and declaration.type in (None, expected):
                left_expected = expected
    left = cfg.expr(left_node, env, left_expected)
    if left.type.family == 'record':
        from py_compiler.syntax.function.calls import select as select_call, invoke
        name = left.type.name + '.__operator_' + '_'.join(str(ord(c)) for c in node.token.text)
        entries = cfg.context.functions.get(name, ())
        if entries:
            entry, values = select_call(entries, (node.operands[1],), cfg, env)
            return invoke(entry, values, cfg, node.token.span, left)
    grammar = select(left.type, 'F.operator', node.token.text)
    right = cfg.expr(node.operands[1], env, left.type)
    return grammar.expand(cfg, left, right, node.token.span)
