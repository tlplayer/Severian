"""Resolve a language operation through its owning declaration."""
from py_compiler.syntax.generic.owned import select
from py_compiler.syntax.catalog import operator_bindings


def binary(cfg, node, env, expected=None):
    binding = operator_bindings(node.token.syntax or cfg.syntax).get(node.token.text)
    if binding is None:
        raise ValueError('operator has no grammar binding contract')
    from py_compiler.syntax.complex.namespaces import selected_operator
    selected = selected_operator(cfg,node,env)
    if selected is not None:
        return selected
    if node.token.text == 'in':
        from py_compiler.syntax.generic.place import expression_type
        container = expression_type(node.operands[1],cfg,env)
        entries = cfg.context.functions.get(container.name+'.contains',())
        if entries:
            from py_compiler.syntax.function.calls import select as select_call, invoke
            receiver=cfg.expr(node.operands[1],env)
            entry,values=select_call(entries,(node.operands[0],),cfg,env)
            return invoke(entry,values,cfg,node.token.span,receiver)
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
    from py_compiler.syntax.complex.units import match_literal, binary as unit_binary
    right_term = node.operands[1]
    unit_rhs = right_term.kind == 'literal' and match_literal(right_term.token.text) is not None
    if not unit_rhs and right_term.kind in ('name','reference'):
        binding = cfg.lookup(right_term, env)
        unit_rhs = binding is not None and binding.type is not None and binding.type.family == 'unit'
    if not unit_rhs and right_term.kind in ('binary', 'unary', 'call', 'member', 'index'):
        from py_compiler.syntax.generic.place import expression_type
        try:
            unit_rhs = expression_type(right_term, cfg, env).family == 'unit'
        except ValueError:
            # Overload selection still belongs to ordinary expression lowering.
            pass
    if unit_rhs or left.type.family == 'unit':
        return unit_binary(node,left,cfg,env)
    if node.token.text in ('==', '!=') and getattr(cfg, 'placement', None) != 'simd':
        from py_compiler.syntax.complex.collection import is_sequence, sequence_equality
        if is_sequence(left.type):
            return sequence_equality(node, left, cfg, env)
    if left.type.family == 'union' and node.token.text in ('==', '!=') and node.operands[1].token.text not in ('None','absent'):
        from py_compiler.syntax.block.union import equality
        return equality(left, node, cfg, env)
    if callable(getattr(left.type, 'binary_operation', None)):
        return left.type.binary_operation(node, left, cfg, env)
    if (left.type.family == 'union' and node.token.text in ('==', '!=')
            and node.operands[1].kind == 'literal' and node.operands[1].token.text in ('None', 'absent')):
        from py_compiler.syntax.operator.logical import identity_expression
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        frozen = Expression('capture', node.token, (ArgumentValue(left),))
        matches = identity_expression(Expression('identity',node.token,(frozen,node.operands[1])),cfg,env)
        return matches if node.token.text == '==' else select(matches.type,'F.unary','not').expand(cfg,matches,node.token.span)
    if left.type.family == 'record':
        from py_compiler.syntax.function.calls import select as select_call, invoke
        name = left.type.name + '.__operator_' + '_'.join(str(ord(c)) for c in node.token.text)
        entries = tuple(e for e in cfg.context.functions.get(name, ()) if getattr(e,'operator_namespace',None) is None)
        if entries:
            entry, values = select_call(entries, (node.operands[1],), cfg, env)
            return invoke(entry, values, cfg, node.token.span, left)
    right_node = node.operands[1]
    right_owner = getattr(right_node.token.grammar, 'owner', None)
    right_expected = left.type
    if left.type.family in ('integer', 'float'):
        if getattr(right_owner, 'family', None) == 'float' and left.type.family == 'integer':
            right_expected = None
        elif right_node.kind in ('name', 'reference'):
            binding = cfg.lookup(right_node, env)
            if binding and binding.type and {left.type.family, binding.type.family} == {'integer', 'float'}:
                right_expected = None
    right = cfg.expr(right_node, env, right_expected)
    if {left.type.family, right.type.family} == {'integer', 'float'}:
        from py_compiler.syntax.primitive.numeric.conversion import ScalarConversion
        if left.type.family == 'integer':
            left = cfg.emit('convert', right.type, (left,), ScalarConversion(left.type, right.type), node.token.span)
        else:
            right = cfg.emit('convert', left.type, (right,), ScalarConversion(right.type, left.type), node.token.span)
    grammar = select(left.type, 'F.operator', node.token.text)
    return grammar.expand(cfg, left, right, node.token.span)
