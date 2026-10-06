"""Callable resolution and exact-one trait dispatch, owned by function syntax."""
from py_compiler.syntax.type.objects import Invoke, TraitBox, Extract, Assert
from py_compiler.hir.hir.src.program import Constant
from py_compiler.mir.cfg.cfg import Edge, Terminator
from py_compiler.mir.ownership.flow import Binding


def literal_value(value, type_, cfg, span):
    constant = Constant(cfg.body.identity + ':literal:' + str(cfg.next_value), None, type_, value, span, cfg.source.path, None)
    return cfg.emit('constant', type_, (), constant, span)


def coerce(value, expected, cfg, span):
    if value is None:
        raise ValueError('a no-result call cannot initialize a value')
    if expected is None or value.type == expected:
        return value
    if expected.family == "union" and value.type in expected.variants:
        from py_compiler.syntax.type.objects import UnionBox
        return cfg.emit("union", expected, (value,), UnionBox(expected.variants.index(value.type)), span)
    if value.type.family == 'record' and expected.family == 'trait':
        declaration = cfg.context.by_name[value.type.name]
        if expected.name in declaration.traits:
            return cfg.emit('trait', expected, (value,), TraitBox(cfg.context.tags[declaration.name]), span)
    raise ValueError(f'{value.type.name} does not satisfy {expected.name}')


def select(entries, arguments, cfg, env):
    candidates = [e for e in entries if len(e.parameters) == len(arguments)]
    if not candidates:
        raise ValueError('no callable matches the argument count')
    # Evaluate each argument once; use an expected type only when all candidates agree.
    values = []
    for index, argument in enumerate(arguments):
        annotations = {e.parameters[index][1] for e in candidates}
        expected = cfg.context.type(next(iter(annotations))) if len(annotations) == 1 and None not in annotations else None
        values.append(cfg.expr(argument, env, expected))
    def accepts(entry):
        for value, (_, annotation) in zip(values, entry.parameters):
            if annotation is None:
                continue
            target = cfg.context.type(annotation)
            if value.type == target:
                continue
            if value.type.family == 'record' and target.name in cfg.context.by_name[value.type.name].traits:
                continue
            return False
        return True
    candidates = [e for e in candidates if accepts(e)]
    if not candidates:
        raise ValueError('no callable matches the argument types')
    if len(candidates) != 1:
        raise ValueError('ambiguous callable overload; use an explicit argument type')
    return candidates[0], tuple(values)


def invoke(entry, values, cfg, span, receiver=None):
    cfg.context.compile(entry)
    operands = tuple(coerce(v, cfg.context.type(t), cfg, span) for v, (_, t) in zip(values, entry.parameters))
    if entry.receiver:
        if receiver is None:
            raise ValueError('method call requires a receiver')
        operands = (receiver, *operands)
    result = cfg.context.type(entry.result or 'absent')
    operation = Invoke(entry.symbol, result)
    if not result.mlir:
        cfg.effect('call', operands, operation, span)
        return None
    value = cfg.emit('call', result, operands, operation, span)
    # Until result lifetime contracts exist, reference results conservatively retain inputs.
    return value


def dispatch(receiver, method, arguments, cfg, env, span, contract_name=None):
    context = cfg.context
    contract = context.by_name[receiver.type.name if receiver else contract_name]
    methods = [m for m in contract.methods if m.name.rsplit('.', 1)[-1] == method]
    specification, values = select(methods, arguments, cfg, env)
    pointer = cfg.emit('extract', context.type('pointer'), (receiver,), Extract(0), span) if receiver else None
    tag = cfg.emit('extract', context.type('i32'), (receiver,), Extract(1), span) if receiver else None
    candidates = []
    for declaration in context.declarations:
        if contract.name in declaration.traits:
            for entry in context.functions.get(declaration.name + '.' + method, ()):
                if tuple(context.type(t) for _, t in entry.parameters) == tuple(context.type(t) for _, t in specification.parameters):
                    candidates.append((declaration, entry))
    if not candidates:
        raise ValueError('trait method has no implementation')
    if receiver is None and sum(not entry.guards for _, entry in candidates) > 1:
        raise ValueError('statically ambiguous trait dispatch: multiple unconditional implementations')
    flags = []
    for declaration, entry in candidates:
        if receiver:
            expected_tag = literal_value(context.tags[declaration.name], context.type('i32'), cfg, span)
            flag = cfg.emit('binary', context.type('bool'), (tag, expected_tag), '==', span)
        else:
            flag = literal_value(True, context.type('bool'), cfg, span)
        guard_env = {}
        for (parameter, _), value in zip(entry.parameters, values):
            binding = Binding(entry.symbol + ':' + parameter, parameter, value.type, True, 'copy', span)
            guard_env[parameter] = binding
            cfg.values[binding.identity] = value
        for guard in entry.guards:
            condition = cfg.expr(guard, guard_env, context.type('bool'))
            from py_compiler.syntax.type.objects import BooleanAnd
            flag = cfg.emit('and', context.type('bool'), (flag, condition), BooleanAnd(), span)
        flags.append(flag)
    from py_compiler.syntax.type.objects import CountBoolean
    count = literal_value(0, context.type('i32'), cfg, span)
    for flag in flags:
        count_value = cfg.emit('count', context.type('i32'), (flag,), CountBoolean(), span)
        count = cfg.emit('binary', context.type('i32'), (count, count_value), '+', span)
    one = literal_value(1, context.type('i32'), cfg, span)
    valid = cfg.emit('binary', context.type('bool'), (count, one), '==', span)
    cfg.effect('assert', (valid,), Assert(f'trait dispatch requires exactly one match: {contract.name}.{method} at {cfg.source.path}:{span.start}'), span)
    result_type = context.type(specification.result or 'absent')
    join = cfg.block((result_type,) if result_type.mlir else ())
    for index, ((declaration, entry), flag) in enumerate(zip(candidates, flags)):
        chosen = cfg.block()
        following = cfg.block() if index + 1 < len(candidates) else None
        cfg.current.terminator = (Terminator('conditional', (Edge(chosen.identity), Edge(following.identity)), flag)
                                  if following else Terminator('jump', (Edge(chosen.identity),)))
        cfg.current = chosen
        selected_receiver = pointer
        if receiver is None:
            provider = context.provider_by_name[declaration.name]
            selected_receiver = provider.instantiate(declaration, (), cfg, env, span)
        value = invoke(entry, values, cfg, span, selected_receiver)
        cfg.current.terminator = Terminator('jump', (Edge(join.identity, (value,) if value else ()),))
        if following:
            cfg.current = following
    cfg.current = join
    if join.parameters and result_type.family in ("record", "trait", "union") and (receiver is None or receiver.identity in cfg.stack_values):
        cfg.stack_values.add(join.parameters[0].identity)
    return join.parameters[0] if join.parameters else None


def lower_expression(node, cfg, env, expected=None):
    context, span = cfg.context, node.token.span
    if node.kind == 'member':
        subject = node.operands[0]
        if subject.kind == 'name':
            declared = context.types.get(context.declaration_name(subject.token.text, cfg.declaration_scope))
            if callable(getattr(declared, 'variant', None)):
                return coerce(literal_value(declared.variant(node.token.text), declared, cfg, span), expected, cfg, span)
        receiver = cfg.expr(node.operands[0], env)
        provider = context.provider_by_name.get(receiver.type.name)
        if not hasattr(provider, 'field_place'):
            raise ValueError('type has no field access provider')
        return coerce(provider.field_place(receiver, node.token.text, cfg, span).read(cfg, span), expected, cfg, span)
    callee, *arguments = node.operands
    receiver = None
    receiver_binding = None
    if callee.kind == 'name':
        name = callee.token.text
        declaration_name = context.declaration_name(name, cfg.declaration_scope)
        provider = context.provider_by_name.get(declaration_name)
        if hasattr(provider, 'instantiate'):
            return coerce(provider.instantiate(context.by_name[declaration_name], arguments, cfg, env, span), expected, cfg, span)
        if name in context.syntax.types:
            from py_compiler.syntax.generic.owned import select as select_grammar
            grammar = select_grammar(context.type(name), 'F.constructor', 'construct')
            return coerce(grammar.expand(cfg, arguments, env), expected, cfg, span)
        entries = context.lookup_functions(name, getattr(cfg, 'declaration_scope', ()))
        if not entries and cfg.receiver:
            entries = context.functions.get(cfg.receiver.name + '.' + name, ())
            receiver = cfg.receiver_value
    elif callee.kind == 'reference' and callee.token.text == 'self' and cfg.receiver:
        entries = context.functions.get(cfg.receiver.name + '.' + callee.operands[0].text, ())
        receiver = cfg.receiver_value
    elif callee.kind == 'member':
        subject = callee.operands[0]
        if subject.kind == 'name':
            name = context.declaration_name(subject.token.text, cfg.declaration_scope)
            declared = context.types.get(name)
            if declared and declared.family == 'trait':
                value = dispatch(None, callee.token.text, arguments, cfg, env, span, name)
                return coerce(value, expected, cfg, span) if expected else value
        receiver_binding = cfg.lookup(subject, env) if subject.kind in ("name", "reference") else None
        receiver = cfg.expr(subject, env)
        if receiver.type.family == 'trait':
            value = dispatch(receiver, callee.token.text, arguments, cfg, env, span)
            return coerce(value, expected, cfg, span) if expected else value
        if callee.token.text.startswith('__'):
            raise ValueError('private method is inaccessible outside its receiver')
        entries = context.functions.get(receiver.type.name + '.' + callee.token.text, ())
    else:
        raise ValueError('expression does not provide a callable')
    entry, values = select(entries, arguments, cfg, env)
    if receiver_binding and receiver_binding.ownership == 'view':
        body = context.compile(entry)
        from py_compiler.syntax.type.storage import FieldPlace
        if body is None or any(o.kind == 'write' and isinstance(o.payload, FieldPlace) for b in body.blocks for o in b.operations):
            raise ValueError('cannot call a mutating method through a view')
    value = invoke(entry, values, cfg, span, receiver)
    return coerce(value, expected, cfg, span) if expected else value
