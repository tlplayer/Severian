"""Callable resolution and exact-one trait dispatch, owned by function syntax."""
from py_compiler.syntax.block.record import implements
from py_compiler.syntax.complex.object import Invoke, TraitBox, Extract, Assert
from py_compiler.hir.hir.src.program import Constant
from py_compiler.hir.thir.src.model import Edge, Terminator
from py_compiler.mir.ownership.flow import Binding


def literal_value(value, type_, cfg, span):
    constant = Constant(cfg.body.identity + ':literal:' + str(cfg.next_value), None, type_, value, span, cfg.source.path, None)
    return cfg.emit('constant', type_, (), constant, span)


def coerce(value, expected, cfg, span):
    if value is None:
        if expected is not None and expected.family == 'union':
            absent = cfg.context.type('absent')
            if absent in expected.variants:
                from py_compiler.syntax.block.union import UnionNone
                result = cfg.emit('union-none', expected, (), UnionNone(expected.variants.index(absent)), span)
                cfg.owned_results.add(result.identity)
                return result
        raise ValueError('a no-result call cannot initialize a value')
    conversion = getattr(expected, 'from_value', None)
    if conversion is not None and value.type != expected:
        return conversion(value, cfg, span)
    if expected is None or value.type == expected:
        return value
    if expected.family == "union" and value.type in expected.variants:
        from py_compiler.syntax.complex.object import UnionBox
        from py_compiler.syntax.generic.storage import copy_storage
        result = cfg.emit('memory-copy', expected, (copy_storage(value, cfg, span),), UnionBox(expected.variants.index(value.type)), span)
        cfg.owned_results.add(result.identity)
        return result
    if expected.family == 'union' and value.type.family == 'union' and expected.accepts_type(value.type):
        from py_compiler.syntax.block.union import UnionWiden
        copied = value.type.copy_value(value, cfg, span)
        result = cfg.emit('memory-copy', expected, (copied,), UnionWiden(), span)
        cfg.owned_results.add(result.identity)
        return result
    if value.type.family == 'record' and expected.family == 'trait':
        declaration = cfg.context.by_name[value.type.name]
        if implements(declaration, expected.name, cfg.context):
            return cfg.emit('trait', expected, (value,), TraitBox(cfg.context.tags[declaration.name]), span)
    raise ValueError(f'{value.type.name} does not satisfy {expected.name}')


def expand_spreads(arguments, cfg, env):
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.generic.storage import root_binding
    expanded = []
    for argument in arguments:
        if argument.kind != 'spread':
            # Freeze earlier arguments before evaluating a later spread source.
            value = cfg.expr(argument, env)
            expanded.append(Expression('capture', argument.token, (ArgumentValue(value, root_binding(argument, cfg, env)),)))
            continue
        source = argument.operands[0]
        if source.kind in ('tuple', 'list'):
            expanded.extend(expand_spreads(source.operands, cfg, env))
            continue
        if source.kind == 'binary' and source.token.text == '..':
            def integer(term):
                if term.kind == 'literal' and term.token.text.isdecimal():
                    return int(term.token.text)
                if term.kind == 'unary' and term.token.text in ('+', '-'):
                    return integer(term.operands[0]) * (-1 if term.token.text == '-' else 1)
                raise ValueError('range spread requires constant bounds')
            start, stop = map(integer, source.operands)
            if stop - start > 65536:
                raise ValueError('range spread exceeds the static argument limit')
            expanded.extend(Expression('literal', source.token.with_text(str(value), kind='NUMBER')) for value in range(start, stop))
            continue
        value = cfg.expr(source, env)
        owner = root_binding(source, cfg, env)
        count = len(value.type.generic_arguments) if getattr(value.type.declaration if value.type.family == 'record' else None, 'template_name', '') == 'tuple' else getattr(value.type, 'count', None)
        if count is None:
            raise ValueError('spread requires a tuple or a sequence with a fixed extent')
        if owner is None:
            temporary = '<spread ' + str(value.identity) + '>'
            owner = Binding(cfg.body.identity + temporary, temporary, value.type, False, 'own', source.token.span)
            env[temporary] = owner
            cfg.local_names.add(temporary)
            cfg.values[owner.identity] = value
            cfg.body.bindings.append(owner)
        receiver = Expression('capture', source.token, (ArgumentValue(value, owner),))
        for position in range(count):
            index = Expression('literal', source.token.with_text(str(position), kind='NUMBER'))
            expanded.append(Expression('index', source.token, (receiver, index)))
    return tuple(expanded)


def select_variadic(entries, arguments, cfg, env, explicit):
    from dataclasses import replace
    from py_compiler.syntax.grammar.expression import Expression
    from py_compiler.syntax.complex.tensor import ArgumentValue
    from py_compiler.syntax.complex.tuple import product_type, construct_product
    from py_compiler.syntax.generic.storage import root_binding
    exact = [entry for entry in entries if not entry.variadic and len(entry.parameters) == len(arguments)]
    if exact:
        return select(exact, arguments, cfg, env, explicit)
    candidates = [entry for entry in entries if entry.variadic and len(arguments) >= len(entry.parameters) - 1]
    if len(candidates) != 1:
        raise ValueError('variadic call requires one matching declaration')
    entry = candidates[0]
    if entry.templates or explicit or any(argument.kind == 'named_argument' for argument in arguments):
        raise ValueError('variadic calls require positional arguments and inferred pack types')
    fixed = len(entry.parameters) - 1
    values = []
    for index, argument in enumerate(arguments):
        annotation = entry.parameters[min(index, fixed)][1]
        value = cfg.expr(argument, env, cfg.context.type(annotation) if annotation else None)
        values.append(value)
    pack_type = product_type(tuple(value.type for value in values[fixed:]), cfg.context)
    token = arguments[0].token if arguments else entry.node.tokens[0]
    pack = construct_product(pack_type, arguments[fixed:], cfg, env, token.span, values[fixed:])
    parameter_types = (*[value.type for value in values[:fixed]], pack_type)
    key = ('variadic', id(entry), tuple(type_.name for type_ in parameter_types))
    specialized = cfg.context.realizations.get(key)
    if specialized is None:
        parameters = tuple((name, type_.name) for (name, _), type_ in zip(entry.parameters, parameter_types))
        specialized = replace(entry, parameters=parameters, variadic=None, body=None, compiling=False,
                              declaration_signature=tuple(type_.name for type_ in parameter_types),
                              parameter_ownership=dict(entry.parameter_ownership))
        cfg.context.realizations[key] = specialized
        cfg.context.compilers[id(specialized)] = cfg.context.compilers[id(entry)]
    captured = [Expression('capture', argument.token, (ArgumentValue(value, root_binding(argument, cfg, env)),))
                for argument, value in zip(arguments[:fixed], values[:fixed])]
    captured.append(Expression('capture', token, (ArgumentValue(pack),)))
    return select((specialized,), tuple(captured), cfg, env)


def select(entries, arguments, cfg, env, explicit=()):
    if explicit:
        # A package can export both a generic declaration and realizations
        # used by its own bodies. Explicit type arguments select the template.
        entries = tuple(entry for entry in entries if entry.templates)
        if not entries:
            raise ValueError('callable does not declare template parameters')
    if any(entry.variadic for entry in entries):
        return select_variadic(entries, arguments, cfg, env, explicit)
    if any(argument.kind == 'named_argument' for argument in arguments):
        from py_compiler.syntax.grammar.expression import Expression
        from py_compiler.syntax.complex.tensor import ArgumentValue
        from py_compiler.syntax.generic.storage import root_binding
        from py_compiler.syntax.complex.generic import explicit_bindings, substitute, type_parameters
        import re
        names = [argument.token.text for argument in arguments if argument.kind == 'named_argument']
        if len(set(names)) != len(names):
            raise ValueError('duplicate named call argument')
        positional = sum(argument.kind != 'named_argument' for argument in arguments)
        def accepts(entry):
            parameters = [name for name, _ in entry.parameters]
            return (positional <= len(parameters) and all(name in parameters[positional:] for name in names)
                    and all(name in names or name in entry.defaults for name in parameters[positional:]))
        candidates = [entry for entry in entries if accepts(entry)]
        if not candidates:
            raise ValueError('named arguments do not match a callable signature')
        orders = {tuple(name for name, _ in entry.parameters) for entry in candidates}
        if len(orders) != 1:
            raise ValueError('named overloads require an unambiguous parameter order')
        order = next(iter(orders))
        supplied, loans = {}, []
        try:
            # Evaluate in source order before arranging values by parameter name.
            for index, argument in enumerate(arguments):
                named = argument.kind == 'named_argument'
                parameter = argument.token.text if named else order[index]
                term = argument.operands[0] if named else argument
                position = order.index(parameter)
                annotations = set()
                for entry in candidates:
                    annotation = entry.parameters[position][1]
                    if entry.templates:
                        bindings, _ = explicit_bindings(entry, explicit, cfg.context)
                        annotation = substitute(annotation, bindings)
                        if set(re.findall(r'\b[A-Za-z_]\w*\b', annotation or '')) & (type_parameters(entry) - set(bindings)):
                            annotation = None
                    annotations.add(annotation)
                expected = cfg.context.type(next(iter(annotations))) if len(annotations) == 1 and None not in annotations else None
                value = cfg.expr(term, env, expected)
                owner = root_binding(term, cfg, env) or cfg.call_result_owners.get(value.identity)
                supplied[parameter] = Expression('capture', term.token, (ArgumentValue(value, owner),))
                if callable(getattr(value.type, 'release', None)):
                    sources = set(cfg.value_dependencies.get(value.identity, ()))
                    if owner is not None:
                        sources.add(cfg.flow.resolve(owner.identity))
                    for source in sources:
                        key = ('named argument', id(supplied), index, source)
                        cfg.flow.begin_loan(key, source)
                        loans.append(key)
        finally:
            for key in loans:
                cfg.flow.end_loan(key)
        normalized = []
        for parameter in order:
            if parameter in supplied:
                normalized.append(supplied[parameter])
            else:
                defaults = [entry.defaults[parameter] for entry in candidates]
                if len({tuple(token.text for token in value.tokens) for value in defaults}) != 1:
                    raise ValueError('overloaded default arguments are ambiguous')
                normalized.append(defaults[0])
        return select(candidates, tuple(normalized), cfg, env, explicit)
    candidates = [e for e in entries if len(e.parameters) - len(e.defaults) <= len(arguments) <= len(e.parameters)]
    if not candidates:
        raise ValueError('no callable matches the argument count')
    if any(len(e.parameters) != len(arguments) for e in candidates):
        if len(candidates) != 1:
            raise ValueError('default arguments require an unambiguous callable signature')
        entry = candidates[0]
        arguments = (*arguments, *(entry.defaults[name] for name, _ in entry.parameters[len(arguments):]))
    # Evaluate each argument once; use an expected type only when all candidates agree.
    values = []
    evaluation_loans = []
    try:
        for index, argument in enumerate(arguments):
            from py_compiler.syntax.complex.generic import explicit_bindings, substitute, type_parameters
            annotations = set()
            for candidate in candidates:
                annotation = candidate.parameters[index][1]
                if candidate.templates:
                    types, _ = explicit_bindings(candidate, explicit, cfg.context)
                    annotation = substitute(annotation, types)
                    import re
                    if set(re.findall(r'\b[A-Za-z_]\w*\b', annotation or '')) & (type_parameters(candidate) - set(types)):
                        annotation = None
                elif explicit:
                    raise ValueError('callable does not declare template parameters')
                annotations.add(annotation)
            expected = cfg.context.type(next(iter(annotations))) if len(annotations) == 1 and None not in annotations else None
            value = cfg.expr(argument, env, expected)
            values.append(value)
            # Keep earlier argument storage alive until later arguments have been
            # evaluated. A nested next()/mutation must not invalidate a yielded view.
            from py_compiler.syntax.generic.storage import contains_borrowed_fields, owns_storage
            if value is not None and (value.type.family == 'record' or owns_storage(value.type) or contains_borrowed_fields(value.type, cfg.context)):
                owner = cfg.call_result_owners.get(value.identity)
                sources = set(cfg.value_dependencies.get(value.identity, ()))
                if owner is not None:
                    sources.add(cfg.flow.resolve(owner.identity))
                for source in sources:
                    key = (id(values), index, source)
                    cfg.flow.begin_loan(key, source)
                    evaluation_loans.append(key)
    finally:
        for key in evaluation_loans:
            cfg.flow.end_loan(key)
    from py_compiler.syntax.complex.generic import realize
    realized, failures, fallbacks = [], [], set()
    for candidate in candidates:
        try:
            selected = realize(candidate, explicit, values, cfg.context) if candidate.templates else candidate
            realized.append(selected)
            if (candidate.templates and not any(parameter.constraint for parameter in candidate.templates)
                    and all(annotation in type_parameters(candidate) for _, annotation in candidate.parameters)):
                fallbacks.add(id(selected))
        except ValueError as failure:
            failures.append(str(failure))
    candidates = realized
    if not candidates:
        raise ValueError('; '.join(failures))
    def ranks(entry):
        from py_compiler.syntax.primitive.numeric.conversion import integer_conversion_rank
        result = []
        for value, (_, annotation) in zip(values, entry.parameters):
            if annotation is None:
                result.append(0)
                continue
            target = cfg.context.type(annotation)
            if value.type == target:
                result.append(0)
                continue
            if value.type.family in ('tensor', 'tuple') and target.family in ('tensor', 'tuple') and target.accepts_type(value.type):
                result.append(1)
                continue
            conversion_rank = integer_conversion_rank(value, target, cfg)
            if conversion_rank is not None:
                result.append(conversion_rank)
                continue
            if value.type.family == 'record' and implements(cfg.context.by_name[value.type.name], target.name, cfg.context):
                result.append(1)
                continue
            return None
        return tuple(result)
    ranked = [(entry, rank) for entry in candidates if (rank := ranks(entry)) is not None]
    # An entirely unconstrained parameter list is a fallback to applicable contracts.
    specific = [(entry, rank) for entry, rank in ranked if id(entry) not in fallbacks]
    if specific:
        ranked = specific
    # A candidate wins only when no argument has a worse conversion and at
    # least one has a better one. Equally good paths remain ambiguous.
    candidates = [entry for entry, rank in ranked if not any(
        all(a <= b for a, b in zip(other, rank)) and any(a < b for a, b in zip(other, rank))
        for _, other in ranked)]
    if not candidates:
        raise ValueError('no callable matches the argument types')
    if len(candidates) != 1:
        raise ValueError('ambiguous callable overload; use an explicit argument type')
    entry = candidates[0]
    loans = []
    if (id(entry) in cfg.context.compilers and not entry.compiling
            and all(annotation for _, annotation in entry.parameters)):
        cfg.context.compile(entry)
    consumed = set()
    for index, (argument, value, (parameter, _)) in enumerate(zip(arguments, values, entry.parameters)):
        mode = entry.parameter_ownership.get(parameter, getattr(value.type, 'parameter_ownership', 'view'))
        from py_compiler.syntax.generic.storage import root_binding
        binding = (root_binding(argument, cfg, env) if mode == 'view' else
                   argument.operands[0].binding(cfg) if argument.kind == 'capture' else
                   cfg.lookup(argument, env) if argument.kind in ('name', 'reference') else None)
        # An allocating projection (for example list slicing) owns a new
        # value, even when its syntax has an existing root binding.
        if (binding is not None and value is not None and value.identity in cfg.owned_results
                and cfg.values.get(binding.identity) != value):
            binding = None
        if binding is None and value is not None:
            binding = cfg.call_result_owners.get(value.identity)
            if mode == 'borrow' and cfg.call_result_modes.get(value.identity) == 'view':
                raise ValueError('cannot pass a returned view as a mutable borrow')
        if binding is None and value is not None and callable(getattr(value.type, 'release', None)) and value.identity in cfg.owned_results:
            temporary = '<owned argument ' + str(value.identity) + '>'
            binding = Binding(cfg.body.identity + ':' + temporary, temporary, value.type, False, 'own', argument.token.span)
            env[temporary] = binding
            cfg.local_names.add(temporary)
            cfg.values[binding.identity] = value
            cfg.body.bindings.append(binding)
            cfg.record_lifetime('declare', binding.identity, binding.name)
            cfg.value_origins[value.identity] = {binding.identity}
            if not hasattr(cfg, 'temporary_call_arguments'):
                cfg.temporary_call_arguments = {}
            cfg.temporary_call_arguments[value.identity] = (binding, env, cfg.local_names, entry)
        if binding is None and mode in ('view', 'borrow') and callable(getattr(value.type, 'release', None)):
            raise ValueError('memory arguments passed by view or borrow require a live binding')
        if mode == 'borrow' and value.type.family == 'shared':
            raise ValueError('shared references expose read-only payloads')
        if mode == 'move' and binding:
            if (binding.identity in cfg.parameter_bindings
                    and binding.name in getattr(cfg, 'inferred_parameters', {})):
                from dataclasses import replace
                binding = replace(binding, ownership='own')
                env[binding.name] = binding
                cfg.callable_entry.parameter_ownership[binding.name] = 'move'
                cfg.value_sources[value.identity] = set()
            if binding.ownership in ('view', 'borrow'):
                raise ValueError('cannot move owned storage from a view or borrow')
            identity = cfg.flow.resolve(binding.identity)
            if identity in consumed or any(owner == identity for _, owner in loans):
                raise ValueError('call arguments alias moved storage')
            consumed.add(identity)
            cfg.flow.move(binding)
        elif mode in ('copy', 'mirror'):
            copy_value = getattr(value.type, 'copy_value', None)
            if copy_value:
                values[index] = copy_value(value, cfg, argument.token.span)
            elif value.type.binding_ownership != 'copy':
                raise ValueError(f'{value.type.name} has no {mode} passing implementation')
        elif binding:
            identity = cfg.flow.resolve(binding.identity)
            if identity in consumed or any(owner == identity and (mode == 'borrow' or previous == 'borrow') for previous, owner in loans):
                raise ValueError('call arguments have conflicting ownership')
            if mode == 'borrow' and binding.ownership == 'view':
                raise ValueError('cannot pass a view as a mutable borrow')
            if mode == 'borrow':
                from py_compiler.syntax.generic.storage import BindingPlace
                with BindingPlace(binding, rebind=False).capture(cfg, 'borrow'):
                    pass
            loans.append((mode, identity))
    return entry, tuple(values)


def invoke(entry, values, cfg, span, receiver=None):
    if getattr(entry, 'foreign_boundary', False) and not cfg.unsafe_depth:
        raise ValueError('foreign calls require an unsafe block')
    if getattr(cfg, 'spawning_async', False):
        from importlib import import_module
        return import_module('py_compiler.syntax.operator.async').start(entry, values, cfg, span, receiver)
    if getattr(cfg, 'mock_cases', None):
        from py_compiler.syntax.block.mock import replacement, specialize
        if entry.symbol in cfg.mock_cases:
            return replacement(entry, values, cfg, span)
        if not getattr(entry, 'c_symbol', None) and not getattr(entry, 'mock_cases', None):
            entry = specialize(entry, cfg)
    source_scope = getattr(cfg, 'block_scope', None)
    target_scope = cfg.context.callable_scope(entry)
    if source_scope is not None and target_scope is not None and not source_scope.allows(target_scope):
        raise ValueError(f'production scope cannot call test declaration {entry.name!r}')
    provider = cfg.context.compilers.get(id(entry))
    if getattr(provider, 'spelling', '') == '->':
        value = provider.expand_call(entry, values, cfg, span)
        finish_call_arguments(entry, values, cfg, span)
        return value
    cfg.context.compile(entry)
    operands = tuple(coerce(v, cfg.context.type(t), cfg, span) for v, (_, t) in zip(values, entry.parameters))
    if entry.receiver:
        if receiver is None:
            raise ValueError('method call requires a receiver')
        operands = (receiver, *operands)
    result = cfg.context.type(entry.result or 'absent')
    modes = tuple(entry.parameter_ownership.get(parameter, getattr(value.type, 'parameter_ownership', 'view'))
                  for value, (parameter, _) in zip(values, entry.parameters))
    modes = tuple('move' if mode in ('copy', 'mirror') and callable(getattr(value.type, 'release', None)) else ('copy' if mode == 'mirror' else mode)
                  for mode, value in zip(modes, values))
    receiver_modes = (entry.receiver_ownership or 'borrow',) if entry.receiver else ()
    layout = (cfg.context.provider_by_name[result.name].layout(result.declaration, cfg.context)
              if result.family == 'record' and entry.result_ownership == 'own' else '')
    operation = Invoke(entry.symbol, result, receiver_modes + modes, layout, entry)
    if getattr(entry, 'c_symbol', None) is not None:
        from py_compiler.syntax.generic.atom import Atom
        from dataclasses import replace
        operation = Atom(entry, tuple(v.type for v in operands), result if result.mlir else None, receiver_modes + modes, ('call',), replace(entry.c_symbol, result_layout=layout))
    if entry.body is not None:
        for destination, sources in getattr(entry.body, 'retained_parameter_sources', ()):
            target = receiver if destination == -1 else values[destination]
            dependencies = set()
            for index in sources:
                source = receiver if index == -1 else values[index]
                dependencies.update(cfg.value_origins.get(source.identity, ()))
                dependencies.update(cfg.value_dependencies.get(source.identity, ()))
            cfg.value_dependencies.setdefault(target.identity, set()).update(dependencies)
            for owner in cfg.value_origins.get(target.identity, ()):
                cfg.flow.retain(owner, '*', cfg.flow.sources(owner) | dependencies)
    if not result.mlir:
        cfg.effect('call', operands, operation, span)
        finish_call_arguments(entry, values, cfg, span)
        return None
    value = cfg.emit('call', result, operands, operation, span)
    # Only declared return dependencies determine storage lifetime. A stack
    # callable argument does not make an independently allocated result stack
    # storage; retaining that flag suppresses the caller's deallocation.
    cfg.stack_values.discard(value.identity)
    if layout:
        cfg.stack_values.add(value.identity)
    contract = getattr(entry, 'semantic_result', None)
    sources = contract.sources if contract is not None else (entry.body.result_sources if entry.body is not None else ())
    if entry.result_ownership == 'own' and not sources:
        cfg.owned_results.add(value.identity)
    dependencies = [receiver if i == -1 else values[i] for i in sources]
    dependencies = [source for source in dependencies if source is not None]
    cfg.value_sources[value.identity] = set().union(*(cfg.value_sources.get(source.identity, set()) for source in dependencies))
    if any(source.identity in cfg.stack_values for source in dependencies):
        cfg.stack_values.add(value.identity)
    retained = entry.body.retained_result_sources if entry.body is not None else ()
    cfg.value_dependencies[value.identity] = set().union(*(cfg.value_origins.get((receiver if i == -1 else values[i]).identity, set()) |
        cfg.value_dependencies.get((receiver if i == -1 else values[i]).identity, set()) for i in retained))
    finish_call_arguments(entry, values, cfg, span)
    from py_compiler.syntax.complex.error import unwrap_result
    return unwrap_result(value, cfg, span)


def finish_call_arguments(entry, values, cfg, span):
    temporaries = getattr(cfg, 'temporary_call_arguments', {})
    contract = getattr(entry, 'semantic_result', None)
    sources = set(contract.sources if contract is not None else getattr(entry.body, 'result_sources', ()))
    sources.update(getattr(entry.body, 'retained_result_sources', ()))
    for _, retained in getattr(entry.body, 'retained_parameter_sources', ()):
        sources.update(retained)
    for index, value in enumerate(values):
        if value.identity not in temporaries or index in sources:
            continue
        binding, env, names, creator = temporaries[value.identity]
        if creator is not entry or value.identity in getattr(cfg, 'deferred_call_arguments', ()):
            continue
        temporaries.pop(value.identity)
        cfg.cleanup_memory((binding,), span)
        env.pop(binding.name, None)
        names.discard(binding.name)


def dispatch(receiver, method, arguments, cfg, env, span, contract_name=None):
    context = cfg.context
    contract = context.by_name[receiver.type.name if receiver else contract_name]
    methods = [m for m in contract.methods if m.name.rsplit('.', 1)[-1] == method]
    specification, values = select(methods, arguments, cfg, env)
    # Every predicate and selected branch borrows these same arguments. Their
    # implicit owners stay in the caller's scope until dispatch has joined.
    cfg.deferred_call_arguments = getattr(cfg, 'deferred_call_arguments', set()) | {v.identity for v in values}
    # Realize providers from their declared trait contract, even when no provider
    # instance has been constructed by the caller.
    from py_compiler.syntax.block.record import record_header
    from py_compiler.syntax.complex.generic import split_arguments
    for template in tuple(context.record_templates.values()):
        if getattr(getattr(template, 'provider', None), 'spelling', None) != 'class':
            continue
        for pattern in record_header(template.node.header)[2]:
            base = pattern.split('[', 1)[0]
            qualified = context.declaration_name(base, template.scope)
            if qualified != contract.template_name:
                continue
            terms = split_arguments(pattern.split('[', 1)[1][:-1]) if '[' in pattern else ()
            inferred = dict(zip(terms, contract.arguments))
            if all(p.name in inferred or p.default is not None for p in template.parameters):
                template.realize(tuple(inferred[p.name].name if p.name in inferred else p.default for p in template.parameters), context)
    pointer = cfg.emit('extract', context.type('pointer'), (receiver,), Extract(0), span) if receiver else None
    tag = cfg.emit('extract', context.type('i32'), (receiver,), Extract(1), span) if receiver else None
    candidates = []
    for declaration in context.declarations:
        if implements(declaration, contract.name, context):
            for entry in context.functions.get(declaration.name + '.' + method, ()):
                if tuple(context.type(t) for _, t in entry.parameters) == tuple(context.type(t) for _, t in specification.parameters):
                    candidates.append((declaration, entry))
    if not candidates:
        raise ValueError('trait method has no implementation')
    if receiver is None:
        from py_compiler.syntax.function.contracts import integer_dispatch_domain
        domains = [integer_dispatch_domain(entry, context) for _, entry in candidates]
        for index, left in enumerate(domains):
            if left is None:
                continue
            for right in domains[index + 1:]:
                if right is None:
                    continue
                if any(low > high for low, high in (*left.values(), *right.values())):
                    continue
                if all(max(left[key][0], right[key][0]) <= min(left[key][1], right[key][1]) for key in left.keys() & right.keys()):
                    raise ValueError('statically ambiguous trait dispatch: overlapping integer domains')
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
            from py_compiler.syntax.complex.object import BooleanAnd
            flag = cfg.emit('and', context.type('bool'), (flag, condition), BooleanAnd(), span)
        flags.append(flag)
    from py_compiler.syntax.complex.object import CountBoolean
    count = literal_value(0, context.type('i32'), cfg, span)
    for flag in flags:
        count_value = cfg.emit('count', context.type('i32'), (flag,), CountBoolean(), span)
        count = cfg.emit('binary', context.type('i32'), (count, count_value), '+', span)
    one = literal_value(1, context.type('i32'), cfg, span)
    valid = cfg.emit('binary', context.type('bool'), (count, one), '==', span)
    cfg.effect('assert', (valid,), Assert(f'trait dispatch requires exactly one match: {contract.name}.{method} at {cfg.source.path}:{span.start}'), span)
    result_type = context.type(specification.result or 'absent')
    join = cfg.block((result_type,) if result_type.mlir else ())
    result_sources = set()
    result_stack = False
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
        # A concrete method may infer a view of one of its fields. The boxed
        # trait's owned result contract must return independent storage.
        if value is not None and specification.result_ownership == 'own' and entry.body.result_sources:
            from py_compiler.syntax.generic.storage import owns_storage, copy_storage
            if owns_storage(value.type):
                value = copy_storage(value, cfg, span)
                cfg.value_sources[value.identity] = set()
        if value is not None:
            result_sources.update(cfg.value_sources.get(value.identity, ()))
            result_stack = result_stack or value.identity in cfg.stack_values
        cfg.current.terminator = Terminator('jump', (Edge(join.identity, (value,) if value else ()),))
        if following:
            cfg.current = following
    cfg.current = join
    if join.parameters and specification.result_ownership in ('borrow', 'view'):
        result = join.parameters[0]
        cfg.value_sources[result.identity] = result_sources
        if result_stack:
            cfg.stack_values.add(result.identity)
        owners = []
        for _, entry in candidates:
            for index in entry.body.result_sources:
                if index >= 0:
                    argument = arguments[index]
                    from py_compiler.syntax.generic.storage import root_binding
                    owner = root_binding(argument, cfg, env) or cfg.call_result_owners.get(values[index].identity)
                    if owner is not None:
                        owners.append(owner)
        if not owners or len({owner.identity for owner in owners}) != 1:
            raise ValueError('returned trait borrow requires one source owner')
        cfg.call_result_owners[result.identity] = owners[0]
        cfg.call_result_modes[result.identity] = specification.result_ownership
    elif join.parameters and result_type.family in ("record", "trait", "union") and (receiver is None or receiver.identity in cfg.stack_values):
        cfg.stack_values.add(join.parameters[0].identity)
    if join.parameters and specification.result_ownership == 'own':
        cfg.owned_results.add(join.parameters[0].identity)
        cfg.stack_values.discard(join.parameters[0].identity)
    return join.parameters[0] if join.parameters else None


def lower_expression(node, cfg, env, expected=None):
    context, span = cfg.context, node.token.span
    if node.kind == 'member':
        subject = node.operands[0]
        if subject.kind == 'name' and subject.token.text in context.package_imports:
            key = subject.token.text + '.' + node.token.text
            place = context.imported_globals.get(key)
            if place is None:
                raise ValueError(f'package has no exported global {key!r}')
            value = place.read(cfg, span)
            from py_compiler.syntax.generic.storage import root_binding
            owner = root_binding(node, cfg, env)
            cfg.values[owner.identity] = value
            if value.type.binding_ownership != 'copy':
                cfg.call_result_owners[value.identity] = owner
                cfg.call_result_modes[value.identity] = 'view'
            return coerce(value, expected, cfg, span)
        if subject.kind == 'name':
            declared = context.types.get(context.declaration_name(subject.token.text, cfg.declaration_scope))
            if callable(getattr(declared, 'variant', None)):
                return coerce(literal_value(declared.variant(node.token.text), declared, cfg, span), expected, cfg, span)
        receiver = cfg.expr(node.operands[0], env)
        provider = receiver.type if callable(getattr(receiver.type,'field_place',None)) else context.provider_by_name.get(receiver.type.name, receiver.type)
        if not hasattr(provider, 'field_place'):
            raise ValueError('type has no field access provider')
        value = provider.field_place(receiver, node.token.text, cfg, span).read(cfg, span)
        if receiver.identity in cfg.call_result_owners and value.type.binding_ownership != 'copy':
            cfg.call_result_owners[value.identity] = cfg.call_result_owners[receiver.identity]
            cfg.call_result_modes[value.identity] = cfg.call_result_modes.get(receiver.identity, 'view')
            if receiver.identity in cfg.call_result_sources:
                cfg.call_result_sources[value.identity] = cfg.call_result_sources[receiver.identity]
        return coerce(value, expected, cfg, span)
    callee, *arguments = node.operands
    if any(argument.kind == 'spread' for argument in arguments):
        arguments = expand_spreads(arguments, cfg, env)
        from dataclasses import replace
        node = replace(node, operands=(callee, *arguments))
    if callee.kind in ('name', 'reference'):
        binding = cfg.lookup(callee, env)
        if binding is not None and binding.type.family == 'function':
            receiver = cfg.expr(callee, env)
            result = receiver.type.call(receiver, arguments, cfg, env, span)
            return coerce(result, expected, cfg, span) if expected else result
    from py_compiler.syntax.function.trait_dispatch import resolve_default_call
    selected = resolve_default_call(callee, arguments, cfg, env, span)
    if selected is not None:
        return coerce(selected, expected, cfg, span) if expected else selected
    imported = callee.operands[0] if callee.kind == 'index' else callee
    if imported.kind == 'member' and imported.operands[0].kind == 'name' and imported.operands[0].token.text in context.package_imports:
        from py_compiler.syntax.grammar.expression import Expression
        from dataclasses import replace
        qualified = imported.operands[0].token.text + '.' + imported.token.text
        resolved = Expression('name', imported.token.with_text(qualified))
        callee = replace(callee, operands=(resolved, callee.operands[1])) if callee.kind == 'index' else resolved
    from py_compiler.syntax.prelude import builtin_call_providers
    base, argument = (callee.operands if callee.kind == 'index' else (callee, None))
    definition = context.types.get(base.token.text) if base.kind == 'name' else None
    if callable(getattr(definition, 'construct_expression', None)):
        return coerce(definition.construct_expression(argument, arguments, cfg, env, span, expected), expected, cfg, span)
    if callable(getattr(definition, 'construct', None)):
        from py_compiler.syntax.grammar.expression import type_spelling
        terms = argument.operands if argument and argument.kind == 'template_arguments' else ((argument,) if argument else ())
        return coerce(definition.construct(tuple(type_spelling(term) for term in terms), arguments, cfg, env, span, expected), expected, cfg, span)
    builtin = builtin_call_providers().get(base.token.text) if base.kind == 'name' else None
    if builtin is not None:
        from py_compiler.syntax.grammar.expression import type_spelling
        if callable(getattr(builtin, 'resolve_arguments', None)):
            type_argument = builtin.resolve_arguments(argument, context)
        else:
            type_argument = context.type(type_spelling(argument)) if argument is not None else None
        return coerce(builtin.call(type_argument, arguments, cfg, env, span), expected, cfg, span) if expected else builtin.call(type_argument, arguments, cfg, env, span)
    if callee.kind == 'index' and base.kind == 'name' and base.token.text == 'pointer':
        if len(arguments) != 1:
            raise ValueError('pointer[T] expects one array')
        from py_compiler.syntax.grammar.expression import type_spelling
        target = context.type('pointer[' + type_spelling(argument) + ']')
        return target.reinterpret(cfg.expr(arguments[0], env), cfg, span)
    explicit = ()
    if callee.kind == 'index':
        from py_compiler.syntax.grammar.expression import type_spelling
        terms = argument.operands if argument.kind == 'template_arguments' else (argument,)
        explicit = tuple(type_spelling(term) for term in terms)
        callee = base
    receiver = None
    receiver_binding = None
    if callee.kind == 'name':
        name = callee.token.text
        local_type = cfg.lookup(callee, env)
        if local_type is not None and local_type.type.family == 'type':
            declared = local_type.type.declaration
            provider = context.provider_by_name.get(declared.name)
            if provider is None or not hasattr(provider, 'instantiate'):
                raise ValueError('type binding has no constructor')
            return provider.instantiate(declared.declaration, arguments, cfg, env, span)
        bound = context.template_types.get(name)
        declaration_name = bound.name if bound is not None else context.declaration_name(name, cfg.declaration_scope)
        if declaration_name in context.record_templates:
            template = context.record_templates[declaration_name]
            if not explicit and expected is not None and getattr(getattr(expected, 'declaration', None), 'template_name', None) == template.name:
                explicit = tuple(t.name for t in expected.generic_arguments)
            if not explicit and any(p.default is None for p in template.parameters):
                inferred = {}
                fields = [child for child in template.node.children if child.provider is None and child.tokens and not child.tokens[0].text.startswith('_memory_')]
                for child, value in zip(fields, arguments):
                    tokens = child.tokens
                    if len(tokens) >= 3 and tokens[1].text == ':' and tokens[2].text in {p.name for p in template.parameters}:
                        annotation = tokens[2].text
                        if value.kind in ('name', 'reference'):
                            binding = cfg.lookup(value, env)
                            actual = binding.type if binding else None
                        elif value.kind == 'call' and value.operands[0].kind == 'name':
                            actual = context.types.get(value.operands[0].token.text)
                        elif value.kind == 'literal':
                            from py_compiler.frontend.parser.contract import Literal
                            from py_compiler.hir.hir.src.program import resolve_literal
                            actual, _ = resolve_literal(Literal('', value.token.span, value.token.text, value.token.kind, None), cfg.syntax, context)
                        else:
                            actual = None
                        if actual is not None:
                            if annotation in inferred and inferred[annotation] != actual.name:
                                raise ValueError('conflicting class field type inference')
                            inferred[annotation] = actual.name
                explicit = tuple(inferred.get(p.name, p.default) for p in template.parameters)
                if any(value is None for value in explicit):
                    raise ValueError('class field types require explicit or inferable type arguments')
            target = template.realize(explicit, context)
            provider = context.provider_by_name[target.name]
            return coerce(provider.instantiate(target.declaration, arguments, cfg, env, span), expected, cfg, span)
        provider = context.provider_by_name.get(declaration_name)
        if hasattr(provider, 'instantiate'):
            value = provider.instantiate(context.by_name[declaration_name], arguments, cfg, env, span)
            variant = getattr(context, 'enum_variants', {}).get(declaration_name)
            if variant is not None:
                value = coerce(value, variant, cfg, span)
            return coerce(value, expected, cfg, span)
        if name in context.bound_syntax.types and not context.lookup_functions(name, getattr(cfg, 'declaration_scope', ())):
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
        from py_compiler.syntax.function.trait_dispatch import qualified
        type_subject = qualified(subject.operands[0]) if subject.kind == 'index' else ''
        if (subject.kind == 'index' and type_subject and
                (subject.operands[0].kind == 'name' and cfg.lookup(subject.operands[0], env) is None or
                 subject.operands[0].kind == 'member' and type_subject in context.record_templates)):
            from py_compiler.syntax.grammar.expression import type_spelling
            declared = context.type(type_spelling(subject))
            if declared.family == 'trait':
                value = dispatch(None, callee.token.text, arguments, cfg, env, span, declared.name)
                return coerce(value, expected, cfg, span) if expected else value
        if subject.kind == 'name':
            name = context.declaration_name(subject.token.text, cfg.declaration_scope)
            declared = context.template_types.get(subject.token.text, context.types.get(name))
            if declared is not None and cfg.lookup(subject, env) is None:
                if declared.family in ('integer', 'float', 'unit', 'byte') and callee.token.text == 'zero' and not arguments:
                    return coerce(literal_value(0, declared, cfg, span), expected, cfg, span)
                if declared.family == 'record':
                    from py_compiler.syntax.complex.collection import static_method
                    return coerce(static_method(declared, callee.token.text, arguments, cfg, env, span), expected, cfg, span)
            if declared and declared.family == 'trait':
                value = dispatch(None, callee.token.text, arguments, cfg, env, span, name)
                return coerce(value, expected, cfg, span) if expected else value
        from py_compiler.syntax.generic.storage import root_binding
        receiver_binding = root_binding(subject, cfg, env)
        receiver = cfg.expr(subject, env)
        receiver_binding = receiver_binding or cfg.call_result_owners.get(receiver.identity)
        returned_mode = cfg.call_result_modes.get(receiver.identity)
        if receiver_binding is not None and returned_mode == 'view':
            from dataclasses import replace
            receiver_binding = replace(receiver_binding, ownership='view')
        if receiver_binding is None and callable(getattr(receiver.type, 'release', None)):
            # Retain an implicit owner in this lexical scope. Returned views can
            # borrow it, while the ordinary scope/return cleanup destroys it.
            temporary = f'<temporary receiver {receiver.identity}>'
            receiver_binding = Binding(cfg.body.identity + ':' + temporary, temporary,
                                       receiver.type, False, 'own', span)
            env[temporary] = receiver_binding
            cfg.local_names.add(temporary)
            cfg.values[receiver_binding.identity] = receiver
            cfg.body.bindings.append(receiver_binding)
            cfg.record_lifetime('declare', receiver_binding.identity, receiver_binding.name)
        from py_compiler.syntax.complex.collection import primitive_method, collection_method
        primitive = primitive_method(receiver, callee.token.text, arguments, cfg, env, span, receiver_binding)
        if primitive is not None:
            # Independent scalar results need no receiver past this expression.
            if receiver_binding is not None and receiver_binding.name.startswith('<temporary receiver '):
                cfg.cleanup_memory((receiver_binding,), span)
                env.pop(receiver_binding.name, None)
                cfg.local_names.discard(receiver_binding.name)
            return coerce(primitive, expected, cfg, span)
        collection = collection_method(receiver, callee.token.text, arguments, cfg, env, span, receiver_binding)
        if collection is not None:
            return coerce(collection, expected, cfg, span)
        method = getattr(receiver.type, 'call_method', None)
        if method is not None:
            value = method(receiver, callee.token.text, arguments, cfg, env, span, receiver_binding)
            if receiver.type.family == 'string' and receiver_binding.name.startswith('<temporary receiver '):
                # Standard string methods return independent storage or scalars.
                cfg.cleanup_memory((receiver_binding,), span)
                env.pop(receiver_binding.name, None)
                cfg.local_names.discard(receiver_binding.name)
            return coerce(value, expected, cfg, span) if expected else value
        if receiver.type.family == 'record' and callee.token.text in ('get', 'set') and not context.functions.get(receiver.type.name + '.' + callee.token.text):
            from py_compiler.syntax.complex.fields import access
            return coerce(access(receiver, callee.token.text, arguments, cfg, env, span, receiver_binding), expected, cfg, span)
        if receiver.type.family == 'trait':
            value = dispatch(receiver, callee.token.text, arguments, cfg, env, span)
            return coerce(value, expected, cfg, span) if expected else value
        if callee.token.text.startswith('__'):
            raise ValueError('private method is inaccessible outside its receiver')
        entries = context.functions.get(receiver.type.name + '.' + callee.token.text, ())
    else:
        raise ValueError('expression does not provide a callable')
    entry, values = select(entries, arguments, cfg, env, explicit)
    if entry.receiver and entry.receiver_ownership is None:
        cfg.context.compile(entry)
    if receiver is not None and receiver_binding is None:
        receiver_binding = getattr(cfg, 'self_binding', None)
    if receiver_binding and (entry.receiver_ownership or 'borrow') == 'borrow':
        if cfg.call_result_modes.get(receiver.identity) == 'view':
            raise ValueError('cannot call a mutating method through a returned view')
        if receiver_binding.ownership == 'view':
            raise ValueError('cannot call a mutating method through a view')
        from py_compiler.syntax.generic.storage import BindingPlace
        from py_compiler.syntax.complex.collection import preserves_shape
        stable = bool(cfg.flow.shape_loans) and preserves_shape(entry, cfg.context)
        borrowers = cfg.flow.borrowers(receiver_binding.identity)
        locked_tasks = {b.identity for b in cfg.body.bindings
                        if b.type.family == 'task' and b.type.locked}
        serialized = (getattr(cfg, 'spawning_async', False) and 'lock' in cfg.async_options
                      and borrowers <= locked_tasks and not cfg.flow.loans)
        if not serialized:
            with BindingPlace(receiver_binding, rebind=False).capture(cfg, 'borrow', preserve_shape=stable):
                pass
    value = invoke(entry, values, cfg, span, receiver)
    if (receiver_binding is not None and receiver_binding.name.startswith('<temporary receiver ')
            and not getattr(entry.body, 'result_sources', ())
            and -1 not in getattr(entry.body, 'retained_result_sources', ())
            and not any(-1 in sources for _,sources in getattr(entry.body,'retained_parameter_sources',()))):
        cfg.cleanup_memory((receiver_binding,),span)
        env.pop(receiver_binding.name,None)
        cfg.local_names.discard(receiver_binding.name)
    if value is not None and entry.body is not None and entry.body.result_sources:
        owners = []
        for index in entry.body.result_sources:
            if index == -1:
                if receiver_binding is not None:
                    owners.append(receiver_binding)
                continue
            if index >= len(arguments):
                continue
            argument = arguments[index]
            from py_compiler.syntax.generic.storage import root_binding
            temporary = getattr(cfg, 'temporary_call_arguments', {}).get(values[index].identity)
            owner = (temporary[0] if temporary is not None else
                     root_binding(argument, cfg, env) or cfg.call_result_owners.get(values[index].identity))
            if owner is not None:
                owners.append(owner)
        if owners:
            cfg.call_result_owners[value.identity] = owners[0]
            if len({owner.identity for owner in owners}) > 1:
                cfg.call_result_sources[value.identity] = frozenset(owner.identity for owner in owners)
            if entry.result_ownership in ('borrow', 'view'):
                cfg.call_result_modes[value.identity] = entry.result_ownership
    return coerce(value, expected, cfg, span) if expected else value


import unittest


class BorrowedTraitRoutingTests(unittest.TestCase):

    def test_c_owned_slice_arguments_are_released_after_borrowed_calls(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''from collections import list, enumerate

def length(values: list[int]) -> int:
    return len(values)

def keep(values: list[int]) -> view list[int]:
    return view values

test "borrowed slice arguments release their independent allocation":
    values = [4, 2, 9]
    repeats = 0
    while repeats < 50:
        assert(length(values[1:]) == 2)
        for index, value in enumerate(values[1:]):
            assert(value == values[index + 1])
        repeats += 1
    assert(values == [4, 2, 9])

test "returned views retain temporary slice arguments":
    values = [4, 2, 9]
    saved = keep(values[1:])
    assert(len(saved) == 2)
    assert(saved[0] == 2 and saved[1] == 9)
    assert(values == [4, 2, 9])
'''
        with tempfile.TemporaryDirectory(prefix='sev-slice-arguments-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/main.sev').write_text(source)
            library = Path(__file__).resolve().parents[3] / 'library/core/collections/collection'
            (root / 'package.json').write_text(json.dumps({
                'package': {'name': 'slice-arguments', 'version': '0.1.0'},
                'dependencies': {'collections': {'path': str(library)}},
            }))
            result = test_package(root, memory=True, coverage=True)
            report = json.loads(Path(result['report']).read_text()) if result.get('report') else result
            failures = [{key: item.get(key) for key in ('label', 'stdout', 'stderr', 'error', 'exit', 'memory')}
                        for item in report.get('tests', ()) if not item.get('passed')]
            self.assertTrue(result['success'], failures or result)

    def test_c_variadic_packs_and_spread_evaluate_once(self):
        import json
        import tempfile
        from pathlib import Path
        from py_compiler.packaging.testing import test_package
        source = '''calls: int = 0
def amount(value: int) -> int:
    return value
def amount(value: string) -> int:
    return len(value)
def total(values...) -> int:
    result = 0
    for value in values:
        result += amount(value)
    return result
def arguments() -> tuple[string,int]:
    module.calls += 1
    return ("abc", 4)
test "heterogeneous and empty packs":
    assert(total(1, "abc", 2) == 6)
    assert(total() == 0)
test "spread retains source evaluation order":
    assert(total(...(0..4)) == 6)
    module.calls = 0
    assert(total(1, ...arguments(), 2) == 10)
    assert(module.calls == 1)
test "fixed sequence spreads":
    values = tuple[int,3](1,2,3)
    assert(total(...values) == 6)
'''
        with tempfile.TemporaryDirectory(prefix='sev-variadic-') as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.sev').write_text(source)
            (root / 'package.json').write_text(json.dumps({'package': {'name': 'variadic', 'version': '0.1.0'}}))
            result = test_package(root)
            self.assertTrue(result['success'], result)

    def test_field_view_arguments_retain_their_record_owner(self):
        source = ('class Buffer:\n    data: tensor[int]\n'
                  'def items(data: tensor[int]) -> view tensor[int]:\n    return view data\n')
        valid = self.compile(source + 'def work() -> int:\n'
            '    buffer = Buffer(tensor[int](2))\n'
            '    values = items(buffer.data)\n    return len(values)\n')
        self.assertFalse(valid.diagnostics, str(valid.diagnostics))
        invalid = self.compile(source + 'def work() -> view tensor[int]:\n'
            '    buffer = Buffer(tensor[int](2))\n    return items(buffer.data)\n')
        self.assertTrue(invalid.diagnostics)
        self.assertIn('lifetime', str(invalid.diagnostics))

    def test_selected_provider_type_constructs_its_record(self):
        source = '''trait Operation:
    def apply(value: int) -> int
class Double: Operation
    def apply(value: int) -> int:
        return value * 2
def apply[F: Operation = Double](value: int) -> int:
    operation = F()
    return operation.apply(value)
def work() -> int:
    return apply(3)
'''
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        self.assertTrue(any(body.declaration == 'Double.apply' for body in result.program.bodies))

    def test_owning_temporary_receivers_live_through_saved_views(self):
        source = ('class Buffer:\n    data: array[int]\n'
                  '    def size(self: view Self) -> int:\n        return len(self.data)\n'
                  '    def values(self: view Self) -> view array[int]:\n        return self.data\n'
                  'def work() -> int:\n    count = Buffer(array[int,2]()).size()\n'
                  '    saved = Buffer(array[int,3]()).values()\n    return count + len(saved)\n')
        result = self.compile(source)
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        from py_compiler.mlir.src.lib import lower, render, verify_c
        text = render(lower(result.program))
        self.assertEqual(text.count('memref.dealloc'), 2)
        verify_c(text)

    def test_view_of_temporary_cannot_escape_its_scope(self):
        source = ('class Buffer:\n    data: array[int]\n'
                  '    def values(self: view Self) -> view array[int]:\n        return self.data\n'
                  'def invalid() -> view array[int]:\n    return Buffer(array[int,1]()).values()\n')
        result = self.compile(source)
        self.assertTrue(result.diagnostics)
        self.assertIn('lifetime', str(result.diagnostics))

    def test_owned_tensor_result_is_freed_with_a_stack_callable_argument(self):
        result = self.compile('class Provider:\n    marker: int\ndef create(provider: Provider) -> tensor[int]:\n    unsafe:\n        return allocate[int](2)\ndef work():\n    provider = Provider(0)\n    values = create(provider)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        from py_compiler.mlir.src.cfg import render_body
        body = next(body for body in result.program.bodies if body.declaration == 'work')
        self.assertIn('memref.dealloc', render_body(body))

    def test_source_function_can_share_a_type_name(self):
        result = self.compile('def any(value: int) -> bool:\n    return value != 0\n'
                              'def check() -> bool:\n    return any(3)\n')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        body = next(body for body in result.program.bodies if body.declaration == 'check')
        self.assertTrue(any(operation.kind == 'call' for block in body.blocks for operation in block.operations))

    source = '''class Values[T]:
    item: T
    count: int
    def sort(self: borrow Self) -> borrow Values[T]:
        return Sort[T].sort(self)
trait Sort[T]:
    def sort(values: borrow Values[T]) -> borrow Values[T]
class Small[T]: Sort[T]
    def sort(values: borrow Values[T]) -> borrow Values[T] with {values.count <= 32,}:
        return borrow values
class Large[T]: Sort[T]
    def sort(values: borrow Values[T]) -> borrow Values[T] with {values.count > 32,}:
        return borrow values
'''

    def compile(self, source, **options):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        return compile_source('routing.sev', source, Syntax(include_tests=False), **options)

    def test_realizes_both_providers_and_preserves_borrowed_abi(self):
        result = self.compile(self.source + '''def route(count: int) -> int:
    values = Values[int](7, count)
    alias = values.sort()
    alias.item = 9
    return values.item
''')
        self.assertFalse(result.diagnostics, str(result.diagnostics))
        providers = [body for body in result.program.bodies if body.declaration in ('Small[i64].sort', 'Large[i64].sort')]
        self.assertEqual(len(providers), 2)
        for body in providers:
            self.assertEqual(body.result_sources, (0,))
            self.assertEqual(body.result_ownership, 'borrow')
            self.assertEqual(body.result_layout, '')

    def test_imported_trait_templates_keep_their_namespace(self):
        producer = self.compile(self.source)
        self.assertFalse(producer.diagnostics, str(producer.diagnostics))
        package = {'package': {'name': 'routing'}, 'templates': producer.templates,
                   'types': [], 'exports': [], 'objects': []}
        result = self.compile('''import routing
def route() -> int:
    values = routing.Values[int](7, 33)
    alias = routing.Sort[int].sort(values)
    alias.item = 9
    return values.item
''', c_packages={'routing': package})
        self.assertFalse(result.diagnostics, str(result.diagnostics))

    def test_known_overlap_is_a_static_diagnostic(self):
        source = self.source.replace('values.count > 32', 'values.count >= 32')
        result = self.compile(source + 'def route():\n    values = Values[int](7, 32)\n    values.sort()\n')
        self.assertTrue(result.diagnostics)
        self.assertIn('overlapping integer domains', str(result.diagnostics[0]))

    def test_local_owner_cannot_escape_through_trait_dispatch(self):
        result = self.compile(self.source + '''def invalid() -> borrow Values[int]:
    values = Values[int](7, 2)
    return values.sort()
''')
        self.assertTrue(result.diagnostics)
        self.assertIn('lifetime', str(result.diagnostics[0]))


class GenericFallbackTests(unittest.TestCase):
    def test_concrete_overload_precedes_generic_with_equal_conversions(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        result = compile_source('fallback.sev', """def count(value: int) -> int:
    return value
def count[T](value: view T) -> int:
    return len(value)
def work() -> int:
    values = tensor[int](3)
    return count(4) + count(i32(4)) + count(values)
""", Syntax(include_tests=False))
        self.assertFalse(result.diagnostics, result.diagnostics)
        self.assertEqual(len([body for body in result.program.bodies if body.declaration == 'count']), 2)
