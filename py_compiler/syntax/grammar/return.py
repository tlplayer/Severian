from py_compiler.syntax.grammar.expression import expression
from py_compiler.mir.cfg.cfg import Terminator


class Return:
    def matches(self, node):
        return node.tokens[0].text == "return"

    def lower(self, node, cfg, env, local_names):
        if cfg.return_handler is not None:
            return cfg.return_handler(node, env)
        if cfg.body.result_ownership in ('borrow', 'view'):
            term = expression(node.tokens[1:])
            explicit = term.token.text if term.kind == 'unary' else None
            if explicit in ('borrow', 'view'):
                term = term.operands[0]
            value = cfg.expr(term, env, cfg.body.result_type)
            binding = cfg.lookup(term, env) if term.kind in ('name', 'reference') else cfg.call_result_owners.get(value.identity)
            mode = explicit or cfg.call_result_modes.get(value.identity) or (binding.ownership if binding else None)
            if cfg.body.result_ownership == 'borrow' and (mode == 'view' or (binding and binding.ownership == 'view')):
                raise ValueError('cannot return a mutable borrow through a view')
            sources = cfg.value_sources.get(value.identity, set())
            if not sources or value.identity in cfg.stack_values:
                raise ValueError('returned borrow escapes the lifetime of its source binding')
            cfg.body.result_sources = tuple(sorted(set(cfg.body.result_sources) | sources))
            for loop in reversed(cfg.loops):
                leave = getattr(loop, 'leave', None)
                if leave:
                    leave(cfg)
            if cfg.phases:
                cfg.phases.leave()
            cfg.cleanup_memory((b for scope in cfg.active_scopes for b in scope.values()), node.span)
            cfg.current.terminator = Terminator('finish', value=value, span=node.span)
            return
        cfg.record_retained_parameters()
        expected = cfg.body.result_type
        if expected is None:
            value = cfg.expr(expression(node.tokens[1:]), env) if len(node.tokens) > 1 else None
            cfg.body.result_type = value.type if value else cfg.syntax.types["absent"]
        else:
            value = cfg.expr(expression(node.tokens[1:]), env, expected) if len(node.tokens) > 1 else None
        if value:
            dependencies = cfg.value_dependencies.get(value.identity, set())
            if dependencies:
                if any(source not in cfg.parameter_bindings for source in dependencies):
                    raise ValueError('returned value retains a borrow of local storage; storage belongs to this callable')
                cfg.body.retained_result_sources = tuple(sorted(set(cfg.body.retained_result_sources) |
                    {cfg.parameter_bindings[source] for source in dependencies}))
                if value.type.family in ('iterator', 'array'):
                    cfg.value_sources[value.identity] = set()
        if value and value.type.family == 'record':
            if value.identity in cfg.call_result_modes:
                raise ValueError('returning a borrowed result by value requires an explicit copy')
            term = expression(node.tokens[1:])
            binding = cfg.lookup(term, env) if term.kind in ('name', 'reference') else None
            if binding and binding.ownership in ('view', 'borrow'):
                raise ValueError('returning a class by value requires ownership; use an explicit copy')
            if term.kind not in ('name', 'call'):
                raise ValueError('returning a class field requires an explicit owned copy')
            # Borrowed fields use the parameter-source contract recorded above;
            # the record itself is returned through the ordinary owned-value ABI.
            cfg.stack_values.discard(value.identity)
            cfg.value_sources[value.identity] = set()
        if value and value.identity in cfg.stack_values:
            raise ValueError("cannot return an object/view whose storage belongs to this callable")
        if value:
            cfg.body.result_sources = tuple(sorted(set(cfg.body.result_sources) | cfg.value_sources.get(value.identity, set())))
        if value and callable(getattr(value.type, 'release', None)):
            term = expression(node.tokens[1:])
            binding = cfg.lookup(term, env) if term.kind in ('name', 'reference') else None
            if value.identity in cfg.owned_results and (binding is None or binding.type != value.type):
                pass
            elif binding:
                if binding.ownership in ('view', 'borrow'):
                    if not cfg.value_sources.get(value.identity):
                        raise ValueError('returning a memory view requires a result lifetime contract')
                else:
                    cfg.flow.move(binding)
            elif term.kind != 'call':
                raise ValueError('memory return requires ownership')
            elif term.operands[0].kind == 'index' and term.operands[0].operands[0].token.text == 'pointer' and not cfg.value_sources.get(value.identity):
                raise ValueError('returning a pointer view requires a result lifetime contract')
        for loop in reversed(cfg.loops):
            leave = getattr(loop, 'leave', None)
            if leave:
                leave(cfg)
        if cfg.phases:
            cfg.phases.leave()
        cfg.cleanup_memory((binding for scope in cfg.active_scopes for binding in scope.values()), node.span)
        cfg.current.terminator = Terminator("finish", value=value, span=node.span)
