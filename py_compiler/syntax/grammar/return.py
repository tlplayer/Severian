from py_compiler.syntax.grammar.expression import expression
from py_compiler.mir.cfg.cfg import Terminator


class Return:
    def matches(self, node):
        return node.tokens[0].text == "return"

    def lower(self, node, cfg, env, local_names):
        expected = cfg.body.result_type
        if expected is None:
            value = cfg.expr(expression(node.tokens[1:]), env) if len(node.tokens) > 1 else None
            cfg.body.result_type = value.type if value else cfg.syntax.types["absent"]
        else:
            value = cfg.expr(expression(node.tokens[1:]), env, expected) if len(node.tokens) > 1 else None
        if value and value.identity in cfg.stack_values:
            raise ValueError("cannot return an object/view whose storage belongs to this callable")
        if value and callable(getattr(value.type, 'release', None)):
            term = expression(node.tokens[1:])
            binding = cfg.lookup(term, env) if term.kind in ('name', 'reference') else None
            if binding:
                if binding.ownership in ('view', 'borrow'):
                    raise ValueError('returning a memory view requires a result lifetime contract')
                cfg.flow.move(binding)
            elif term.kind != 'call':
                raise ValueError('memory return requires ownership')
            elif term.operands[0].kind == 'index' and term.operands[0].operands[0].token.text == 'pointer':
                raise ValueError('returning a pointer view requires a result lifetime contract')
        if cfg.phases:
            cfg.phases.leave()
        cfg.cleanup_memory((binding for scope in cfg.active_scopes for binding in scope.values()), node.span)
        cfg.current.terminator = Terminator("finish", value=value)
