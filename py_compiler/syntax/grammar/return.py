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
        if cfg.phases:
            cfg.phases.leave()
        cfg.current.terminator = Terminator("finish", value=value)
