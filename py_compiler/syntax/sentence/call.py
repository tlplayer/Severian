from py_compiler.syntax.sentence.syntax import expression
from py_compiler.syntax.type.objects import Assert


class Call:
    def matches(self, node):
        return len(node.tokens) > 1 and not any(t.text in ('=', ':=', '+=', '-=', '*=') for t in node.tokens)

    def lower(self, node, cfg, env, local_names):
        if node.tokens[0].text == 'assert':
            value = cfg.expr(expression(node.tokens[1:]), env, cfg.syntax.types['bool'])
            cfg.effect('assert', (value,), Assert(f'assertion failed at {cfg.source.path}:{cfg.source.position(node.span.start)[0]}'), node.span)
            return
        term = expression(node.tokens)
        if term.kind != 'call':
            raise ValueError('expression statement requires a call')
        cfg.expr(term, env)
