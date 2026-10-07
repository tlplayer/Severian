from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation
from py_compiler.mir.cfg.cfg import Edge, Terminator


class Logical(ScalarOperation):
    def lower_expression(self, cfg, node, env):
        left = cfg.expr(node.operands[0], env, self.owner)
        rhs, join = cfg.block(), cfg.block((self.owner,))
        direct = Edge(join.identity, (left,))
        edges = (Edge(rhs.identity), direct) if self.spelling == 'and' else (direct, Edge(rhs.identity))
        cfg.current.terminator = Terminator('conditional', edges, left)
        cfg.current = rhs
        right = cfg.expr(node.operands[1], env, self.owner)
        cfg.current.terminator = Terminator('jump', (Edge(join.identity, (right,)),))
        cfg.current = join
        return join.parameters[0]
