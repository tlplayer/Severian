"""Boolean expressions and their effects.

Implemented evaluation contract:
    A and B -> evaluate A once; evaluate B only when A is true.
    A or B  -> evaluate A once; evaluate B only when A is false.
    Both operands must satisfy bool, including a statically skipped operand.
    Reached effects occur left to right; skipped calls perform no mutation,
    borrow or consumption. Static ownership joins conservatively account for
    every CFG edge unless that edge has been proven unreachable.
    not x == y -> not (x == y), with x and y each evaluated once.
    a() < b() < c() -> save a(), save b(), compare; only on success call c()
    and compare the saved b(). No operand is duplicated or speculated.

Type-predicate contract (pending implementation; acceptance cases in block/if.py):
    x is T tests the active variant against a resolved type T. It reads x once
    and returns bool; it does not convert, copy, consume or extend a loan.
    This type form is distinct from the value-identity form documented for is.
    True edge: possible_types(x) intersect variants(T).
    False edge: possible_types(x) minus variants(T).
    is not and not swap the true/false facts; comparisons bind before not.
    For A and B, check B under A.true; result.true = B.true;
        result.false = join(A.false, B.false).
    For A or B, check B under A.false; result.false = B.false;
        result.true = join(A.true, B.true).
    Joins union possible types, possible moves and borrow dependencies, and
    intersect definite initialization; terminating/unreachable edges contribute
    no facts. An else after `x is int and x > 0` still sees int | string:
    the predicate may have failed OR the int may have been nonpositive.
    Refinement belongs to the tested binding/storage version. Projecting a
    variant preserves its owner and loan dependencies, not a new owning copy.
    Assignment/replacement, including through an alias or a mutating call,
    invalidates affected refinements; declared assignment types remain intact.
    A read-only call preserves facts. Consumption remains a move even when
    the consumed binding has a refined type. Calls are evaluated once, but a
    test of a temporary does not refine a later, separately evaluated call.
"""

from py_compiler.syntax.primitive.numeric.grammar import ScalarOperation
from py_compiler.hir.thir.src.model import Edge, Terminator


class Logical(ScalarOperation):
    def lower_expression(self, cfg, node, env):
        left = cfg.expr(node.operands[0], env, self.owner)
        from py_compiler.mir.ownership.flow import join_expression
        rhs, skipped = cfg.block(), cfg.block()
        values, flow = dict(cfg.values), cfg.flow.fork()
        direct = Edge(skipped.identity)
        edges = (Edge(rhs.identity), direct) if self.spelling == 'and' else (direct, Edge(rhs.identity))
        cfg.current.terminator = Terminator('conditional', edges, left)
        cfg.current = rhs
        right = cfg.expr(node.operands[1], env, self.owner)
        return join_expression(cfg, [
            (skipped, values, flow, left),
            (cfg.current, dict(cfg.values), cfg.flow.fork(), right),
        ], env, node.token.span)


import unittest


class LogicalOwnershipTests(unittest.TestCase):
    def test_short_circuit_consumption_and_cleanup(self):
        from py_compiler.frontend.src.lib import compile_source
        from py_compiler.syntax.recognition import Syntax
        # A move parameter consumes the caller's binding on the RHS path.
        for operator in ('and', 'or'):
            for mode in ('view', 'move'):
                for use_after in (False, True):
                    with self.subTest(operator=operator, mode=mode, use_after=use_after):
                        source = (f'def consume(value: {mode} string) -> bool:\n    return true\n'
                                  'def work(flag: bool):\n    owner = "owned"\n'
                                  f'    result = flag {operator} consume(owner)\n')
                        if use_after:
                            source += '    observed = owner\n'
                        result = compile_source('short-circuit-ownership.sev', source, Syntax())
                        if use_after and mode == 'move':
                            self.assertTrue(result.diagnostics)
                            self.assertIn('moved', str(result.diagnostics))
                        else:
                            self.assertFalse(result.diagnostics, str(result.diagnostics))
