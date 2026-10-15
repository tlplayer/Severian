"""Expected runtime failure boundary, retaining its lexical execution scope."""
from py_compiler.syntax.generic.block import BlockProvider


class Throws(BlockProvider):
    spelling = 'throws'
    required_scope = 'test'

    def lower(self, node, cfg, env, nodes, index):
        if getattr(cfg, 'expected_failure_node', None) != node.identity:
            raise ValueError('runtime throws must be the sole test boundary, optionally enclosed by unsafe blocks')
        cfg.scope(node.children, dict(env))
        cfg.flow.end_scope(cfg.live_binding_ids(env))
        return 1
