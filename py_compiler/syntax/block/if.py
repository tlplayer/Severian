from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.mir.cfg.cfg import Edge, Terminator
from py_compiler.mir.ownership.flow import FlowState


class Conditional(BlockProvider):
    spelling = "if"
    has_condition = True
    attachment = "condition-chain"

    def parse_header(self, items):
        header = super().parse_header(items)
        expression(header)
        return header

    def attach(self, previous, parent):
        super().attach(previous, parent)
        if self.continuation and not previous.provider.has_condition:
            raise ValueError(f"{self.spelling} cannot follow a terminal branch")

    def lower(self, node, cfg, env, nodes, index):
        chain = [node]
        while index + len(chain) < len(nodes):
            following = nodes[index + len(chain)]
            if not getattr(following.provider, "continuation", False):
                break
            chain.append(following)
        self.lower_chain(cfg, chain, env)
        return len(chain)

    def lower_chain(self, cfg, chain, env):
        outer = tuple(b for b in env.values() if b.identity not in cfg.storage)
        initial_values, initial_flow = dict(cfg.values), cfg.flow.fork()
        exits = []
        for node in chain:
            cfg.values, cfg.flow = dict(initial_values), initial_flow.fork()
            if node.provider.has_condition:
                condition = cfg.expr(expression(node.header), env, cfg.syntax.types["bool"])
                if condition.type.name != "bool":
                    raise ValueError("if/elif condition must satisfy bool")
                body, following = cfg.block(), cfg.block()
                cfg.current.terminator = Terminator("conditional", (Edge(body.identity), Edge(following.identity)), condition)
                cfg.current = body
            cfg.scope(node.children, dict(env))
            cfg.flow.end_scope(cfg.live_binding_ids(env))
            if cfg.current.terminator is None:
                exits.append((cfg.current, dict(cfg.values), cfg.flow.fork()))
            if node.provider.has_condition:
                cfg.current = following
        if chain[-1].provider.has_condition:
            exits.append((cfg.current, initial_values, initial_flow))
        if not exits:
            return
        for binding in outer:
            if callable(getattr(binding.type, 'release', None)):
                moved = {binding.identity in flow.moved for _, _, flow in exits}
                if len(moved) != 1:
                    raise ValueError('memory ownership must agree across continuing branches')
        outer = tuple(b for b in outer if not (callable(getattr(b.type, 'release', None)) and b.identity in exits[0][2].moved))
        join = cfg.block(tuple(b.type for b in outer))
        for block, values, _ in exits:
            block.terminator = Terminator("jump", (Edge(join.identity, tuple(values[b.identity] for b in outer)),))
        for binding, value in zip(outer, join.parameters):
            cfg.value_sources[value.identity] = set().union(*(cfg.value_sources.get(values[binding.identity].identity, set()) for _, values, _ in exits))
            if any(values[binding.identity].identity in cfg.stack_values for _, values, _ in exits):
                cfg.stack_values.add(value.identity)
        cfg.current = join
        cfg.values = dict(initial_values)
        cfg.values.update({b.identity: value for b, value in zip(outer, join.parameters)})
        cfg.flow = FlowState.join([flow for _, _, flow in exits])
