"""Bounded channel selection checks readiness and consumes one selected value."""
from py_compiler.syntax.generic.block import BlockProvider


class Select(BlockProvider):
    spelling = 'select'

    def parse_header(self, items):
        header = super().parse_header(items)
        if len(header) < 4 or [t.text for t in header[:3]] != ['with', 'limit', '=']:
            raise ValueError('select requires with limit=count')
        return header[3:]

    def lower(self, node, cfg, env, nodes, index):
        from py_compiler.syntax.grammar.expression import expression
        from py_compiler.hir.thir.src.model import Edge, Terminator
        from py_compiler.syntax.function.calls import literal_value
        from py_compiler.syntax.generic.owned import select
        from py_compiler.mir.ownership.flow import Binding, FlowState
        limit = cfg.expr(expression(node.header), env, cfg.context.type('int'))
        cases = []
        for child in node.children:
            if getattr(child.provider, 'spelling', '') != 'case':
                raise ValueError('select requires case blocks')
            if len(child.header) == 1 and child.header[0].text == 'error':
                # These channels stay open until their owning scope ends.
                # There is no closed-input edge while that owner remains live.
                continue
            if len(child.header) < 3 or child.header[1].text != 'from':
                raise ValueError('select case requires binding from channel')
            channel = cfg.expr(expression(child.header[2:]), env)
            if channel.type.family != 'channel':
                raise ValueError('select inputs must be channels')
            cases.append((child, channel))
        if not cases:
            raise ValueError('select requires a channel case')
        outer = tuple(b for b in env.values() if b.identity not in cfg.storage)
        initial = dict(cfg.values)
        initial_flow = cfg.flow.fork()
        loop = cfg.block((*[b.type for b in outer], cfg.context.type('int')))
        zero = literal_value(0, cfg.context.type('int'), cfg, node.span)
        cfg.current.terminator = Terminator('jump', (Edge(loop.identity, (*[cfg.values[b.identity] for b in outer], zero)),))
        cfg.current = loop
        cfg.values.update({b.identity: v for b, v in zip(outer, loop.parameters)})
        counter = loop.parameters[-1]
        pending = select(counter.type, 'F.operator', '<').expand(cfg, counter, limit, node.span)
        scan, done = cfg.block(), cfg.block()
        cfg.current.terminator = Terminator('conditional', (Edge(scan.identity), Edge(done.identity)), pending)
        cfg.current = scan
        loop_values, flows = dict(cfg.values), []
        for child, channel in cases:
            ready = channel.type.ready(channel, cfg, child.span)
            chosen, following = cfg.block(), cfg.block()
            cfg.current.terminator = Terminator('conditional', (Edge(chosen.identity), Edge(following.identity)), ready)
            cfg.current = chosen
            value = channel.type.receive(channel, cfg, child.span)
            name = child.header[0].text
            binding = Binding(child.identity + ':value', name, value.type, False, 'own', child.span)
            local = {**env, name: binding}
            cfg.body.bindings.append(binding)
            cfg.values[binding.identity] = value
            cfg.record_lifetime('declare', binding.identity, binding.name)
            cfg.scope(child.children, local, local_names={name})
            if cfg.current.terminator is None:
                one = literal_value(1, counter.type, cfg, node.span)
                next_count = select(counter.type, 'F.operator', '+').expand(cfg, counter, one, node.span)
                cfg.current.terminator = Terminator('jump', (Edge(loop.identity, (*[cfg.values[b.identity] for b in outer], next_count)),))
                flows.append(cfg.flow.fork())
            cfg.current, cfg.values, cfg.flow = following, dict(loop_values), initial_flow.fork()
        cfg.effect('select-yield', (), Yield(), node.span)
        cfg.current.terminator = Terminator('jump', (Edge(loop.identity, (*[cfg.values[b.identity] for b in outer], counter)),))
        cfg.current = done
        cfg.values = loop_values
        cfg.flow = FlowState.join([initial_flow, *flows])
        return 1


class Yield:
    external_symbol_contracts = (('sched_yield', (), 'i32'),)

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (), None, (), ('IO',), self)

    def render_operation(self, operation):
        from py_compiler.mlir.src.cfg import temporary_suffix
        return ['%yield_' + temporary_suffix(operation) + ' = func.call @sched_yield() : () -> i32']
