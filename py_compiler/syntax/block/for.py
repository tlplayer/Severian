"""Synchronous iterator consumers lower producer continuations to ordinary CFG."""
from dataclasses import dataclass, replace
from importlib import import_module
from py_compiler.syntax.generic.block import BlockProvider
from py_compiler.syntax.grammar.expression import expression
from py_compiler.mir.cfg.cfg import Edge, Terminator
from py_compiler.mir.ownership.flow import Binding, FlowState

loop_syntax = import_module('py_compiler.syntax.block.while')


def bindings(*scopes):
    return tuple({b.identity: b for scope in scopes for b in scope.values()}.values())


def edge(cfg, target, carried, span):
    values = tuple(cfg.values[b.identity] for b in carried)
    for source, destination in zip(values, target.parameters):
        cfg.value_sources.setdefault(destination.identity, set()).update(cfg.value_sources.get(source.identity, ()))
        if source.identity in cfg.stack_values:
            cfg.stack_values.add(destination.identity)
    cfg.current.terminator = Terminator('jump', (Edge(target.identity, values),), span=span)


@dataclass
class FrozenCapture:
    value: object
    owner: object

    @property
    def contract(self):
        return self.value.type

    def type(self, cfg, expected=None):
        return self.contract

    def binding(self, cfg):
        return self.owner

    def lower(self, cfg, expected=None):
        cfg.flow.read(self.owner)
        return self.value


class ConsumerJump:
    def __init__(self, consumer, resume, carried, flow, depth):
        self.consumer, self.resume, self.carried = consumer, resume, carried
        self.flow, self.depth = flow, depth
        self.exits = []

    def jump(self, cfg, span, *, exit_loop=False):
        consumer = self.consumer
        kept = consumer.carried if exit_loop else self.carried
        live = {b.identity for b in kept} | consumer.ambient
        depth = consumer.depth if exit_loop else self.depth
        cfg.cleanup_memory((b for scope in cfg.active_scopes[depth:] for b in scope.values()
                            if b.identity not in live), span)
        cfg.flow.end_scope(live)
        baseline = consumer.flow if exit_loop else self.flow
        expected = baseline.fork()
        expected.end_scope(live)
        if cfg.flow != expected:
            raise ValueError('iterator loop cannot change outer ownership on a continuing edge')
        edge(cfg, consumer.following if exit_loop else self.resume, kept, span)
        (consumer.exits if exit_loop else self.exits).append(cfg.flow.fork())


class Consumer:
    def __init__(self, node, cfg, env, carried, following, ambient, element):
        self.node, self.cfg, self.env = node, cfg, env
        self.carried, self.following, self.ambient = carried, following, ambient
        self.element = element
        self.flow = cfg.flow.fork()
        self.depth = len(cfg.active_scopes)
        self.captures, self.handler, self.loops = cfg.grammar_captures, cfg.yield_handler, list(cfg.loops)
        self.exits = []
        self.active = set(cfg.active_grammars)
        self.count = 0

    def emit(self, term, producer_env, span):
        cfg = self.cfg
        value = cfg.expr(term, producer_env, self.element)
        if value.type != self.element:
            raise ValueError('yield value does not satisfy iterator element type')
        carried = tuple(b for b in bindings(self.env, *cfg.active_scopes[self.depth:])
                        if b.identity not in cfg.storage and b.identity not in cfg.flow.moved)
        resume = cfg.block(tuple(b.type for b in carried))
        state, values = cfg.flow.fork(), dict(cfg.values)
        frame = ConsumerJump(self, resume, carried, state, len(cfg.active_scopes))
        name = self.node.header[0].text
        item = Binding(self.node.identity + ':item:' + str(self.count), name, value.type, False, 'copy', span)
        self.count += 1
        body_env = dict(self.env)
        body_env[name] = item
        cfg.values[item.identity] = value
        cfg.body.bindings.append(item)
        previous = cfg.grammar_captures, cfg.yield_handler, cfg.loops, cfg.active_grammars
        cfg.grammar_captures, cfg.yield_handler, cfg.loops = self.captures, self.handler, [*self.loops, frame]
        cfg.active_grammars = set(self.active)
        try:
            def clone(node):
                return replace(node, identity=node.identity + ':yield:' + str(self.count),
                               children=[clone(child) for child in node.children])
            cfg.scope([clone(n) for n in self.node.children], body_env, local_names={name})
            if cfg.current.terminator is None:
                frame.jump(cfg, span)
        finally:
            cfg.grammar_captures, cfg.yield_handler, cfg.loops, cfg.active_grammars = previous
        cfg.current, cfg.values = resume, values
        cfg.flow = FlowState.join(frame.exits) if frame.exits else state
        cfg.values.update({b.identity: v for b, v in zip(carried, resume.parameters)})


class For(BlockProvider):
    spelling = 'for'

    def parse_header(self, items):
        header = super().parse_header(items)
        if len(header) < 3 or header[0].kind != 'IDENTIFIER' or header[1].text != 'in':
            raise ValueError('expected for name in iterable [with name = value]:')
        source_tokens, initializer = loop_syntax.split_loop_initializer(header[2:], self.spelling)
        source = expression(source_tokens)
        if initializer and initializer[0].text == header[0].text:
            raise ValueError('iteration binding and with binding must be distinct')
        return header[0], source, initializer

    def lower(self, node, cfg, env, nodes, index):
        from py_compiler.syntax.block.grammar import BoundGrammar, expression_grammar
        from py_compiler.syntax.generic.place import expression_type
        from py_compiler.syntax.generic.storage import root_binding
        _, source, initializer = node.header
        match = expression_grammar(source, cfg, env)
        if match is not None and match.provider.result.family == 'iterator':
            provider, captures = match.provider, match.captures
            if set(captures) != {'self'}:
                raise ValueError('iterator grammar currently requires only a self capture')
            source = captures['self'].term
        else:
            type_ = expression_type(source, cfg, env)
            candidates = [BoundGrammar(d, cfg, env) for d in cfg.context.sentence_grammars
                          if d[3] is not None and d[3].family == 'iterator'
                          and tuple(cfg.declaration_scope[:len(d[2])]) == d[2]
                          and d[1] == ('iterator', '(', ('self', type_), ')')]
            if len(candidates) != 1:
                raise ValueError('for requires one applicable iterator(self) grammar')
            provider = candidates[0]
        if not cfg.block_scope.allows(provider.node.scope):
            raise ValueError('production scope cannot use a test grammar')
        if provider.node.identity in cfg.active_grammars:
            raise ValueError('recursive iterator expansion requires a frame implementation')
        owner = root_binding(source, cfg, env)
        if owner is None:
            raise ValueError('iterator source must have a live owner binding')
        value = cfg.expr(source, env)
        loop_env, local_names = loop_syntax.initialize_loop(node, initializer, cfg, env)
        loan = Binding(node.identity + ':iterator-view', ':iterator-view', owner.type, True, 'view', node.span)
        cfg.flow.loans[loan.identity] = cfg.flow.resolve(owner.identity)
        loop_env[loan.name] = loan
        cfg.values[loan.identity] = cfg.read_binding(owner, node.span)
        cfg.body.bindings.append(loan)
        carried = tuple(b for b in bindings(loop_env) if b.identity not in cfg.storage and b.identity not in cfg.flow.moved)
        following = cfg.block(tuple(b.type for b in carried))
        ambient = {b.identity for scope in cfg.active_scopes for b in scope.values()} | cfg.live_binding_ids(loop_env)
        consumer = Consumer(node, cfg, loop_env, carried, following, ambient, provider.result.element)
        original_values = dict(cfg.values)
        previous = cfg.grammar_captures, cfg.yield_handler, cfg.loops
        cfg.loops = []
        cfg.grammar_captures = {'self': FrozenCapture(value, owner)}
        cfg.yield_handler = consumer.emit
        cfg.active_grammars.add(provider.node.identity)
        # Hidden names carry consumer SSA through producer loops without name capture.
        producer_env = {':consumer:' + b.identity: b for b in loop_env.values()}
        cfg.grammar_expansions += 1
        suffix = ':iterator:' + str(cfg.grammar_expansions)
        def clone(n):
            return replace(n, identity=n.identity + suffix, children=[clone(c) for c in n.children])
        try:
            cfg.scope([clone(n) for n in provider.node.children], producer_env)
            if cfg.current.terminator is None:
                cfg.flow.end_scope(ambient)
                edge(cfg, following, carried, node.span)
                consumer.exits.append(cfg.flow.fork())
        finally:
            cfg.grammar_captures, cfg.yield_handler, cfg.loops = previous
            cfg.active_grammars.remove(provider.node.identity)
        cfg.current, cfg.values = following, original_values
        cfg.values.update({b.identity: v for b, v in zip(carried, following.parameters)})
        cfg.flow = FlowState.join(consumer.exits) if consumer.exits else consumer.flow
        cfg.flow.loans.pop(loan.identity, None)
        cfg.cleanup_memory((loop_env[n] for n in local_names), node.span)
        cfg.flow.end_scope(cfg.live_binding_ids(env) | {b.identity for scope in cfg.active_scopes for b in scope.values()})
        return 1
