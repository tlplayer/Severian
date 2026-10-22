"""Views follow binding identities; borrows additionally constrain storage mutation."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Binding:
    identity: str
    name: str
    type: object
    constant: bool
    ownership: str
    origin: object


@dataclass
class FlowState:
    moved: set = field(default_factory=set)
    loans: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    retained: dict = field(default_factory=dict)
    reporter: object = field(default=None, compare=False, repr=False)
    view_sources: dict = field(default_factory=dict)
    shape_loans: set = field(default_factory=set)
    mirror_readers: set = field(default_factory=set)

    def fork(self):
        return FlowState(set(self.moved), dict(self.loans), dict(self.views),
                         {owner: dict(slots) for owner, slots in self.retained.items()}, self.reporter,
                         dict(self.view_sources), set(self.shape_loans), set(self.mirror_readers))

    def read(self, binding):
        self.resolve(binding.identity)
        if self.reporter:
            sources, pending = set(), [binding.identity]
            while pending:
                identity = pending.pop()
                if identity in sources:
                    continue
                sources.add(identity)
                pending.extend(self.view_sources.get(identity, ()))
                if identity in self.views:
                    pending.append(self.views[identity])
            self.reporter('read', binding.identity, binding.name, sources)

    def resolve(self, identity, _seen=()):
        visited = set(_seen)
        while True:
            if identity in self.moved:
                raise ValueError("binding or viewed owner was moved or dropped")
            if identity in visited:
                raise ValueError("cyclic view binding")
            visited.add(identity)
            for source in self.view_sources.get(identity, ()):
                self.resolve(source, visited)
            if identity not in self.views:
                return identity
            identity = self.views[identity]

    def replace(self, binding):
        if binding.constant:
            raise ValueError(f"cannot rebind constant {binding.name!r} declared with :=")
        self.transfer(binding)
        self.moved.discard(binding.identity)
        if self.reporter:
            self.reporter('initialize', binding.identity, binding.name, ())
        self.view_sources.pop(binding.identity, None)

    def retain(self, owner, slot, sources):
        sources = frozenset(self.resolve(source) for source in sources)
        if owner in sources:
            raise ValueError('an owner cannot retain a borrow into itself')
        if self.reporter:
            self.reporter('retain', owner, slot, sources)
        if sources:
            self.retained.setdefault(owner, {})[slot] = sources
        elif owner in self.retained:
            self.retained[owner].pop(slot, None)
            if not self.retained[owner]:
                self.retained.pop(owner)

    def sources(self, owner):
        return set().union(*self.retained.get(owner, {}).values()) if owner in self.retained else set()

    def borrowers(self, owner):
        return {container for container in self.retained if owner in self.sources(container)}

    def transfer(self, binding):
        if self.reporter:
            self.reporter('check', binding.identity, binding.name, ())
        if any(binding.identity in (sources if isinstance(sources, frozenset) else (sources,)) for sources in self.loans.values()) or self.borrowers(binding.identity):
            raise ValueError(f"cannot replace, move or drop {binding.name!r} while a borrow is live")

    def move(self, binding):
        self.read(binding)
        self.transfer(binding)
        # Moving a view transfers the view handle, not the storage it observes.
        self.moved.add(binding.identity)
        if self.reporter:
            self.reporter('move', binding.identity, binding.name, ())

    def drop(self, binding):
        self.move(binding)
        self.end_loan(binding.identity)
        self.views.pop(binding.identity, None)
        self.view_sources.pop(binding.identity, None)
        self.end_retained(binding.identity)

    def begin_loan(self, identity, source, *, shape=False):
        self.loans[identity] = source
        if shape:
            self.shape_loans.add(identity)
        if self.reporter:
            self.reporter('retain', identity, 'shape' if shape else 'loan', (source,))

    def end_loan(self, identity):
        if identity in self.loans:
            self.loans.pop(identity)
            if self.reporter:
                self.reporter('end-loan', identity, 'shape' if identity in self.shape_loans else 'loan', ())
            self.shape_loans.discard(identity)

    def loan(self, binding, owner):
        self.read(owner)
        if binding.ownership == 'borrow':
            identity = self.resolve(owner.identity)
            conflicts = [loan for loan, sources in self.loans.items()
                         if loan != owner.identity and identity in
                         (sources if isinstance(sources, frozenset) else (sources,))]
            if conflicts or self.borrowers(identity):
                raise ValueError('cannot acquire a borrow while another borrow is live')
        captures_storage = (getattr(binding.type, 'family', None) == 'pointer' or
                            (getattr(binding.type, 'family', None) in ('array', 'tensor', 'tuple', 'record', 'shared', 'string', 'union', 'dynamic') and binding.type != owner.type))
        if binding.ownership == 'borrow' or (captures_storage and getattr(owner.type, 'binding_ownership', None) != 'copy'):
            self.begin_loan(binding.identity, self.resolve(owner.identity))
        self.views[binding.identity] = owner.identity
        self.resolve(binding.identity)

    def end_retained(self, owner):
        if self.reporter:
            self.reporter('end', owner, '', ())
        self.retained.pop(owner, None)

    def copy_retained(self, destination, source):
        if self.reporter:
            self.reporter('copy', destination, source, ())
        slots = dict(self.retained.get(source, {}))
        if slots:
            self.retained[destination] = slots
        else:
            self.retained.pop(destination, None)

    def end_scope(self, live):
        for owner in set(self.retained) - set(live):
            self.end_retained(owner)
        self.retained = {owner: slots for owner, slots in self.retained.items() if owner in live}
        for owner in self.retained:
            if any(source not in live or source in self.moved for source in self.sources(owner)):
                raise ValueError('stored borrow escapes the lifetime of its source binding')
        for loan in set(self.loans) - set(live):
            self.end_loan(loan)
        for view, owner in self.views.items():
            if view in live and owner not in live:
                raise ValueError("view escapes the lifetime of its source binding")
        self.views = {view: owner for view, owner in self.views.items() if view in live}
        for view, sources in self.view_sources.items():
            if view in live and not sources <= live:
                raise ValueError('view escapes the lifetime of a possible source binding')
        self.view_sources = {view: sources for view, sources in self.view_sources.items() if view in live}
        self.moved.intersection_update(live)
        self.mirror_readers.intersection_update(live)

    @staticmethod
    def join(states):
        result = FlowState()
        for state in states:
            result.mirror_readers.update(state.mirror_readers)
            result.shape_loans.update(state.shape_loans)
            for view, sources in state.view_sources.items():
                result.view_sources[view] = result.view_sources.get(view, frozenset()) | sources
            result.moved.update(state.moved)
            for loan, sources in state.loans.items():
                sources = sources if isinstance(sources, frozenset) else frozenset((sources,))
                previous = result.loans.get(loan, frozenset())
                previous = previous if isinstance(previous, frozenset) else frozenset((previous,))
                combined = previous | sources
                result.loans[loan] = next(iter(combined)) if len(combined) == 1 else combined
            for owner, slots in state.retained.items():
                for slot, sources in slots.items():
                    previous = result.retained.setdefault(owner, {}).get(slot, frozenset())
                    result.retained[owner][slot] = previous | sources
            for view, owner in state.views.items():
                if any(other.views.get(view) != owner for other in states):
                    raise ValueError("branches retarget a view differently; explicit copy required at join")
                result.views[view] = owner
        return result


def reconcile_owners(cfg, exits, bindings, span):
    """At a join, consume surviving owners on paths where they remain live."""
    partial = [binding for binding in bindings
               if binding.ownership in ('own', 'move', 'copy', 'mirror')
               and callable(getattr(binding.type, 'release', None))
               and len({binding.identity in flow.moved for _, _, flow in exits}) > 1]
    previous = cfg.current, cfg.values, cfg.flow
    try:
        for index, (block, values, flow) in enumerate(exits):
            cfg.current, cfg.values, cfg.flow = block, values, flow
            terminator = block.terminator
            block.terminator = None
            cfg.cleanup_memory((b for b in partial if b.identity not in flow.moved), span)
            cfg.current.terminator = terminator
            exits[index] = (cfg.current, values, flow)
    finally:
        cfg.current, cfg.values, cfg.flow = previous


def join_expression(cfg, exits, env, span):
    """Join expression exits with the same owner cleanup as statement branches."""
    bindings = tuple(b for b in env.values() if b.identity not in cfg.storage)
    ownership_exits = [(block, values, flow) for block, values, flow, _ in exits]
    reconcile_owners(cfg, ownership_exits, bindings, span)
    flow = FlowState.join([state for _, _, state in ownership_exits])
    carried = tuple(b for b in bindings if b.identity not in flow.moved)
    join = cfg.block((exits[0][3].type, *(b.type for b in carried)))
    from py_compiler.hir.thir.src.model import Edge, Terminator
    for (block, values, _), (_, _, _, result) in zip(ownership_exits, exits):
        block.terminator = Terminator('jump', (Edge(join.identity,
            (result, *(values[b.identity] for b in carried))),))
    cfg.current, cfg.flow = join, flow
    cfg.values = dict(exits[0][1])
    for binding, value in zip(carried, join.parameters[1:]):
        cfg.values[binding.identity] = value
        sources = [values[binding.identity] for _, values, _ in ownership_exits]
        for facts in (cfg.value_sources, cfg.value_dependencies, cfg.value_origins):
            facts[value.identity] = set().union(*(facts.get(source.identity, set()) for source in sources))
        if any(source.identity in cfg.stack_values for source in sources):
            cfg.stack_values.add(value.identity)
    return join.parameters[0]


import unittest


class OwnershipTests(unittest.TestCase):
    def test_exclusive_borrow_cannot_overlap_an_existing_loan(self):
        owner = Binding('owner', 'values', None, False, 'own', None)
        first = Binding('first', 'first', None, False, 'borrow', None)
        second = Binding('second', 'second', None, False, 'borrow', None)
        state = FlowState()
        state.loan(first, owner)
        with self.assertRaisesRegex(ValueError, 'another borrow is live'):
            state.loan(second, owner)
        state.end_loan(first.identity)
        state.loan(second, owner)

    def test_constant_binding_and_loan_are_distinct(self):
        owner = Binding("owner", "text", None, False, "own", None)
        alias = Binding("alias", "alias", None, True, "borrow", None)
        state = FlowState()
        state.loan(alias, owner)
        with self.assertRaisesRegex(ValueError, "live"):
            state.replace(owner)
        with self.assertRaisesRegex(ValueError, "constant"):
            state.replace(alias)
        state.end_scope({"owner"})
        state.replace(owner)

    def test_view_follows_replacement_without_owning_or_borrowing(self):
        owner = Binding("owner", "text", None, False, "own", None)
        alias = Binding("alias", "alias", None, True, "view", None)
        state = FlowState()
        state.loan(alias, owner)
        state.replace(owner)
        self.assertEqual(state.resolve(alias.identity), owner.identity)
        state.move(owner)
        with self.assertRaisesRegex(ValueError, "moved"):
            state.read(alias)

    def test_move_on_one_path_is_possible_move_at_join(self):
        binding = Binding("x", "x", None, False, "own", None)
        left, right = FlowState(), FlowState()
        left.move(binding)
        with self.assertRaisesRegex(ValueError, "moved"):
            FlowState.join([left, right]).read(binding)


@dataclass(frozen=True)
class LifetimeEvent:
    kind: str
    owner: str
    slot: object
    sources: frozenset

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (), None, (), (), self)

    def render_operation(self, operation):
        return []


@dataclass(frozen=True)
class OwnershipRestore:
    """Proven producer facts at one suspended iterator continuation."""
    moved: frozenset
    initialized: frozenset
    retained: tuple

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self, (), None, (), (), self)

    def render_operation(self, operation):
        return []


def ownership_failure(body, operation, message):
    if body.source_snapshot is not None and operation.span is not None:
        from py_compiler.frontend.source.source import Diagnostic
        raise Diagnostic('MIR', message, body.source_snapshot, operation.span)
    raise ValueError(message)


def verify_retained_lifetimes(body):
    """Forward may-analysis: a borrow present on any incoming path pins its source."""
    from py_compiler.mir.cfg.cfg import forward_dataflow
    def transfer(block, state, check=False):
        state = dict(state)
        for operation in block.operations:
            event = operation.payload
            if isinstance(event, OwnershipRestore):
                state = dict(event.retained)
                continue
            if not isinstance(event, LifetimeEvent):
                continue
            key = event.owner, event.slot
            if event.kind == 'retain':
                sources = event.sources
                if (event.slot == '*' or isinstance(event.slot, tuple) and event.slot[0] == 'field') and sources:
                    sources = state.get(key, frozenset()) | sources
                if sources:
                    state[key] = sources
                else:
                    state.pop(key, None)
            elif event.kind == 'copy':
                copied = {(event.owner, slot): sources for (owner, slot), sources in state.items() if owner == event.slot}
                state = {key: sources for key, sources in state.items() if key[0] != event.owner}
                state.update(copied)
            elif event.kind == 'end-loan':
                state.pop(key, None)
            elif event.kind == 'end':
                state = {key: sources for key, sources in state.items() if key[0] != event.owner}
            elif check and event.kind in ('check', 'element-check') and any(
                    event.owner in sources for key, sources in state.items()
                    if not (key[1] == 'loan' and key[0] in event.sources)
                    and not (key[1] == 'shape' and event.kind == 'element-check')):
                ownership_failure(body, operation, f'cannot replace, move or drop {event.slot!r} while a stored borrow is live on a control-flow path')
        return state
    def merge(states):
        result = {}
        for state in states:
            for key, sources in state.items():
                result[key] = result.get(key, frozenset()) | sources
        return result
    incoming, outgoing = forward_dataflow(body.blocks, {}, merge, transfer)
    for identity, state in incoming.items():
        transfer(body.blocks[identity], state, check=True)
    _, bindings = verify_binding_lifetimes(body)
    return {identity: OwnershipRestore(frozenset(bindings[identity][0]),
                                      frozenset(bindings[identity][1]), tuple(state.items()))
            for identity, state in outgoing.items()}


def verify_binding_lifetimes(body):
    """Moves are may-facts; initialization is a must-fact, including backedges."""
    from py_compiler.mir.cfg.cfg import forward_dataflow
    declared = {op.payload.owner for block in body.blocks for op in block.operations
                if isinstance(op.payload, LifetimeEvent) and op.payload.kind == 'declare'}

    def merge(states):
        return (set().union(*(state[0] for state in states)),
                set.intersection(*(state[1] for state in states)))

    def transfer(block, state, check=False):
        moved, initialized = map(set, state)
        for operation in block.operations:
            event = operation.payload
            if isinstance(event, OwnershipRestore):
                moved, initialized = set(event.moved), set(event.initialized)
                continue
            if not isinstance(event, LifetimeEvent):
                continue
            if event.kind in ('initialize', 'declare'):
                moved.discard(event.owner)
                initialized.add(event.owner)
            elif event.kind == 'move':
                moved.add(event.owner)
                initialized.discard(event.owner)
            elif check and event.kind == 'read':
                for source in {event.owner, *event.sources}:
                    if source in moved:
                        ownership_failure(body, operation, f'binding or viewed owner {event.slot!r} was moved on a control-flow path')
                    if source in declared and source not in initialized:
                        ownership_failure(body, operation, f'binding {event.slot!r} is not initialized on every incoming control-flow path')
        return moved, initialized

    incoming, outgoing = forward_dataflow(body.blocks, (set(), set()), merge, transfer)
    for identity, state in incoming.items():
        transfer(body.blocks[identity], state, check=True)
    return incoming, outgoing


class OwnershipDataflowTests(unittest.TestCase):
    def body(self, specifications):
        from py_compiler.mir.cfg.cfg import Body, ExecutionBlock, Operation, Edge, Terminator
        blocks = []
        for identity, (events, targets) in enumerate(specifications):
            operations = [Operation('lifetime', None, (), LifetimeEvent(*event), None) for event in events]
            blocks.append(ExecutionBlock(identity, (), operations,
                Terminator('jump' if targets else 'finish', tuple(Edge(t) for t in targets))))
        return Body('flow', 'flow', 'flow.sev', None, blocks)

    def event(self, kind, owner='owner', slot='', sources=()):
        return kind, owner, slot, frozenset(sources)

    def test_short_circuit_possible_move_rejects_read(self):
        body = self.body([
            ([self.event('declare')], (1, 2)),
            ([self.event('move')], (2,)),
            ([self.event('read')], ()),
        ])
        with self.assertRaisesRegex(ValueError, 'moved'):
            verify_binding_lifetimes(body)

    def test_initialization_requires_every_reachable_predecessor(self):
        body = self.body([
            ([], (1, 2)), ([self.event('declare')], (2,)),
            ([self.event('read')], ()),
        ])
        with self.assertRaisesRegex(ValueError, 'not initialized'):
            verify_binding_lifetimes(body)
        body.blocks[0].terminator = type(body.blocks[0].terminator)('jump', (type(body.blocks[0].terminator.edges[0])(1),))
        verify_binding_lifetimes(body)

    def test_replacement_on_every_continue_and_consumption_on_break(self):
        for replacement, exits, rejected in ((True, False, False), (False, False, True), (False, True, False)):
            with self.subTest(replacement=replacement, exits=exits):
                events = [self.event('read'), self.event('move')]
                if replacement:
                    events.append(self.event('initialize'))
                body = self.body([
                    ([self.event('declare')], (1,)),
                    ([], (2, 3)), (events, (3 if exits else 1,)), ([], ()),
                ])
                if rejected:
                    with self.assertRaisesRegex(ValueError, 'moved'):
                        verify_binding_lifetimes(body)
                else:
                    verify_binding_lifetimes(body)

    def test_one_unreplaced_continue_rejects_next_iteration(self):
        body = self.body([
            ([self.event('declare')], (1,)),
            ([self.event('read'), self.event('move')], (2, 3)),
            ([self.event('initialize')], (1,)), ([], (1,)),
        ])
        with self.assertRaisesRegex(ValueError, 'moved'):
            verify_binding_lifetimes(body)

    def test_borrow_sources_survive_join_and_backedge(self):
        body = self.body([
            ([], (1,)), ([self.event('check', 'source', 'source')], (2, 3)),
            ([self.event('retain', 'holder', 'slot', ('source',))], (1,)), ([], ()),
        ])
        with self.assertRaisesRegex(ValueError, 'stored borrow'):
            verify_retained_lifetimes(body)
        body.blocks[2].operations.append(type(body.blocks[2].operations[0])(
            'lifetime', None, (), LifetimeEvent('end', 'holder', '', frozenset()), None))
        verify_retained_lifetimes(body)

    def test_loan_join_keeps_both_sources(self):
        joined = FlowState.join([FlowState(loans={'loan': 'a'}), FlowState(loans={'loan': 'b'})])
        for source in ('a', 'b'):
            with self.assertRaisesRegex(ValueError, 'borrow is live'):
                joined.transfer(Binding(source, source, None, False, 'own', None))

    def test_resume_restores_moves_initialization_and_borrows(self):
        from py_compiler.mir.cfg.cfg import Operation
        producer = self.body([
            ([self.event('declare'), self.event('move'),
              self.event('declare', 'position'),
              self.event('retain', 'holder', 'slot', ('source',))], (1,)),
            ([], ()),
        ])
        snapshot = verify_retained_lifetimes(producer)[0]
        for event, diagnostic in ((self.event('read'), 'moved'),
                                  (self.event('check', 'source', 'source'), 'stored borrow'),
                                  (self.event('read', 'position'), None)):
            with self.subTest(event=event):
                resumed = self.body([([], (1,)), ([event], ())])
                resumed.blocks[0].operations.append(Operation('ownership-restore', None, (), snapshot, None))
                if diagnostic:
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        verify_retained_lifetimes(resumed)
                else:
                    verify_retained_lifetimes(resumed)

    def test_initialization_diagnostic_names_binding_and_read_span(self):
        from dataclasses import replace
        from py_compiler.frontend.source.source import SourceFile, Diagnostic
        body = self.body([([], (1, 2)), ([self.event('declare', 'value')], (2,)),
                          ([self.event('read', 'value', 'value')], ())])
        source = SourceFile('missing-initialization.sev', 'class Example:\n    value\n')
        span = source.span(source.text.index('value'), len(source.text) - 1)
        body.source_snapshot = source
        body.blocks[2].operations[0] = replace(body.blocks[2].operations[0], span=span)
        with self.assertRaises(Diagnostic) as raised:
            verify_binding_lifetimes(body)
        self.assertEqual(raised.exception.span, span)
        self.assertIn("binding 'value' is not initialized", str(raised.exception))
