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

    def fork(self):
        return FlowState(set(self.moved), dict(self.loans), dict(self.views),
                         {owner: dict(slots) for owner, slots in self.retained.items()}, self.reporter)

    def read(self, binding):
        self.resolve(binding.identity)

    def resolve(self, identity):
        visited = set()
        while True:
            if identity in self.moved:
                raise ValueError("binding or viewed owner was moved or dropped")
            if identity in visited:
                raise ValueError("cyclic view binding")
            visited.add(identity)
            if identity not in self.views:
                return identity
            identity = self.views[identity]

    def replace(self, binding):
        if binding.constant:
            raise ValueError(f"cannot rebind constant {binding.name!r} declared with :=")
        self.transfer(binding)
        self.moved.discard(binding.identity)

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
        if binding.identity in self.loans.values() or self.borrowers(binding.identity):
            raise ValueError(f"cannot replace, move or drop {binding.name!r} while a borrow is live")

    def move(self, binding):
        self.read(binding)
        self.transfer(binding)
        # Moving a view transfers the view handle, not the storage it observes.
        self.moved.add(binding.identity)

    def drop(self, binding):
        self.move(binding)
        self.loans.pop(binding.identity, None)
        self.views.pop(binding.identity, None)
        self.end_retained(binding.identity)

    def loan(self, binding, owner):
        self.read(owner)
        captures_storage = (getattr(binding.type, 'family', None) == 'pointer' or
                            (getattr(binding.type, 'family', None) in ('array', 'record', 'shared', 'string', 'union', 'dynamic') and binding.type != owner.type))
        if binding.ownership == 'borrow' or (captures_storage and getattr(owner.type, 'binding_ownership', None) != 'copy'):
            self.loans[binding.identity] = self.resolve(owner.identity)
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
        self.loans = {loan: owner for loan, owner in self.loans.items() if loan in live}
        for view, owner in self.views.items():
            if view in live and owner not in live:
                raise ValueError("view escapes the lifetime of its source binding")
        self.views = {view: owner for view, owner in self.views.items() if view in live}
        self.moved.intersection_update(live)

    @staticmethod
    def join(states):
        result = FlowState()
        for state in states:
            result.moved.update(state.moved)
            result.loans.update(state.loans)
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


import unittest


class OwnershipTests(unittest.TestCase):
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


def verify_retained_lifetimes(body):
    """Forward may-analysis: a borrow present on any incoming path pins its source."""
    incoming = {block.identity: {} for block in body.blocks}
    reachable, pending = set(), [0]
    while pending:
        identity = pending.pop()
        if identity not in reachable:
            reachable.add(identity)
            pending.extend(edge.target for edge in body.blocks[identity].terminator.edges)
    def transfer(block, state, check=False):
        state = dict(state)
        for operation in block.operations:
            event = operation.payload
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
            elif event.kind == 'end':
                state = {key: sources for key, sources in state.items() if key[0] != event.owner}
            elif check and event.kind == 'check' and any(event.owner in sources for sources in state.values()):
                raise ValueError(f'cannot replace, move or drop {event.slot!r} while a stored borrow is live on a control-flow path')
        return state
    changed = True
    while changed:
        changed = False
        for block in body.blocks:
            if block.identity not in reachable:
                continue
            outgoing = transfer(block, incoming[block.identity])
            for edge in block.terminator.edges:
                joined = dict(incoming[edge.target])
                for key, sources in outgoing.items():
                    joined[key] = joined.get(key, frozenset()) | sources
                if joined != incoming[edge.target]:
                    incoming[edge.target] = joined
                    changed = True
    for block in body.blocks:
        if block.identity in reachable:
            transfer(block, incoming[block.identity], check=True)
