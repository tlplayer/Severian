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

    def fork(self):
        return FlowState(set(self.moved), dict(self.loans), dict(self.views))

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

    def transfer(self, binding):
        if binding.identity in self.loans.values():
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

    def loan(self, binding, owner):
        self.read(owner)
        if binding.ownership == "borrow":
            self.loans[binding.identity] = self.resolve(owner.identity)
        self.views[binding.identity] = owner.identity
        self.resolve(binding.identity)

    def end_scope(self, live):
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
            for view, owner in state.views.items():
                if any(other.views.get(view) != owner for other in states):
                    raise ValueError("branches retarget a view differently; explicit copy required at join")
                result.views[view] = owner
        return result


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
