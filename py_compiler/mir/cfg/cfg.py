"""Typed SSA execution graph. Source containment is retained separately."""
from dataclasses import dataclass, field


def forward_dataflow(blocks, boundary, merge, transfer):
    """Solve reachable forward equations with analysis-specific merge/transfer.

    Missing output means unreachable so far, not an empty predecessor fact.
    Recompute from current outputs so kills survive loop iterations.
    """
    from collections import deque
    predecessors = {block.identity: set() for block in blocks}
    for block in blocks:
        for edge in block.terminator.edges:
            predecessors[edge.target].add(block.identity)
    incoming, outgoing = {}, {}
    pending, queued = deque([0]), {0}
    while pending:
        identity = pending.popleft()
        queued.remove(identity)
        states = [outgoing[p] for p in sorted(predecessors[identity]) if p in outgoing]
        if identity == 0:
            states.insert(0, boundary)
        if not states:
            continue
        incoming[identity] = merge(states)
        result = transfer(blocks[identity], incoming[identity])
        if identity in outgoing and result == outgoing[identity]:
            continue
        outgoing[identity] = result
        for edge in blocks[identity].terminator.edges:
            if edge.target not in queued:
                pending.append(edge.target)
                queued.add(edge.target)
    return incoming, outgoing


@dataclass(frozen=True)
class Value:
    identity: int
    type: object


@dataclass(frozen=True)
class Operation:
    kind: str
    result: Value
    operands: tuple
    payload: object
    span: object
    atom: object = None
    lowering_position: tuple[int, int] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Edge:
    target: int
    arguments: tuple = ()


@dataclass(frozen=True)
class Terminator:
    kind: str
    edges: tuple = ()
    value: Value | None = None
    message: str = ""
    span: object = None


@dataclass
class ExecutionBlock:
    identity: int
    parameters: tuple = ()
    operations: list = field(default_factory=list)
    terminator: Terminator | None = None


@dataclass
class Body:
    identity: str
    name: str
    source: str
    result_type: object
    blocks: list = field(default_factory=list)
    bindings: list = field(default_factory=list)
    declaration: str = ""
    storage: list = field(default_factory=list)
    complexity: dict = field(default_factory=dict)
    test: dict | None = None
    result_sources: tuple = ()
    retained_result_sources: tuple = ()
    retained_parameter_sources: tuple = ()
    result_ownership: str = 'own'
    dispatch_guards: tuple = ()
    parameter_defaults: dict = field(default_factory=dict)
    result_layout: str = ""
    source_snapshot: object = None
    source_span: object = None
    memory_regions: tuple = ()
    call_contracts: tuple | None = None

    def verify(self):
        if not self.blocks:
            raise ValueError("CFG has no entry")
        definitions, predecessors = {}, {b.identity: set() for b in self.blocks}
        for index, block in enumerate(self.blocks):
            if block.identity != index or block.terminator is None:
                raise ValueError("CFG identity or terminator missing")
            for value in (*block.parameters, *(o.result for o in block.operations if o.result is not None)):
                if value.identity in definitions:
                    raise ValueError("duplicate SSA definition")
                definitions[value.identity] = (index, value.type)
            term = block.terminator
            if term.kind not in ("jump", "conditional", "finish", "panic"):
                raise ValueError("unknown CFG terminator")
            expected_edges = {"jump": 1, "conditional": 2, "finish": 0, "panic": 0}[term.kind]
            if len(term.edges) != expected_edges:
                raise ValueError("CFG terminator edge count mismatch")
            if term.kind == "conditional" and (term.value is None or term.value.type.name != "bool"):
                raise ValueError("conditional requires bool")
            if term.kind == "finish":
                actual = term.value.type if term.value else None
                expected = self.result_type if self.result_type.mlir else None
                if actual != expected:
                    raise ValueError("return type mismatch")
            for edge in term.edges:
                if not 0 <= edge.target < len(self.blocks):
                    raise ValueError("CFG target outside body")
                parameters = self.blocks[edge.target].parameters
                if tuple(v.type for v in edge.arguments) != tuple(v.type for v in parameters):
                    raise ValueError("CFG edge argument contract mismatch")
                predecessors[edge.target].add(index)
        reachable, pending = set(), [0]
        while pending:
            node = pending.pop()
            if node not in reachable:
                reachable.add(node)
                pending.extend(e.target for e in self.blocks[node].terminator.edges)
        dominators = {b: ({0} if b == 0 else set(reachable)) for b in reachable}
        changed = True
        while changed:
            changed = False
            for b in reachable - {0}:
                incoming = predecessors[b] & reachable
                new = {b} | set.intersection(*(dominators[p] for p in incoming))
                if new != dominators[b]:
                    dominators[b], changed = new, True
        def transfer_moves(block, incoming):
            outgoing = incoming - {v.identity for v in block.parameters}
            for operation in block.operations:
                if operation.result:
                    outgoing.discard(operation.result.identity)
                if operation.atom:
                    outgoing.update(v.identity for v, mode in zip(operation.operands, operation.atom.ownership) if mode == 'move')
            return outgoing
        moved_in, _ = forward_dataflow(self.blocks, set(),
                                       lambda states: set().union(*states), transfer_moves)
        for b in reachable:
            moved = moved_in[b] - {v.identity for v in self.blocks[b].parameters}
            available = {v.identity for v in self.blocks[b].parameters}
            def check(value):
                if value.identity in moved:
                    raise ValueError("atom uses a value moved on a control-flow path")
                owner = definitions.get(value.identity)
                if owner is None or owner[1] != value.type or owner[0] not in dominators[b]:
                    raise ValueError("SSA use lacks a dominating typed definition")
                if owner[0] == b and value.identity not in available:
                    raise ValueError("SSA use before definition")
            for operation in self.blocks[b].operations:
                if operation.atom is not None:
                    operation.atom.verify(operation)
                for operand in operation.operands:
                    check(operand)
                if operation.atom:
                    for operand, mode in zip(operation.operands, operation.atom.ownership):
                        if mode == 'move':
                            if operand.identity in moved:
                                raise ValueError("atom moves the same value twice")
                            moved.add(operand.identity)
                if operation.result is not None:
                    moved.discard(operation.result.identity)
                    available.add(operation.result.identity)
            term = self.blocks[b].terminator
            if term.value:
                check(term.value)
            for edge in term.edges:
                for value in edge.arguments:
                    check(value)


import unittest
from py_compiler.syntax.primitive.catalog import primitives


class CfgTests(unittest.TestCase):
    def test_rejects_nondominating_value(self):
        t = primitives()
        flag, missing = Value(0, t["bool"]), Value(1, t["i64"])
        body = Body("test", "test", "x", t["i64"], [
            ExecutionBlock(0, (flag,), [], Terminator("conditional", (Edge(1), Edge(2)), flag)),
            ExecutionBlock(1, (), [Operation("constant", missing, (), 1, None)], Terminator("jump", (Edge(2),))),
            ExecutionBlock(2, (), [], Terminator("finish", value=missing))])
        with self.assertRaisesRegex(ValueError, "dominating"):
            body.verify()
