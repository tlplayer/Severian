"""Typed SSA execution graph. Source containment is retained separately."""
from dataclasses import dataclass, field


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
        for b in reachable:
            available = {v.identity for v in self.blocks[b].parameters}
            def check(value):
                owner = definitions.get(value.identity)
                if owner is None or owner[1] != value.type or owner[0] not in dominators[b]:
                    raise ValueError("SSA use lacks a dominating typed definition")
                if owner[0] == b and value.identity not in available:
                    raise ValueError("SSA use before definition")
            for operation in self.blocks[b].operations:
                for operand in operation.operands:
                    check(operand)
                if operation.result is not None:
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
