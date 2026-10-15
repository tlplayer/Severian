"""Typed semantic expressions and continuations; no executable MIR or Atoms.

Checking uses mutable drafts. Published bodies contain immutable regions and
contracts; executable operations and memory placement are constructed by MIR.
"""
from dataclasses import dataclass, field, fields
from types import MappingProxyType


def freeze_data(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze_data(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_data(item) for item in value)
    return value


def thaw_data(value):
    if isinstance(value, MappingProxyType):
        return {key: thaw_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(thaw_data(item) for item in value)
    return value


@dataclass(frozen=True)
class Value:
    identity: int
    type: object


@dataclass(frozen=True)
class OperationContract:
    inputs: tuple
    result: object
    ownership: tuple
    effects: tuple

    @classmethod
    def check(cls, provider, expression):
        # Providers expose their operation contract together with their backend
        # implementation. Retain only semantic facts in THIR.
        selected = provider.atom(expression)
        selected.verify(expression)
        return cls(selected.inputs, selected.result, selected.ownership, selected.effects)

    def verify(self, expression):
        if tuple(value.type for value in expression.operands) != self.inputs:
            raise ValueError('typed expression input contract mismatch')
        if (expression.result.type if expression.result else None) != self.result:
            raise ValueError('typed expression result contract mismatch')


@dataclass(frozen=True)
class Expression:
    kind: str
    result: Value | None
    operands: tuple
    payload: object
    span: object
    contract: OperationContract | None = None


@dataclass(frozen=True)
class SourceExpression:
    identity: int
    kind: str
    type: object
    span: object
    symbol: str | None
    children: tuple
    operations: tuple
    value: Value | None


@dataclass(frozen=True)
class Edge:
    target: int
    arguments: tuple = ()


@dataclass(frozen=True)
class Terminator:
    kind: str
    edges: tuple = ()
    value: Value | None = None
    message: str = ''
    span: object = None


@dataclass
class Region:
    identity: int
    parameters: tuple = ()
    operations: list = field(default_factory=list)
    terminator: Terminator | None = None


@dataclass(frozen=True)
class TypedRegion:
    identity: int
    parameters: tuple
    operations: tuple
    terminator: Terminator


@dataclass(frozen=True)
class TypedBody:
    identity: str
    name: str
    source: str
    result_type: object
    blocks: tuple = ()
    bindings: tuple = ()
    declaration: str = ""
    storage: tuple = ()
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
    expressions: tuple = ()

    def verify(self):
        for expression in self.expressions:
            if expression.type is None or getattr(expression.type, 'family', 'unresolved') == 'unresolved':
                raise ValueError('source expression requires a resolved type')
        if not self.blocks:
            raise ValueError("CFG has no entry")
        definitions = {}
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
        # THIR verifies types and references. Dominance, path-sensitive move
        # checking and memory placement belong to executable MIR validation.
        def check(value):
            if value is not None and (value.identity not in definitions or definitions[value.identity][1] != value.type):
                raise ValueError('typed expression references an unresolved value')
        for block in self.blocks:
            for operation in block.operations:
                if operation.contract is None:
                    raise ValueError('typed expression requires a finalized operation contract')
                operation.contract.verify(operation)
                for operand in operation.operands:
                    check(operand)
            check(block.terminator.value)
            for edge in block.terminator.edges:
                for argument in edge.arguments:
                    check(argument)


class DraftBody:
    def __init__(self, identity, name, source, result_type):
        initial = TypedBody(identity, name, source, result_type)
        for description in fields(TypedBody):
            setattr(self, description.name, getattr(initial, description.name))
        self.blocks, self.bindings, self.storage, self.expressions = [], [], [], []

    verify = TypedBody.verify

    def freeze(self):
        self.verify()
        data = {description.name: freeze_data(getattr(self, description.name))
                for description in fields(TypedBody)}
        data['blocks'] = tuple(TypedRegion(block.identity, tuple(block.parameters),
                                          tuple(block.operations), block.terminator)
                               for block in self.blocks)
        return TypedBody(**data)
