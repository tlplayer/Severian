"""SIP-0017: indexed terms, sentences and blocks; distinct identity and containment."""
from dataclasses import dataclass, field
from py_compiler.frontend.source.source import Span


@dataclass(frozen=True)
class Literal:
    identity: str
    span: Span
    spelling: str
    lexical_kind: str
    expected_type: str | None
    explicit_constructor: bool = False


@dataclass(frozen=True)
class Sentence:
    identity: str
    block: str
    span: Span
    term: str
    name: str | None
    binding_operator: str | None


@dataclass
class Block:
    identity: str
    span: Span
    scope: str
    contents: list[str] = field(default_factory=list)


@dataclass
class BlockGraph:
    blocks: dict[str, Block] = field(default_factory=dict)
    sentences: dict[str, Sentence] = field(default_factory=dict)
    terms: dict[str, Literal] = field(default_factory=dict)
    dependencies: list[tuple[str, str]] = field(default_factory=list)
