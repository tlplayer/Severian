"""Typed primitive values; Python storage never determines Severian type identity."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from py_compiler.frontend.source.source import Diagnostic, Span
from py_compiler.syntax.primitive.catalog import Primitive
from py_compiler.syntax.symbol.forms import decode_quoted


@dataclass(frozen=True)
class Constant:
    identity: str
    name: str | None
    type: Primitive
    value: int | str | Decimal | bool | None
    span: Span
    source: str
    binding_operator: str | None

    def atom(self, operation):
        from py_compiler.syntax.generic.atom import Atom
        return Atom(self.type, (), self.type, (), (), self)

    def render_operation(self, operation):
        from hashlib import sha256
        from py_compiler.mlir.src.cfg import name
        symbol = "__sev_constant_" + sha256(self.identity.encode()).hexdigest()
        return [f"{name(operation.result)} = func.call @{symbol}() : () -> {self.type.mlir}"]


@dataclass(frozen=True)
class Submodule:
    identity: str
    declarations: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class Program:
    constants: tuple[Constant, ...]
    submodules: tuple[Submodule, ...]
    bodies: tuple = ()
    declarations: tuple = ()


def resolve_literal(literal, syntax):
    from py_compiler.frontend.source.source import SourceFile
    from py_compiler.syntax.generic.grammar import SourceWindow
    spelling = literal.spelling
    unsigned = spelling[1:] if spelling.startswith(('+', '-')) else spelling
    source = SourceFile('<literal>', unsigned)
    if not unsigned:
        raise ValueError('empty literal')
    match = syntax.registry.recognize('Y', SourceWindow(source, 0, len(unsigned)))
    owner = getattr(match.provider, 'owner', None)
    if match.end != len(unsigned) or owner is None or not hasattr(owner, 'decode'):
        raise ValueError('no type declares this literal')
    expected = literal.expected_type
    if expected and expected not in syntax.types:
        raise ValueError(f'unknown primitive type {expected!r}')
    target = syntax.types[expected] if expected else owner
    value = owner.decode(spelling)
    return target, target.accept_literal(owner, value)


def partition(source, graph, syntax):
    constants, submodules, diagnostics = [], [], []
    for sentence in graph.sentences.values():
        literal = graph.terms[sentence.term]
        try:
            type_, value = resolve_literal(literal, syntax)
            constants.append(Constant(sentence.identity, sentence.name, type_, value, literal.span,
                                      source.path, sentence.binding_operator))
            # This milestone has no references: every declaration is a singleton SCC.
            submodules.append(Submodule(sentence.identity, (sentence.identity,)))
        except (ValueError, InvalidOperation) as failure:
            diagnostics.append(Diagnostic("HIR", str(failure), source, literal.span))
    return Program(tuple(constants), tuple(submodules)), diagnostics


import unittest
from py_compiler.frontend.parser.contract import Literal
from py_compiler.syntax.recognition import Syntax


class HirTests(unittest.TestCase):
    def value(self, spelling, expected=None, kind="NUMBER"):
        return resolve_literal(Literal("x", Span("x", 0, len(spelling)), spelling, kind, expected), Syntax())

    def test_integer_boundaries(self):
        for prefix in ("i", "u"):
            for bits in (8, 16, 32, 64, 128):
                name = f"{prefix}{bits}"
                low = -(1 << (bits - 1)) if prefix == "i" else 0
                high = (1 << (bits - (prefix == "i"))) - 1
                for number in (low, high):
                    self.assertEqual(self.value(str(number), name)[1], number)
                for number in (low - 1, high + 1):
                    with self.assertRaises(ValueError):
                        self.value(str(number), name)

    def test_no_python_coercion_or_precision_loss(self):
        with self.assertRaises(ValueError):
            self.value("true", "i32", "KEYWORD")
        self.assertEqual(str(self.value("1.00000000000000000001", "f128")[1]), "1.00000000000000000001")
        self.assertEqual(self.value("4B")[0].name, "byte")
        with self.assertRaises(ValueError):
            self.value("4B", "u8")
