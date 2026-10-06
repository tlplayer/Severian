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
    spelling = literal.spelling
    types = syntax.types
    expected = literal.expected_type
    if expected and expected not in types:
        raise ValueError(f"unknown primitive type {expected!r}")
    kind, value = "", None
    if literal.lexical_kind == "CHAR":
        kind, value = "char", ord(decode_quoted(spelling))
    elif literal.lexical_kind == "STRING":
        kind, value = "string", decode_quoted(spelling)
    elif spelling in ("true", "false"):
        kind, value = "bool", spelling == "true"
    elif spelling in ("None", "absent", "unit"):
        kind = spelling
    else:
        number = spelling.replace("_", "")
        if number.endswith("B") and not number.lstrip("+-").lower().startswith("0x"):
            kind, number = "byte", number[:-1]
        base = number.lstrip("+-").lower()
        if base.startswith(("0x", "0o", "0b")):
            value = int(number, 0)
            kind = kind or "i64"
        elif any(c in number for c in ".eE"):
            if kind == "byte":
                raise ValueError("byte quantities require an integral amount")
            kind, value = "f64", Decimal(number)
        else:
            kind, value = kind or "i64", int(number, 10)
    target = types[expected] if expected else types[kind]
    family = target.family
    if family == "integer" and kind == "i64":
        low = -(1 << (target.bits - 1)) if target.signed else 0
        high = (1 << (target.bits - int(target.signed))) - 1
        if not low <= value <= high:
            raise ValueError(f"literal {value} is outside {target.name} range [{low}, {high}]")
    elif family == "float" and kind in ("i64", "f64"):
        value = Decimal(value)
        if not value.is_finite():
            raise ValueError("non-finite numeric literals require an explicit provider")
    elif target.name == "pointer" and kind == "i64" and expected:
        if not 0 <= value < (1 << target.bits):
            raise ValueError("pointer address outside target width")
    elif target.name != kind:
        raise ValueError(f"{kind} literal does not satisfy {target.name}")
    if kind == "byte" and not -(1 << 63) <= value < (1 << 63):
        raise ValueError("byte quantity exceeds its signed 64-bit representation")
    return target, value


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
