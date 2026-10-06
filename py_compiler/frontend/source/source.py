"""SIP-0019: immutable source snapshots; scalar coordinates, explicit UTF-8 boundary."""
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Span:
    source: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceFile:
    path: str
    text: str

    @property
    def identity(self):
        return sha256((self.path + "\0" + self.text).encode()).hexdigest()

    def span(self, start, end):
        if not 0 <= start <= end <= len(self.text):
            raise ValueError("span outside source snapshot")
        return Span(self.identity, start, end)

    def position(self, offset):
        prefix = self.text[:offset].replace("\r\n", "\n").replace("\r", "\n")
        return prefix.count("\n") + 1, len(prefix.rsplit("\n", 1)[-1]) + 1

    def utf8_span(self, span):
        if span.source != self.identity:
            raise ValueError("span belongs to another snapshot")
        return len(self.text[:span.start].encode()), len(self.text[:span.end].encode())


class Diagnostic(Exception):
    def __init__(self, stage, message, source, span):
        super().__init__(message)
        self.stage, self.message, self.source, self.span = stage, message, source, span

    def __str__(self):
        line, column = self.source.position(self.span.start)
        lines = self.source.text.splitlines()
        excerpt = lines[line - 1] if line <= len(lines) else ""
        marker = " " * (column - 1) + "^" * max(1, min(self.span.end - self.span.start, 60))
        return f"{self.source.path}:{line}:{column}: {self.stage}: {self.message}\n  {excerpt}\n  {marker}"


import unittest


class SourceTests(unittest.TestCase):
    def test_scalar_and_byte_coordinates(self):
        source = SourceFile("unicode.sev", "λ😀\nx")
        self.assertEqual(source.utf8_span(source.span(1, 2)), (2, 6))
        self.assertEqual(source.position(3), (2, 1))
        with self.assertRaises(ValueError):
            source.span(0, 6)
