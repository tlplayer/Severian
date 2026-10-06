from dataclasses import dataclass
from py_compiler.syntax.keywords.keyword import KEYWORDS
from py_compiler.syntax.primitive.catalog import primitives
from py_compiler.syntax.symbol.forms import NUMBER, SYMBOLS, quoted_end


@dataclass(frozen=True)
class Syntax:
    pointer_bits: int = 64
    keywords: frozenset = KEYWORDS
    symbols: tuple = SYMBOLS

    @property
    def block_providers(self):
        from py_compiler.syntax.block.block import providers
        return providers()

    @property
    def sentence_providers(self):
        from importlib import import_module
        from py_compiler.syntax.sentence.unimplemented import Unimplemented
        from py_compiler.syntax.keywords.drop import Drop
        from py_compiler.syntax.sentence.assignment import Assignment
        return (Unimplemented(), import_module("py_compiler.syntax.sentence.return").Return(), Drop(), Assignment())

    @property
    def scope_providers(self):
        from py_compiler.syntax.keywords.scope import SCOPES
        return SCOPES

    @property
    def types(self):
        return primitives(self.pointer_bits)

    def recognize(self, text, cursor):
        char = text[cursor]
        if char in "\"'":
            end = quoted_end(text, cursor)
            kind = "CHAR" if char == "'" and not text.startswith("'''", cursor) else "STRING"
            return kind, end
        if char.isalpha() or char == "_":
            end = cursor + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            return ("KEYWORD" if text[cursor:end] in self.keywords else "IDENTIFIER"), end
        match = NUMBER.match(text, cursor)
        if match:
            end = match.end()
            if end < len(text) and (text[end].isalnum() or text[end] == "_"):
                raise ValueError("invalid numeric literal or unsupported suffix")
            return "NUMBER", end
        for symbol in self.symbols:
            if text.startswith(symbol, cursor):
                return "SYMBOL", cursor + len(symbol)
        raise ValueError(f"unrecognized character {char!r}")
