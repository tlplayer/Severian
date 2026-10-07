"""Literal recognition and decoding belong to syntax (SIP-0011, SIP-0018)."""
import re

NUMBER = re.compile(r"(?:0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*|0[bB][01](?:_?[01])*|0[oO][0-7](?:_?[0-7])*|[0-9](?:_?[0-9])*(?:\.(?!\.)(?:[0-9](?:_?[0-9])*)?)?(?:[eE][+-]?[0-9](?:_?[0-9])*)?)(?:B)?")


def quoted_end(text, start):
    quote = text[start]
    delimiter = quote * 3 if text.startswith(quote * 3, start) else quote
    cursor = start + len(delimiter)
    while cursor < len(text):
        if text.startswith(delimiter, cursor):
            return cursor + len(delimiter)
        if text[cursor] in "\r\n" and len(delimiter) == 1:
            raise ValueError("line ending in quoted literal")
        cursor += 2 if text[cursor] == "\\" else 1
    raise ValueError("unterminated quoted literal")


def decode_quoted(spelling):
    width = 3 if spelling.startswith(spelling[0] * 3) else 1
    text = spelling[width:-width]
    result, cursor = [], 0
    escapes = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "\\": "\\", "'": "'", '"': '"'}
    while cursor < len(text):
        value = text[cursor]
        cursor += 1
        if value == "\\":
            if cursor == len(text):
                raise ValueError("incomplete escape")
            escape = text[cursor]
            cursor += 1
            if escape in escapes:
                value = escapes[escape]
            elif escape in ("x", "u", "U"):
                braced = escape == "u" and text[cursor:cursor + 1] == "{"
                if braced:
                    cursor += 1
                    end = text.find("}", cursor)
                    if end < 0 or not 1 <= end - cursor <= 6:
                        raise ValueError("invalid Unicode escape")
                else:
                    end = cursor + {"x": 2, "u": 4, "U": 8}[escape]
                digits = text[cursor:end]
                if end > len(text) or not re.fullmatch("[0-9a-fA-F]+", digits):
                    raise ValueError("invalid hexadecimal escape")
                scalar = int(digits, 16)
                if scalar > 0x10FFFF or 0xD800 <= scalar <= 0xDFFF:
                    raise ValueError("escape is not a Unicode scalar")
                value = chr(scalar)
                cursor = end + int(braced)
            else:
                raise ValueError(f"unknown escape \\{escape}")
        result.append(value)
    decoded = "".join(result)
    if width == 1 and spelling[0] == "'" and len(decoded) != 1:
        raise ValueError("character literal requires exactly one Unicode scalar")
    return decoded


import unittest


class LiteralTests(unittest.TestCase):
    def test_unicode_escape_contract(self):
        self.assertEqual(decode_quoted(r"'\u{1f600}'"), "😀")
        self.assertEqual(decode_quoted('"λ\\n"'), "λ\n")
        for invalid in (r"'\uD800'", "'ab'", r"'\q'"):
            with self.assertRaises(ValueError):
                decode_quoted(invalid)
