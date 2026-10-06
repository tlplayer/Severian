"""Representation declarations mirror syntax/primitive; language identity survives lowering."""
from py_compiler.syntax.primitive.contract import Primitive
from py_compiler.syntax.primitive.int.declarations import declarations as integers
from py_compiler.syntax.primitive.float.declarations import declarations as floats
from py_compiler.syntax.primitive.bool.declarations import BOOL
from py_compiler.syntax.primitive.char.declarations import CHAR
from py_compiler.syntax.primitive.string.declarations import STRING
from py_compiler.syntax.primitive.byte.byte import BYTE
from py_compiler.syntax.primitive.unit.declarations import UNIT
from py_compiler.syntax.primitive.pointer.declarations import declaration as pointer
from py_compiler.syntax.symbol.none import declaration as none
from py_compiler.syntax.symbol.absent import declaration as absent


def primitives(pointer_bits=64):
    if pointer_bits not in (32, 64):
        raise ValueError("pointer width must be 32 or 64")
    values = integers(pointer_bits) + floats()
    values += [BOOL, CHAR, STRING, BYTE, UNIT, pointer(pointer_bits), none(pointer_bits), absent(pointer_bits)]
    result = {value.name: value for value in values}
    # Family defaults, not extra runtime wrapper types.
    result["int"], result["float"] = result["i64"], result["f64"]
    return result


import unittest


class PrimitiveTests(unittest.TestCase):
    def test_semantic_identity_is_not_storage_identity(self):
        types = primitives(32)
        self.assertNotEqual(types["u8"], types["i8"])
        self.assertEqual(types["usize"].mlir, "i32")
        self.assertEqual(types["byte"].family, "byte")
        self.assertNotEqual(types["None"], types["absent"])
