from py_compiler.syntax.primitive.numeric.types import Byte

# A storage quantity retains its own semantic identity, distinct from u8 and int.
BYTE = Byte("byte", "byte", "i64", 64, True)
