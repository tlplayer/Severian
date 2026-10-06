from py_compiler.syntax.primitive.contract import Primitive

# A storage quantity retains its own semantic identity, distinct from u8 and int.
BYTE = Primitive("byte", "byte", "i64", 64, True)
