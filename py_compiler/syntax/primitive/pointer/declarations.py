from py_compiler.syntax.primitive.contract import Primitive


def declaration(pointer_bits):
    return Primitive("pointer", "pointer", "!llvm.ptr", pointer_bits)
