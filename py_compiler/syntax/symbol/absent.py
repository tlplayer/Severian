from py_compiler.syntax.primitive.contract import Primitive


def declaration(pointer_bits):
    return Primitive("absent", "absence", "!llvm.ptr", pointer_bits)
