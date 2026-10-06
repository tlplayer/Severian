from py_compiler.syntax.primitive.contract import Primitive


def declarations(pointer_bits):
    values = [Primitive(f"{prefix}{bits}", "integer", f"i{bits}", bits, prefix == "i")
              for prefix in ("i", "u") for bits in (8, 16, 32, 64, 128)]
    return values + [Primitive("index", "integer", "index", pointer_bits, True),
                     Primitive("isize", "integer", f"i{pointer_bits}", pointer_bits, True),
                     Primitive("usize", "integer", f"i{pointer_bits}", pointer_bits)]
