from py_compiler.syntax.primitive.numeric.types import Integer


def declarations(pointer_bits):
    values = [Integer(f"{prefix}{bits}", "integer", f"i{bits}", bits, prefix == "i")
              for prefix in ("i", "u") for bits in (8, 16, 32, 64, 128)]
    return values + [Integer("index", "integer", "index", pointer_bits, True),
                     Integer("isize", "integer", f"i{pointer_bits}", pointer_bits, True),
                     Integer("usize", "integer", f"i{pointer_bits}", pointer_bits)]
