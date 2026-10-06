from py_compiler.syntax.primitive.contract import Primitive


def declarations():
    return [Primitive(name, "float", spelling, bits) for name, spelling, bits in (
        ("f16", "f16", 16), ("f32", "f32", 32), ("f64", "f64", 64),
        ("f128", "f128", 128), ("bf16", "bf16", 16),
        ("f8e4m3fn", "f8E4M3FN", 8), ("f8e5m2", "f8E5M2", 8))]
