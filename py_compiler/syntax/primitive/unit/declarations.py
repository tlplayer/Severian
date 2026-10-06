from py_compiler.syntax.primitive.contract import Primitive

# Mirrors UnitPrimitive's current no-value MLIR declaration.
# Dimensioned quantities (e.g. byte) retain their own representation and identity.
UNIT = Primitive("unit", "unit", "")
