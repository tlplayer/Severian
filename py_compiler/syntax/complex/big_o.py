"""Function complexity is described by the BigO enum type."""
from enum import IntEnum


class BigO(IntEnum):
    constant = 0
    linear = 1
    quadratic = 2
    undefined = 3
    exponential = 4
    combinatorial = 5


def declaration():
    from py_compiler.syntax.block.enum import EnumType
    return EnumType('BigO', 'enum', 'i32', 32, False, tuple(BigO.__members__))
