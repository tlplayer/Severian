# Bitwise operators

API ID: `operator.binary`

```sev
def merge_flags(left: u64, right: u64, mask: u64) -> u64:
    return (left | right) & mask ^ right
```

`|`, `&`, and `^` select `BitwiseOr`, `BitwiseAnd`, and `BitwiseXor`.
They operate on integer bit patterns and are eager. They are distinct from the
short-circuit boolean operators `or` and `and`.

Width and signedness remain operand representation data. Backend lowering must
preserve the declared width and reject unsupported representations explicitly.

Current weakness: executable symmetry coverage currently exercises `u64`; it
does not yet span the full fixed-width integer matrix.
