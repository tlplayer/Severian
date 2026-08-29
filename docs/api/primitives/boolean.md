# Boolean

API ID: `primitive.boolean`

`bool` contains `true` and `false`. It is a Copy value with no ownership or
allocation effect. Conditions for `if`, `while`, and logical operators require
`bool`; numeric values do not become booleans implicitly.

```sev
def both(left: bool, right: bool) -> bool:
    return left and right

test "boolean logic":
    assert(both(true, true))
    assert(not both(true, false))
```

The universal representation is `PrimitiveRepresentation.Boolean`; MLIR uses
`i1`. Short-circuit `and` and `or` are control-flow operations, not eager
bitwise operations.

Current weakness: editor highlighting is lexical, so it cannot yet distinguish
a boolean condition from an identifier with an invalid inferred type.
