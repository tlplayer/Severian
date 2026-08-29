# `bool`

API ID: `primitive.boolean`; Universal path: `universal.primitive.bool`.

## Representation

`bool` uses `PrimitiveRepresentation.Boolean`, is a default boolean-literal
target, and is a Copy value. Scalar MLIR uses `i1`. FFI lowering uses the
target ABI boolean conversion rather than assuming that a C `_Bool` and an
arbitrary integer have identical call representation.

## Source semantics

The values are `true` and `false`. `not` is unary; `and` and `or` are
short-circuit logical operators. Equality and inequality are defined, but
ordering and implicit numeric truthiness are not. Short-circuiting must preserve
effects by skipping the right operand when its value is unnecessary.

```sev
def enabled(ready: bool, permitted: bool) -> bool:
    return ready and permitted
```

