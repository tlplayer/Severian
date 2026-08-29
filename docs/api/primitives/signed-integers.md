# Signed integers

API ID: `primitive.signed_integer`

`int` uses the language's machine integer representation. `i8`, `i16`, `i32`,
`i64`, and `i128` have fixed widths; `isize` has pointer width. Width is type
data (`B` in the appendix), not an operation suffix.

```sev
def widen(value: i8) -> i128:
    return i128(value)

test "signed width is explicit":
    small: i8 = 8
    wide: i128 = widen(small)
    assert(wide == 8)
```

Checked arithmetic reports overflow rather than silently changing width.
Division and remainder report division by zero. Comparisons return `bool`.
Lowering selects signed MLIR operations where signedness matters.

Current weakness: the API registry contains the full width family, but every
operation/width/target combination is not yet represented by a passing backend
conformance matrix.
