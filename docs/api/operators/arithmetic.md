# Arithmetic operators

API ID: `operator.binary`

```sev
def polynomial(x: i64) -> i64:
    return x ** 2 + 3 * x - 4
```

Arithmetic resolves `Add`, `Subtract`, `Multiply`, `Divide`, `Remainder`, or
`Power` against operand signatures. Result type and overflow behavior come from
the selected signature. Integer division and floating division use different
scalar lowering while retaining `Divide` as the source operation identity.

Division/remainder by zero and checked integer overflow are observable errors.

Current weakness: backend execution coverage is not exhaustive for all
registered widths and formats.
