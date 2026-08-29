# Unary operators

API ID: `operator.unary`

```sev
def direction(value: int, enabled: bool) -> int:
    if not enabled:
        return 0
    return -value if value > 0 else +value
```

`+` selects `UnaryOperator.Positive`, `-` selects `Negative`, and `not`
selects `Not`. The selected operator declaration determines input ownership and
result type. Numeric sign operations and boolean negation are not interchangeable.

Target legality depends on the resolved representation. A missing target
lowering is reported by legalization rather than by inventing a new operator.

Current weakness: the representation/target matrix is not yet exhaustive for
all numeric widths.
