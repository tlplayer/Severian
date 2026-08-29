# Logical and membership operators

API ID: `operator.binary`

```sev
def acceptable(value: int, allowed: list[int], ready: bool) -> bool:
    return ready and value in allowed
```

`and` and `or` short-circuit: the right operand is evaluated only when needed.
`in` selects the `Contains` identity and resolves against the right operand's
membership signature.

Short-circuit behavior is an observable control/effect contract. A lowering
must not eagerly execute a right operand that mutates state or throws.

Current weakness: effectful short-circuit and custom-container membership need
dedicated symmetry cases beyond the scalar reference slice.
