# Constraints

API ID: `constraint.generic_bound`

Generic bounds, variadic shape packs, value predicates, function contracts, and property constraints have distinct evaluation times. Shape equality and divisibility can be proven at compile time or retained as specialization guards.

```sev
def constrained[T](value: T) -> T:
    return value
```

Resolved constraints become HIR evidence or runtime guards, not specialized source names. Current weakness: source syntax for the full value-constraint algebra is still stabilizing.
