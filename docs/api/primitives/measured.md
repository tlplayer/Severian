# Measured values

API ID: `primitive.measured`

The registry currently includes `data_size`, `duration`, `data_rate`,
`frequency`, `percentage`, `temperature`, `voltage`, `current`, and `power`.
They use an `f64` physical representation but retain distinct semantic types.

```sev
def within_budget(elapsed: duration, budget: duration) -> bool:
    return elapsed <= budget
```

Matching quantities may be compared and added. Multiplication or division is
legal only when the dimensional result is defined. Representation equality is
not permission to add a temperature to a duration.

Current weakness: the primitive registry is complete, but the public API does
not yet enumerate and test the complete dimensional algebra. That keeps this
section partial.
