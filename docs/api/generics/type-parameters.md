# Type parameters

API ID: `generic.function`

```sev
def identity[T](value: T) -> T:
    return value

test "explicit ordinary type application":
    assert(identity[int](42) == 42)
```

An applied call preserves the callee definition identity and carries a
substitution such as `{T → int}`. The substitution applies to parameters,
results, locals, nested applications, defaults, calls, and class applications.
It does not rename the function to `identity_int`.

Constraints participate in ordinary overload resolution. Wrong generic arity,
an unsatisfied constraint, or an ambiguous overload is a compile-time error.

Current weakness: downstream dependency interfaces do not preserve every
generic body and enum constructor consistently, so some cross-package applied
calls cannot yet be specialized from their complete bodies.
