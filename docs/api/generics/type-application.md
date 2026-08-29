# Type application

API ID: `generic.type_application`

`callee[T](arguments...)` is ordinary generic call syntax. Resolution validates
generic kind and arity, selects an overload, and produces a resolved callee plus
substitution.

```sev
def pair[T](left: T, right: T) -> list[T]:
    return [left, right]

test "application selects one body":
    assert(pair[i64](20, 22) == [20, 22])
```

Specialization discovery must inspect the applied callee and specialize the
complete function body. Library functions such as `load[T]` receive no special
semantic branch: their type argument follows this same path.

Current weakness: cross-package body preservation remains incomplete for some
self-hosted declarations. This is a generic interface problem, not a reason to
add dtype-named loader functions.
