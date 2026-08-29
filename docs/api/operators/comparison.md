# Comparison operators

API ID: `operator.binary`

```sev
def ordered(left: f32, middle: f32, right: f32) -> bool:
    return left <= middle and middle < right
```

The six comparison identities return the result declared by their visible
signature, normally `bool`. Integer lowering observes signedness; floating
lowering must define ordered/unordered NaN behavior rather than inheriting it
accidentally from a backend.

Current weakness: the API still needs an explicit floating NaN comparison table
and symmetry cases for every predicate.
