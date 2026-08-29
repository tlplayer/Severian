# Patterns

API ID: `pattern.destructure`

Destructuring, match cases, typed case bindings, and comprehension bindings are the current pattern surface. Bindings introduce names with explicit scope and ownership behavior.

```sev
def pattern_probe(values: list[int]) -> int:
    for value in values:
        return value
    return 0
```

Patterns lower to typed projections and control tests. Current weakness: nested algebraic destructuring is not yet documented as a complete grammar.
