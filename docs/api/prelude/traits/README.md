# Prelude traits

API ID: `prelude.trait.iterable`

Always-visible behavioral contracts support errors, iteration, operator selection, and capability bounds. Traits constrain generic parameters without changing function identity.

```sev
def trait_subject[T](value: T) -> T:
    return value
```

Current weakness: every bootstrap trait method is not yet expanded into its own machine record.
