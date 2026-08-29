# Prelude constants

API ID: `prelude.constant.none`

Always-visible literal and contextual values include `true`, `false`, `None`, `self`, and `runtime`. Contextual values are resolved by scope and never emitted as global mutable state.

```sev
def absent() -> None:
    return None
```

Current weakness: contextual constant availability is not yet rendered as a scope table.
