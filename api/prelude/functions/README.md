# Prelude functions

API ID: `prelude.function.assert`

Always-visible functions include assertions, failure, printing, ranges, sizes, enumeration, zipping, indices, presence tests, and typed error matching. They resolve as ordinary callable identities.

```sev
def prelude_size(values: list[int]) -> usize:
    assert(size(values) >= 0)
    return size(values)
```

Current weakness: effect and allocation summaries are not yet expanded per function in generated docs.
