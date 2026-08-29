# Conversions

API ID: `conversion.kind`

Conversions use one structural identity plus `Identity`, `Promote`, `Checked`, or `Lossy` policy data. The target constructor chooses a registered conversion; source and destination representations stay fields, never operation or symbol suffixes.

```sev
def conversion_probe(value: i32) -> i64:
    return i64(value)
```

HIR preserves `{from, to, kind}` through backend lowering. Current weakness: aggregate and user-defined conversions do not yet have the same exhaustive matrix as numeric primitives.
