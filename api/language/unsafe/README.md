# Unsafe

API ID: `unsafe.scope`

`unsafe` scopes explicitly admit operations whose memory-safety obligations cannot be proven. Unsafe never erases type, tensor shape, effect, backend legality, or ABI validation.

```sev
def safe_subject(value: int) -> int:
    return value
```

HIR records the unsafe boundary for diagnostics and auditing. Current weakness: unsafe-operation categories are not yet individually queryable API members.
