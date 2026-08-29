# Concurrency

API ID: `concurrency.async`

`async`, `await`, task ownership, channel selection, locks, and placement policies describe concurrency without changing called-operation identity. Task owners are self scope, runtime, or inferred.

```sev
def concurrency_value(value: int) -> int:
    return value
```

Async lowering carries ownership and suspension points into MIR. Current weakness: a portable executable async/channel backend matrix is not complete.
