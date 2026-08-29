# Bytes, absence, unit, and process arguments

API ID: `primitive.text_and_storage`

`bytes` is untyped byte storage. `None` is the absence member used in unions.
`unit` is the result of an operation with no value. `args` is the registered
process-argument representation exposed through the process boundary.

```sev
def preserve_bytes(value: bytes) -> bytes:
    return value

def choose(found: bool, value: string) -> string | None:
    if found:
        return value
    return None
```

These types must not be conflated because some lower to pointer-bearing ABI
records while `None` and `unit` may carry no payload. The active union member
controls ownership and destruction.

Current weakness: several runtime APIs return higher-level `list[string]`
views instead of exposing `args` directly, so their exact relationship needs a
dedicated ABI page.
