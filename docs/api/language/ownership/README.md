# Ownership

API ID: `ownership.borrow`

Move, copy, shared borrow, mutable borrow, address-of, and explicit drop are language API. Calls and operators use resolved signatures to determine how operands are consumed; optimizers may not infer ownership from naming conventions.

```sev
def ownership_probe(value: int) -> int:
    copied = copy value
    return copied
```

Ownership checking annotates HIR before MIR. Current weakness: editor semantic tokens do not yet expose move/borrow state inline.
