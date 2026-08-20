# Lowering

Lowering converts typed MIR into target-resolved LIR.

## API

```text
lower(mir, universal_context, target_spec) -> LirModule
```

Lowering may:

- Resolve pointer-sized integer width from the target.
- Select concrete calling and memory representations.
- Expand language operations into runtime or target operations.
- Reject a representation unsupported by the selected target.

Lowering may not:

- Call `severian_primitives::load()` or `definition()`.
- Read `.sev` files.
- Match `category.as_str()` or `representation.as_str()`.
- Decide whether a source operator is legal.
- Define C or MLIR type spelling.
- Silently map BF16 to F32 or an unknown integer width to I64.

The universal type definition is passed in or retrieved from the injected context. The lowered result is stored once in LIR and reused by emitters.
