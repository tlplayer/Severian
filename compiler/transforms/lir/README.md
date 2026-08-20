# Lowered intermediate representation

LIR is the shared output of target lowering and the input to concrete emitters.

## Owns

- Target-resolved value types.
- Backend-neutral constants.
- Lowered operations.
- Value, block, function, and module identities.
- Optional debug/source-location references.

Examples of LIR types include a concrete signed 32-bit integer, BF16, a target pointer, or a runtime string handle. Pointer-sized source types have already been resolved using the ABI target layout.

## Does not own

- Primitive source declarations.
- Literal or operator resolution.
- Primitive category strings.
- C, MLIR, LLVM, XLA, or Triton spelling.
- Backend capability fallback.
- Artifact creation or linker invocation.

## Dependency rule

```text
MIR + UniversalContext + ABI Target -> lowering -> LIR
LIR -> C emitter
LIR -> MLIR emitter
LIR -> other backend emitters
```

Emitters may extend their local model after LIR, but shared lowered semantics remain here.
