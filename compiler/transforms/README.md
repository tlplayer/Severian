# Compiler transforms

Transforms change program representation without redefining source-language semantics.

## Stages

```text
HIR -> MIR -> lowering(TargetSpec) -> LIR -> emitter/backend
```

- MIR owns executable control flow and language-level operations.
- Lowering resolves target-dependent representation and expands operations when required.
- LIR owns concrete target-resolved types and backend-neutral lowered operations.
- MLIR and other emitters map LIR into their own syntax or APIs.

## Rules

1. Transforms receive `&UniversalContext`; they never reload primitive source.
2. Type and operator meaning is already resolved before MIR lowering.
3. Lowering may inspect typed `PrimitiveRepresentation`; it may not match raw category or representation strings.
4. `usize` and `isize` widths come from `TargetSpec`, not a fixed constant.
5. Shared lowered types and operations live in LIR, not in a backend crate.
6. C or MLIR spelling lives in the corresponding emitter, not on `LirType`.
7. Unsupported target capabilities return errors. No fallback to an unrelated width or format is allowed.
8. A transform must preserve stable IDs or record an explicit mapping.
