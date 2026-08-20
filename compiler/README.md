# Severian compiler architecture

The compiler uses one owner for each semantic concept. Compiler phases may carry different representations, but they may not independently redefine the same language rule.

## Pipeline

```text
source
  -> lexer
  -> parser
  -> AST
  -> bootstrap/declaration loading
  -> UniversalContext
  -> semantic analysis
  -> HIR
  -> ownership validation
  -> MIR
  -> target lowering
  -> LIR
  -> backend emission
  -> artifact
```

`UniversalContext` is built once by the driver and passed through the pipeline. No later phase reloads or reparses core primitive source.

## Ownership

| Concept | Sole owner |
| --- | --- |
| Stable declaration and type identities | `compiler/universal` |
| Type definitions and primitive metadata | `compiler/universal` |
| Literal kinds, canonical constant values, and literal resolution | `compiler/universal` |
| Operator identities, signatures, and result resolution | `compiler/universal` |
| Raw source spelling and source spans | AST |
| Typed expressions and bindings | HIR |
| Executable operations and control flow | MIR |
| Target-resolved physical types and operations | LIR |
| C, MLIR, LLVM, XLA, or Triton spelling | The corresponding backend/emitter |
| `.pkg` and `.pkgi` serialization | `compiler/boundaries/interface` |
| Loading `library/core/primitives` | `compiler/bootstrap` |

## Dependency direction

```text
universal <- bootstrap
universal <- frontend
universal <- HIR/MIR/lowering
universal <- interface conversion

AST/HIR/MIR/LIR flow forward only.
boundaries consume compiler models; they do not define language semantics.
library source never depends on Rust compiler crates.
```

`compiler/universal` must not depend on frontend, HIR, MIR, LIR, lowering, a backend, an interface format, or `library/core/primitives`.

## Hard rules

1. Primitive names, categories, representations, literal defaults, and operator names are not matched directly outside bootstrap validation and universal resolution.
2. Only `compiler/bootstrap` reads `library/core/primitives`.
3. A compiler phase receives `&UniversalContext`; it does not call a global `load()` function.
4. Semantic analysis delegates literal and operator resolution to `compiler/universal`.
5. Lowering accepts typed definitions and `TargetSpec`; it does not reinterpret source strings.
6. Backends consume LIR and return an unsupported-capability error when they cannot represent an operation. Silent fallback is prohibited.
7. A semantic enum or ID is defined once. A second copy is allowed only when the new representation has a distinct invariant or loses/gains information.
8. Backend spelling methods do not live on universal or LIR types.
9. Interface DTOs are serialization models, not the live compiler type system.
10. Every architectural move must preserve or add a vertical end-to-end test.

## Change test

Before adding a match table, enum, catalog, or loader, answer:

- Which directory owns this fact?
- Can an existing owner answer the question through an API?
- Would adding this code create a second source of truth?

If the answer to the last question is yes, add or extend the owner API instead.
