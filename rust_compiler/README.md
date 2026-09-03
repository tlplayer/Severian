# Severian compiler architecture

The compiler uses one owner for each semantic concept. Compiler phases may carry different representations, but they may not independently redefine the same language rule.

## Rust directory map

```text
compiler/
  universal/            language-wide semantic authority
    primitive/          primitive schema, capabilities, and operators
    types/              type context, compatibility, and resolution
    literal.rs          canonical literal kinds and values
    operator.rs         canonical operator identities and signatures
    ids.rs              stable universal identities
    type_system.rs      structural types, inference, and constraints
  source/               source files, spans, and source identity
  frontend/
    lexer/              tokenization
    parser/             syntax construction
    ast/                source-preserving syntax model
    modules/            packages, modules, imports, and visibility
    semantic/           validation and enrichment through universal
    hir/                typed program representation
    ownership/          ownership and effect validation
  transforms/
    mir/                control/data-flow representation
    lir/                target-resolved lowering representation
    lowering/           MIR to LIR
    mlir/               MLIR construction and verification
  compile/              CompileType planning and dispatch
  bootstrap/            universal context and source-protocol assembly
  target/               target, device, feature, and capability selection
  diagnostics/          diagnostics and rendering
  boundaries/           ABI, FFI, interfaces, backends, XXI, and driver
  runtime/              native runtime implementations
  artifact/             compiled artifact identities and metadata
```

AST, HIR, MIR, and LIR may retain stage-specific representations, but language
facts shared between stages must be referenced from `universal` rather than
redeclared by those stages.

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
  -> CompileType planner
       Standard: MIR -> LIR -> MLIR
       Compiler: typed MIR region -> verified MLIR
  -> MLIR composition and target pipeline
  -> backend/link emission
  -> artifact
```

`UniversalContext` is built once by the driver and passed through the pipeline. No later phase reloads or reparses core primitive source.

ABI and FFI are typed services beside this pipeline. Lowering consults them
only for an external boundary operation; ordinary code never passes through
XXI, FFI, or ABI phases.

## Ownership

| Concept | Sole owner |
| --- | --- |
| Stable declaration and type identities | `compiler/universal` |
| Primitive definitions, metadata, capabilities, and operators | `compiler/universal/primitive` |
| Structural and nominal type context | `compiler/universal/types` |
| Literal kinds, canonical constant values, and literal resolution | `compiler/universal` |
| Operator identities, signatures, and result resolution | `compiler/universal` |
| Raw source spelling and source spans | AST |
| Typed expressions and bindings | HIR |
| Executable operations and control flow | MIR |
| Stable CompileType routes | `compiler/universal` |
| MIR region partitioning and handler dispatch | `compiler/compile` |
| `Compiler` and `CompileType[C]` source protocols | `library/core/compile` |
| Neutral target, feature, device, and capability selection | `compiler/target` |
| Target-resolved physical types and operations | LIR |
| Calling conventions, concrete layouts, pass modes, and symbols | `compiler/boundaries/abi` |
| Foreign ownership, lifetime, and conversion plans | `compiler/boundaries/ffi` |
| `@c`/`@rust` declarations and external imports | `compiler/boundaries/xxi` |
| C, MLIR, LLVM, XLA, or Triton spelling | The corresponding backend/emitter |
| `.pkg` and `.pkgi` serialization | `compiler/boundaries/interface` |
| Loading compiler protocols into stable routes | `compiler/bootstrap` |

## Dependency direction

```text
universal <- bootstrap
universal <- frontend
universal <- HIR/MIR/compile/lowering
universal <- interface conversion

AST/HIR/MIR/LIR flow forward only.
boundaries consume compiler models; they do not define language semantics.
library source never depends on Rust compiler crates.
compile -> universal + MIR + MLIR interface + target
```

`compiler/universal` must not depend on frontend, HIR, MIR, LIR, lowering, a backend, an interface format, or the standard library.

## Hard rules

1. Primitive names, categories, representations, literal defaults, capabilities, and operator signatures are defined only in `compiler/universal/primitive`.
2. Bootstrap installs universal primitives before loading source-defined compiler protocols.
3. A compiler phase receives `&UniversalContext`; it does not call a global `load()` function.
4. Semantic analysis delegates literal and operator resolution to `compiler/universal`.
5. Lowering accepts typed definitions and a neutral `TargetSpec`; it does not reinterpret source strings.
6. Backends consume LIR and return an unsupported-capability error when they cannot represent an operation. Silent fallback is prohibited.
7. A semantic enum or ID is defined once. A second copy is allowed only when the new representation has a distinct invariant or loses/gains information.
8. Backend spelling methods do not live on universal or LIR types.
9. Interface DTOs are serialization models, not the live compiler type system.
10. Every architectural move must preserve or add a vertical end-to-end test.
11. CompileType routes are stable declaration identities. Universal, MIR,
    lowering, and backends contain no specialized type or compiler names.
12. Custom handlers emit MLIR that is signature-checked, capability-checked,
    and verified before it rejoins ordinary MLIR.
13. External declarations remain ordinary decorated declarations. FFI and ABI
    are consulted only while lowering an external call operation.

## Change test

Before adding a match table, enum, catalog, or loader, answer:

- Which directory owns this fact?
- Can an existing owner answer the question through an API?
- Would adding this code create a second source of truth?

If the answer to the last question is yes, add or extend the owner API instead.
