# Severian MLIR transform

This package owns the transition from Severian's physical compiler IR to MLIR.
Its canonical intermediate form is the source-level `MlirProgram` object graph
in `src/ir/`; textual `.mlir` is not the compiler's working representation.

## Data flow

```text
Universal/LIR objects
        |
        v
MlirProgram                 .sev-owned typed object graph
  MlirFunctionOp
  MlirRegion
  MlirBlock
  MlirOperation             open (dialect, mnemonic) identity
  MlirValue / MlirType
        |
        +--> structural verifier
        |
        +--> native materializer
               |
               v
          mlir::ModuleOp    live, in-memory MLIR object
               |
               v
          MLIR PassManager
               |
       +-------+-------+
       |       |       |
      LLVM    ROCDL   NVVM
```

`src/ir/model.sev` and `src/ir/builder.sev` are the source of truth for the
builder contract. Operation identity is deliberately open: adding
`dialect.mnemonic` does not add an enum case or a dtype/rank-named compiler
function. Element representation, rank, dimensions, and address space remain
fields on types.

`src/ir/validate.sev` rejects malformed SSA, region ownership, terminators,
known operation signatures, calls, and returns before native materialization.
Unknown operation names remain legal object data so dialect extensions do not
require edits to the core builder.

`src/ir/text.sev` is a terminal printer for `--emit=mlir`, diagnostics, golden
tests, and bootstrap comparison. No transform is permitted to parse or edit its
output.

## Native boundary

The bootstrap provider in `compiler/transforms/mlir/src/native.rs` constructs a
real `mlir::ModuleOp` through MLIR operation-state APIs. It does not accept a
module string. It owns the MLIR context and module lifetime, loads registered
dialects, accepts open operation names, attaches typed operands/results,
attributes and owned regions, and invokes MLIR's native verifier. Native handles
carry their originating context, so a type, attribute, value, block, region, or
operation from another builder is rejected before an MLIR call is made.

The native materialization vocabulary currently covers:

- signless, signed, and unsigned integers; index, BF16, F16, F32, F64, and
  address-space-qualified LLVM pointers;
- ranked and unranked tensors and memrefs, with rank zero distinct from unknown
  rank and `-1` reserved for a dynamic dimension;
- function signatures, typed block arguments, SSA operands/results, nested
  regions, declarations, definitions, and file/line/column locations;
- integer, floating, boolean, string, type, recursive array, flat-symbol, and
  structurally built affine-map attributes.

The only crate-visible module parser entry is instrumented. Native-construction
tests snapshot its call count around materialization and verification; a changed
count fails the test. Text parsing remains isolated to the legacy artifact
verifier while its callers are migrated.

This Rust layer is the narrow `Severian -> upstream MLIR` implementation
boundary, not a second compiler IR. The builder semantics and graph live in
`.sev`; Rust supplies native library access during bootstrap. As Severian's
native interfaces mature, this provider can move upward without changing the
`MlirProgram` contract.

## Ownership rules

- `MlirOpBuilder` owns source object identity and SSA numbering.
- A `MlirRegion` is transferred exactly once into its parent operation.
- A native operation is transferred exactly once into a block.
- A native function operation is transferred exactly once into the module.
- Builder handles cannot cross MLIR contexts.
- The native module owns all transferred operations, blocks, regions, types,
  attributes, and values until it is destroyed.
- Text owns nothing and cannot be fed back into production lowering.

## Current checkpoint

Implemented and tested:

- open typed operation construction in `.sev`;
- ranked tensor types with independently dynamic dimensions;
- structural validation before printing/materialization;
- Universal scalar lowering routed through the typed builder API;
- direct native `ModuleOp` construction and native MLIR verification;
- native source locations and provider-owned verifier diagnostics;
- structural affine maps and the complete attribute families listed above;
- checked context ownership and parser-call instrumentation;
- shared-MLIR linking, avoiding the former whole-archive memory spike.

Still required for the production executable path:

1. Materialize every `MlirProgram` node into the native builder, preserving a
   source-value-to-`MlirValue` table and region ownership.
2. Replace `StructuredModuleBuilder::print` consumers with that materializer.
3. Move the existing CPU/GPU pass pipelines into an in-process MLIR
   `PassManager`.
4. Translate the LLVM dialect to `llvm::Module`, emit an object with
   `TargetMachine`, and link it with embedded LLD.
5. Retain textual tools only behind explicit debug/bootstrap flags.

The executable source corpus is `examples/typed_module.sev`. Native construction
tests live beside the bootstrap provider.
