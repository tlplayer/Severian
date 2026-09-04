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
  MlirOperationState        universal source-owned construction contract
  MlirOperation             completed open (dialect, mnemonic) operation
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
builder contract. `MlirOperationState` carries operands, expected result types,
typed attributes, typed properties, successors, regions, per-region terminator
requirements, location, and result-inference intent. Operation identity is
deliberately open: adding
`dialect.mnemonic` does not add an enum case or a dtype/rank-named compiler
function. Element representation, rank, dimensions, and address space remain
fields on types.

`src/ir/validate.sev` rejects malformed SSA, region ownership, terminators,
duplicate attributes/properties, missing successors, calls, and returns before
native materialization. It contains no dialect operation-name signature table;
dialect semantics are checked by source operation definitions and upstream
MLIR verification. Unknown operation names remain legal object data so dialect
extensions do not require edits to the core builder.

`src/ir/text.sev` is a terminal printer for `--emit=mlir`, diagnostics, golden
tests, and bootstrap comparison. No transform is permitted to parse or edit its
output.

## Generated lowering functions

`MlirFunctionIdentity` describes package, module, resolved callable, generic
specialization, and overload ordinal as source data. `builder_create_generated_function`
creates an ordinary `MlirFunctionOp`. Its `mlir_function_name` argument is
optional: when absent, `.sev` code generates a stable collision-free name from
the complete identity; when present, the exact source-selected name is used.
The compiler neither generates nor searches for these names.

This is the intended path:

```text
Y -> O -> source-defined lowering F -> generated MlirFunctionOp
  -> MlirOperationState graph -> MlirProgram -> upstream MLIR
```

## Temporary bootstrap boundary

The bootstrap provider in `rust_compiler/transforms/mlir/src/native.rs` constructs a
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

The `__sev_mlir_*` functions in `src/ffi.sev` and `native/provider.rs` are
temporary bootstrap debt, not an accepted permanent escape hatch. They may
transport generic state but may not name or implement a dialect operation.
Their deletion gate is a bootstrapped Severian foreign/native-call layer able
to invoke the upstream MLIR C API from the same source-owned graph. At that
point both files are removed; `MlirProgram` and generated lowering functions do
not change.

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
- complete source-owned `MlirOperationState`, including properties, successors,
  and native result inference;
- source-generated lowering-function names with an optional exact override;
- generic validation and generic terminal printing without operation-name
  dispatch tables;
- ranked tensor types with independently dynamic dimensions;
- structural validation before printing/materialization;
- Universal scalar lowering routed through the typed builder API;
- direct native `ModuleOp` construction and native MLIR verification;
- native source locations and provider-owned verifier diagnostics;
- structural affine maps and the complete attribute families listed above;
- checked context ownership and parser-call instrumentation;
- shared-MLIR linking, avoiding the former whole-archive memory spike.

Still required for the fixed-point production executable path:

1. Replace `StructuredModuleBuilder::print` consumers with the source-owned
   `MlirProgram` materializer.
2. Move the existing CPU/GPU pass pipelines into an in-process MLIR
   `PassManager`.
3. Translate the LLVM dialect to `llvm::Module`, emit an object with
   `TargetMachine`, and link it with embedded LLD.
4. Retain textual tools only behind explicit debug/bootstrap flags.
5. Bootstrap the native-call layer and delete `src/ffi.sev` plus
   `native/provider.rs`; no `__sev_mlir_*` symbol remains in the gold path.

The executable source corpus is `examples/typed_module.sev`. Native construction
tests live beside the bootstrap provider.
