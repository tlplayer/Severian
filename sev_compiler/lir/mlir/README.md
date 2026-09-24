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
        +--> fixed-point compiler handoff
               |
               v
          upstream MLIR
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
the fixed-point handoff. It contains no dialect operation-name signature table;
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
The Rust compiler neither generates nor searches for these names.

This is the intended path:

```text
Y -> O -> source-defined lowering F -> generated MlirFunctionOp
  -> MlirOperationState graph -> MlirProgram -> upstream MLIR
```

## Fixed-point boundary

The former operation/type/array/affine draft provider and the dead public Rust
`OperationState` builder have been removed. They had no production caller and
duplicated construction already owned by `MlirOperationState` and
`MlirProgram`. The remaining private Rust MLIR C bindings support verification
of the current bootstrap artifact path; they expose no operation-construction
surface. No replacement helper-name protocol is permitted. The fixed-point
compiler boundary accepts the completed, validated program graph; it does not
ask Rust to replay or interpret individual builder actions.

## Ownership rules

- `MlirOpBuilder` owns source object identity and SSA numbering.
- A `MlirRegion` is transferred exactly once into its parent operation.
- Text owns nothing and cannot be fed back into production lowering.

## Current checkpoint

Implemented and tested:

- open typed operation construction in `.sev`;
- complete source-owned `MlirOperationState`, including properties, successors,
  and result-inference intent;
- source-generated lowering-function names with an optional exact override;
- generic validation and generic terminal printing without operation-name
  dispatch tables;
- ranked tensor types with independently dynamic dimensions;
- structural validation before printing or fixed-point handoff;
- Universal scalar lowering routed through the typed builder API;
- structural affine maps and the complete attribute families listed above;
- deletion of the unused operation/type/array/affine draft provider surface;
- deletion of the dead Rust `OperationState` interpreter and its construction
  FFI declarations.

Still required for the fixed-point production executable path:

1. Make stage-2 consume the completed source-owned `MlirProgram` directly.
2. Delete the operation/type branches in
   `rust_compiler/transforms/mlir/src/emit/mod.rs` and the remaining
   `StructuredModuleBuilder::print` consumers after that cutover.
3. Move the existing CPU/GPU pass pipelines into an in-process MLIR
   `PassManager`.
4. Translate the LLVM dialect to `llvm::Module`, emit an object with
   `TargetMachine`, and link it with embedded LLD.
5. Retain textual tools only behind explicit debug/bootstrap flags. No
   helper-name lookup or per-operation Rust callback may be introduced while
   completing these steps.

The executable source corpus and fixed-point builder acceptance tests live in
`examples/typed_module.sev`.
