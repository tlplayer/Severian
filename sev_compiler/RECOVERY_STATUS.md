# Compiler recovery implementation — build blocked

The follow-up implementation pass stopped after its ten-minute window, then
attempted one compiler build and ran the two existing inline MLIR binding tests.
The build is blocked and one test fails; recovery is not complete.

Implemented:

- Replaced the duplicate C-oriented driver facade with delegation to the
  existing callable pipeline and an explicit MIR-to-MLIR lowering entry.
- Placed native MLIR tooling and semantic-interface serialization under
  `library/package/compile/{backend,interface}`. The deleted C source emitter
  was not reinstated. Native runtime/provider compilation remains external.
- Redirected compiler manifests and most former `universal` references to
  concrete syntax, HIR, MIR, primitive, and graph owners. Reintroduced required
  leaf contracts and the small MIR CFG builder in their owning directories.
- Moved the explicit prelude provider entry and policy to `library/prelude`.
- Updated archive generation to read concrete model owners and write into the
  current driver/interface package locations. Generation has not been run.
- Added declaration-owned `MlirTypeBinding` and `MlirTypeProvider`. Primitive
  descriptors preserve their bindings through the scalar compatibility view.
  MLIR scalar and aggregate kinds are open data; the type printer substitutes
  structural/type arguments rather than switching on a closed kind enum.
- Retained class/trait decorators during parsing, preserved class metadata
  during specialization, and connected class `@mlir("...")` bindings to
  callable type lowering. Structural fields and generic arguments are passed
  to the binding. Trait annotation inheritance now resolves before lowering,
  substitutes generic arguments, and rejects conflicting inherited bindings.
- Preserved submodule keys in `MlirObject` through native emission. Named object
  copies have bounded filenames and retain their complete key for collision
  checking. Their membership still comes from the existing semantic plan.
- Existing inline parser, binding, and printing regression cases remain beside
  their implementations. This follow-up added no tests.
- Moved index and memref storage spelling into primitive MLIR binding metadata.
  Lowering consumes that metadata; the printer retains resolved parameters and
  supplies structural dimensions, elements, and rank.
- Replaced the unresolved tuple grammar service with declaration-owned
  constructor metadata consumed by generic delimited-expression parsing.
- Restored the optional native quality provider under
  `library/package/compile/backend/runtime` and updated the adapter path.
- Removed production manifest cycles, moved semantic test-only module usage to
  dev dependencies, and moved callable identity helpers out of the symbol leaf.
  Dev dependencies still form a cycle that blocks the bootstrap resolver.
- Updated archive generation to resolve former qualified aliases to concrete
  model owners and import only owners used by each generated archive chunk.

Validation on 2026-09-24:

- `sev_rust build sev_compiler --bin sev_compiler` failed during dependency
  resolution: `interface -> semantic -> modules -> interface`. The resolver
  includes the semantic package's dev dependency on modules. No compiler
  diagnostics or generated archive validation were reached.
- `sev_rust test sev_compiler/syntax/type/mlir.sev` ran exactly two existing
  tests: the open-dialect/resolved-parameter case passed; the unresolved-parameter
  rejection case failed with `error: unresolved MLIR type parameter: T` escaping
  the test's expected catch. The cause remains unresolved.
- Both commands used `/usr/bin` first on PATH to select the system Python.
  Logs: `/tmp/sev-recovery-build.log` and `/tmp/sev-recovery-unit-tests.log`.
- Source diff whitespace checks passed. A production-only manifest traversal
  found no cycles; it did not cover the dev dependency cycle above.

Remaining before claiming recovery complete:

1. Complete the source-contract migration and inspect compile diagnostics.
   Dormant custom-compilation code still references the removed `TypeContext`;
   the full HIR/MIR carrier split is unfinished. Existing malformed/stubbed
   source also remains, including the legacy test-runner file.
2. Remove the remaining dev dependency cycle without losing the semantic package
   integration coverage, then inspect actual compiler diagnostics.
3. Review generated archive schemas. The updated generator and recovered source
   compiler have not been validated by a successful build.
4. Resolve the failing inline rejection test and validate declaration-driven
   storage/ABI lowering and trait inheritance after bootstrap recovery.
5. Connect production block indexing and resolution, outward references, and
   SCC partitioning. The block API still does not determine normal compilation
   boundaries. `.so` grouping and the consumed/unconsumed block report remain.

After the dependency cycle is addressed, retry
`sev_rust build sev_compiler --bin sev_compiler`. The focused test command is
`sev_rust test sev_compiler/syntax/type/mlir.sev`; the full suite remains unrun.
