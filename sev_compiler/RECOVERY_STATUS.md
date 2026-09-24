# Compiler recovery implementation — build blocked

The latest six-minute recovery pass moved the quality provider into
`library/package/diagnostic/runtime/quality.c`, repaired bootstrap paths, and
rebuilt the Rust seed successfully. The source compiler remains build blocked;
recovery is not complete.

Latest implementation and validation (2026-09-24):

- Copied the quality provider before removing the old backend copy, verified
  byte equality, and updated native linking and build-input hashing together.
  Also repaired the source driver's ownership-pipeline input path.
- Removed an unused semantic test import and its `modules` dev dependency.
  Both existing tests remain; the earlier manifest dependency cycle is gone.
- Declared `MlirTypeBinding.render` as returning `string | Error`. Both existing
  inline tests now pass, including unresolved-parameter rejection.
- Updated Rust bootstrap cache roots, the backend's embedded ownership pipeline,
  and both prelude packaging tools to their current source locations.
- Changed numeric literal recognition to import its local primitive contract.
- Extended the Rust parser to accept field assignments through indexed receivers
  such as `blocks[index].kind = value`, preserving the receiver expression.
  Added one inline Rust regression test in the same implementation file; that
  new test has not been run.
- `cargo build -p severian-driver --bin sev` succeeded. The rebuilt debug seed
  built `docs/examples/00-getting-started/01-hello.sev`. The installed seed passed
  the hello and variables example tests (one each). The rebuilt debug seed also
  passed both inline MLIR binding tests.
- `PATH=/usr/bin:$PATH package.pkg/debug/sev build sev_compiler --bin sev_compiler`
  now stops at E000128: a cross-package source import cycle between `syntax` and
  `primitives`, reported at `syntax/function/function.sev:11`. The primitive
  contract imports syntax models that eventually import the primitive contract.
  This requires an ownership/dependency repair, not an exception to cycle checks.
- Logs: `/tmp/sev-recovery-{seed,build,example-build,bindings,hello,variables}-current.log`.
  Source whitespace checks pass. The moved native provider has not yet been
  exercised through a successfully rebuilt source compiler.

To run the new parser regression:
`cargo test -p severian-parser indexed_receivers_support_field_assignment_and_update`.
The `sev_rust` launcher prefers the existing release binary; use
`package.pkg/debug/sev` explicitly to exercise the rebuilt debug seed.

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
- Restored the optional native quality provider, now owned by
  `library/package/diagnostic/runtime`, and updated the adapter path.
- Removed production manifest cycles, moved semantic test-only module usage to
  dev dependencies, and moved callable identity helpers out of the symbol leaf.
  The latest pass removed the unused dev dependency that blocked resolution.
- Updated archive generation to resolve former qualified aliases to concrete
  model owners and import only owners used by each generated archive chunk.

Previous pass validation on 2026-09-24 (superseded above):

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
2. Repair the cross-package source import cycle between syntax and primitives
   by separating shared contracts from implementation dependencies.
3. Review generated archive schemas. The updated generator and recovered source
   compiler have not been validated by a successful build.
4. Validate declaration-driven storage/ABI lowering and trait inheritance after
   bootstrap recovery. Both inline MLIR binding tests now pass.
5. Connect production block indexing and resolution, outward references, and
   SCC partitioning. The block API still does not determine normal compilation
   boundaries. `.so` grouping and the consumed/unconsumed block report remain.

After the dependency cycle is addressed, retry
`sev_rust build sev_compiler --bin sev_compiler`. The focused test command is
`sev_rust test sev_compiler/syntax/type/mlir.sev`; the full suite remains unrun.
