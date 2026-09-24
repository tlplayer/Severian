# Compiler recovery implementation — unbuilt

This pass stopped at the requested time limit without running a compiler,
generator, build, or test. Source imports, manifests, and diffs were inspected.
It is an implementation checkpoint, not a claim that the compiler builds.

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
  to the binding. Trait annotation inheritance is not yet implemented.
- Preserved submodule keys in `MlirObject` through native emission. Named object
  copies have bounded filenames and retain their complete key for collision
  checking. Their membership still comes from the existing semantic plan.
- Added inline parser, binding, and printing regression cases; none were run.

Remaining before claiming recovery complete:

1. Complete the source-contract migration and inspect compile diagnostics.
   Dormant custom-compilation code still references the removed `TypeContext`;
   the full HIR/MIR carrier split is unfinished. Existing malformed/stubbed
   source also remains, including the legacy test-runner file.
2. Replace the unresolved tuple grammar service import in
   `syntax/primitive/tuple/tuple.sev`. The external native adapter's optional
   quality-runtime provider also needs reconciliation with the retained runtime.
3. Review generated archive schemas and package dependency cycles. The new
   generator and source compiler have not been executed.
4. Finish declaration-driven storage/ABI lowering. `callable_storage_type`
   still contains host memref/index layout policy; this is not covered by the
   new generic type-spelling renderer.
5. Connect production block indexing and resolution, outward references, and
   SCC partitioning. The block API still does not determine normal compilation
   boundaries. `.so` grouping and the consumed/unconsumed block report remain.

After these implementation gaps are addressed, the first bootstrap command is
`sev_rust build sev_compiler --bin sev_compiler`. Tests remain for the user to
run, using `cd sev_compiler && sev test` after bootstrap recovery.
