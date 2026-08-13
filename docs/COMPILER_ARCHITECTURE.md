# Compiler architecture

Severian uses traits at extension points and strong types at correctness
boundaries. A compiler-wide trait hierarchy is not the goal. The intended
pipeline is:

```text
AST -> resolved HIR -> owned HIR -> MIR -> optimized MIR -> target IR -> artifact
```

Each arrow must eventually consume the concrete output type of the preceding
stage, so an invalid stage order cannot be assembled accidentally.

## Current invariant boundary

Stable identity types now exist for functions, type definitions, variants,
bindings, fields, methods, modules, packages, symbols, and intrinsics. Names
remain human-readable metadata. Resolved binding definitions and uses carry a
`BindingRef { id, name }`; ownership, dataflow, diagnostics, MIR locals, native
lowering, and StableHLO lowering key local state by `BindingId`. Package linking
namespaces binding IDs before merging modules. Pre-resolution semantic scopes
remain name-keyed because resolving a source spelling is their purpose.

HIR is verified after semantic resolution, package linking, and every HIR
transformation. MIR verifies dense block identity, valid successor targets,
complete terminators, Boolean control-flow conditions, and unique function
identity before target lowering. Return types remain a HIR invariant until MIR
represents result wrapping and reachability without its HIR sidecar.
`sev build --verify-each` exposes
those successful boundaries in development logs.

## Migration order

1. Move post-resolution bindings, fields, methods, modules, packages, symbols,
   and intrinsics from string identity to their stable IDs.
2. Make MIR self-contained with locals, places, operands, rvalues, explicit
   storage lifetime, moves, borrows, and drops.
3. Move ownership and initialization checking to generic MIR dataflow.
4. Separate cached analyses from transformations and invalidate analysis
   results explicitly.
5. Add visitors and rewriters per IR rather than one universal traversal trait.
6. Replace runtime-symbol tests with typed intrinsic IDs and effect summaries.
7. Put native, XLA, WebAssembly, IREE, and kernel code generation behind the
   backend extension point; inject host access for deterministic testing.
8. Converge `ValueType` and `TypeId` through the interned type context.
9. Introduce memoized compiler queries only after the preceding identities and
   immutable boundaries are dependable.

The binding-identity vertical slice is complete. The next implementation step
is removing the HIR sidecar from MIR: expressions must become MIR operands and
rvalues, assignment targets must become places, and storage/move/borrow/drop
lifetime must be explicit before ownership analysis moves onto the CFG.
