# Current compiler build and block graph audit

Command: `sev_rust build sev_compiler --bin sev_compiler`.

Result: **failed during package resolution**, before source analysis. No compiler executable, block graph, derived submodule, `.o` or `.so` was produced by this attempt. No tests were run.

First blocker: `sev_compiler/package.json` declares `backend` at the missing `sev_compiler/boundaries/backend/package.json`.

## Inventory

- 218 compiler `.sev` files inventoried.
- 99 reached by the literal-import traversal from `sev_compiler/sev_compiler/src/main.sev`.
- 78 non-test/example files outside that traversal.
- 19 missing path-dependency declarations across compiler manifests (10 distinct missing manifest paths).
- 188 unresolved import occurrences encountered in the traversal (including dependencies outside `sev_compiler`).

[All files](files.csv), [files outside static traversal](outside-static-entry-traversal.csv), [missing packages](missing-packages.csv), [missing imports](missing-imports.csv), [build log](build.log).

**Static reachability is not symbol liveness.** Missing packages truncate traversal. Generators, dynamic imports and source providers are not executed by this inventory. Files outside the traversal are review candidates, not proven dead code. No file has measured block-graph consumption because the build never reached that phase. Existing tests/examples are identified separately.

## What contributes today

| Files | Current role |
| --- | --- |
| `syntax/block/block.sev`, `frontend/parser/parser/src/block.sev` | Block structure and inspection. |
| `sev_compiler/src/main.sev`, `driver/driver/src/pipeline/source.sev` | Invoke discovery for `--emit blocks`; normal compilation takes another path. |
| `frontend/lexer/lexer/src/{scanner,token}/mod.sev`, `syntax/primitive/{contract,catalog,declarations,literal}.sev` | Lexical discovery and declared literal providers. |
| `hir/block/dependencies.sev`, `hir/hir/src/lib.sev` | Block dependency API/export; not connected to normal compilation. |
| `hir/module/module.sev` | Stores an initially empty `Program.blocks`; no production population found. |
| `hir/module/hierarchy.sev` | Adds block provenance after resolved body construction. |
| `frontend/modules/graph/{model,queries,verify}.sev`, `mir/mir/src/semantic_graph.sev` | Old definition graph and SCC machinery; not the new block graph. |

There are no production callers of `index_block_file`; its current callers are inline tests. `consume_blocks` likewise has no normal compilation caller. No compiler manifest declares a dependency on `sev-compiler-hir`. Physical inclusion through model imports does not activate these APIs.

## Difference from the requested pipeline

1. Block discovery exists, but normal compilation does not use it to seed the graph.
2. The default block resolver is a no-op; a compiler resolver still needs to register declarations and resolve references from reached bodies.
3. `resolve_block_name` walks lexical scopes, then prelude, then throws. It does not retain outward references for later module/import/package resolution. Binding kinds do not yet implement the proposed precedence tiers.
4. `BlockUnit` already receives package/module/submodule strings. Submodules are not derived from block SCCs.
5. `library/package/src/semantic_dependencies.sev` selects submodules from a source path or manifest mapping before the graph exists.
6. `frontend/modules/graph/queries.sev` computes SCCs only within those preassigned submodules, skipping cross-submodule edges.
7. Artifact selection still consumes that older semantic plan, not `BlockSelection` or block-derived partitions.

## Artifacts

`library/package/src/object_units.sev` plans one object per old semantic submodule. The driver selects functions and produces object MLIR parts, but `object_outputs` carries only text, losing the submodule key at that boundary. The required backend source package is missing, so actual object emission cannot be reached or verified.

`library/package/src/realization.sev` retains object artifacts by target name; shared-library payload handling also exists in `payload.sev`. Neither establishes the requested block-derived submodule artifact identity. A future artifact record needs the stable submodule key, owning module/package, member blocks, exported interface, dependencies and actual artifact paths, including empty/non-emitting units with a reason.

## Partition policy

SCCs identify mutually dependent blocks. They do not by themselves group an acyclic pair `B -> A` into one submodule. The proposed A/B and C/D examples additionally need an ownership/interface or compilation-closure grouping rule; after that grouping, cycles in the unit graph can be collapsed. Without that rule, reporting the desired partitions would be guessing.

## Missing package declarations

- `sev_compiler/abi/package.json`
- `sev_compiler/backend/package.json`
- `sev_compiler/boundaries/backend/package.json`
- `sev_compiler/boundaries/interface/package.json`
- `sev_compiler/interface/package.json`
- `sev_compiler/lir/package.json`
- `sev_compiler/mir/lowering/package.json`
- `sev_compiler/universal/package.json`
- `sev_compiler/xxi/package.json`
- `universal/package.json`

## Files outside static entry traversal

These are candidates for review only, with the limitations above. The CSV includes ownership and role for each.

- `sev_compiler/compile/compile/src/error.sev` — no direct block-graph integration identified
- `sev_compiler/compile/compile/src/lib.sev` — no direct block-graph integration identified
- `sev_compiler/compile/compile/src/model.sev` — no direct block-graph integration identified
- `sev_compiler/compile/compile/src/planner.sev` — no direct block-graph integration identified
- `sev_compiler/compile/compile/src/registry.sev` — no direct block-graph integration identified
- `sev_compiler/driver/driver/src/config.sev` — no direct block-graph integration identified
- `sev_compiler/driver/driver/src/lib.sev` — no direct block-graph integration identified
- `sev_compiler/driver/driver/src/main.sev` — no direct block-graph integration identified
- `sev_compiler/driver/driver/src/pipeline/mod.sev` — no direct block-graph integration identified
- `sev_compiler/driver/driver/src/test_runner/mod.sev` — no direct block-graph integration identified
- `sev_compiler/frontend/modules/graph/model.sev` — legacy definition graph
- `sev_compiler/frontend/modules/graph/queries.sev` — legacy SCC pass
- `sev_compiler/frontend/modules/graph/verify.sev` — legacy graph verification
- `sev_compiler/frontend/modules/modules/src/lib.sev` — no direct block-graph integration identified
- `sev_compiler/frontend/semantic/semantic/src/callable/semantic_ir_tests.sev` — no direct block-graph integration identified
- `sev_compiler/frontend/semantic/semantic/src/package/generic.sev` — no direct block-graph integration identified
- `sev_compiler/frontend/semantic/semantic/src/package/tests.sev` — no direct block-graph integration identified
- `sev_compiler/frontend/semantic/semantic/src/package.sev` — no direct block-graph integration identified
- `sev_compiler/hir/block/dependencies.sev` — dormant block graph API
- `sev_compiler/hir/hir/src/lib.sev` — dormant block graph export
- `sev_compiler/hir/module/hierarchy.sev` — legacy hierarchy bridge
- `sev_compiler/hir/module/module.sev` — empty block graph carrier
- `sev_compiler/lir/lowering/lowering/src/lib.sev` — no direct block-graph integration identified
- `sev_compiler/lir/mlir/build.sev` — no direct block-graph integration identified
- `sev_compiler/lir/mlir/mlir/src/emit/mod.sev` — no direct block-graph integration identified
- `sev_compiler/lir/mlir/mlir/src/ir/tests.sev` — no direct block-graph integration identified
- `sev_compiler/lir/mlir/mlir/src/lib.sev` — no direct block-graph integration identified
- `sev_compiler/lir/mlir/mlir/src/verify.sev` — no direct block-graph integration identified
- `sev_compiler/mir/cfg/availability.sev` — no direct block-graph integration identified
- `sev_compiler/mir/cfg/block.sev` — no direct block-graph integration identified
- `sev_compiler/mir/cfg/dominance.sev` — no direct block-graph integration identified
- `sev_compiler/mir/mir/src/build/mod.sev` — no direct block-graph integration identified
- `sev_compiler/mir/mir/src/lib.sev` — no direct block-graph integration identified
- `sev_compiler/mir/mir/src/passes.sev` — no direct block-graph integration identified
- `sev_compiler/mir/mir/src/verify.sev` — no direct block-graph integration identified
- `sev_compiler/mir/ownership/ownership/src/lib.sev` — no direct block-graph integration identified
- `sev_compiler/mir/ownership/ownership/src/validate/mod.sev` — no direct block-graph integration identified
- `sev_compiler/pipeline.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/constraint/dispatch.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/constraint/graph.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/constraint/proof.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/constraint/refinement.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/function/trait.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/grammar/contracts.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/grammar/grammar.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/array/array.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/array/storage.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/bool/bool.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/bytes/bytes.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/char/char.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/char/encoding.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/char/units.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/char/utf8.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/enum/enum.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/error/error.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/float/float.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/int/int.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/iterator/iterator.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/numeric/conversion.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/numeric/operators.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/numeric/power.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/pointer/pointer.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/primitive.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/string/core.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/string/format.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/string/intrinsics.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/string/methods.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/string/storage.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/string/string.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/string/string_lists.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/tuple/tuple.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/primitive/units/units.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/sentence/syntax.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/type/integer_spelling.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/type/scalar_conversion.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/type/slice_bounds.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/type/storage.sev` — no direct block-graph integration identified
- `sev_compiler/syntax/type/tagged.sev` — no direct block-graph integration identified
