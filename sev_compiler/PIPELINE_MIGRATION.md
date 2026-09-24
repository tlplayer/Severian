# Syntax, primitives, CFG and MIR ownership migration

## September 24 continuation

Seven-minute code-only continuation; no builds, tests, or test additions.

- Replaced scalar-address-only definite initialization with structural storage
  places. MIR now follows aliases, field projections and incoming block arguments;
  whole-place writes initialize descendants, while writes through ambiguous
  pointers do not initialize every possible owner. Scalar projection reads use
  predecessor intersection and loop fixed points. Aggregate completeness and
  active-variant checking remain incomplete.
- Added hierarchy traversal for expression-owned blocks, match arms, lambda
  bodies and nested expression operands. Inline-only callable expansions receive
  independent layouts. Nested expression lowering restores source origins, and
  block-result expressions retain their body's lexical origin. Capture identity
  and shared declaration ownership still require further review.
- Added class-method dispatch expansion before receiver registration. Runtime
  guards share identical progressive prefixes with explicit conditional
  evaluation; uniqueness is still checked before calling any implementation.
  This extends the existing scalar predicate subset, not a completed general
  trait-interface constraint solver. Inherited interface prefixes, general
  predicates, static overlap proofs and reusable dispatch graph serialization
  remain required.
- Changed cleanup record lookup to structural registration instead of subtracting
  1000. MIR bool/unit/char decisions now use declaration-owned identities at the
  edited sites. Rust catalog generation and legacy buffer archive encoding are
  unchanged.
- Repaired malformed pass-manager declarations and calls; removed the superseded
  scalar-only initialization helpers after wiring their replacement.

HIR memory checks remain necessary: MIR still lacks complete move/loan events
and cleanup obligations. The HIR/MIR carrier split and arbitrary sentence
declarations remain unfinished. All edits in this continuation are uncompiled.

Status: implementation in progress; stopped within the requested time limit. No build
or tests were run. No tests were added. This is not a claim that the compiler
builds or that the full architecture migration is complete.

## Implemented

- Added `syntax` as a dependency-free package. It owns source identity and spans,
  lexical and token contracts, compiler identity contracts, type annotations,
  literal payloads, operator descriptors, sentence composition and source block
  provenance. Source loading now imports source coordinates from syntax.
- Added `primitives`. Concrete primitive descriptor classes own their existing
  numeric identities, stable declaration identities, representations and literal
  defaults. The `Primitive` trait requires identity and literal metadata. The old
  scalar catalog is a compatibility projection of those descriptors; its
  separate `ScalarIdentities` enum and duplicated descriptor branches are gone.
- Routed numeric recognition and normalization through primitives. Lexer tokens
  retain primitive literal construction metadata, original lexemes and spans;
  parser numeric expressions retain that metadata for semantic type selection.
  Hexadecimal normalization uses decimal digits instead of a signed host integer.
- Replaced production `universal.TypeId(<known primitive number>)` construction
  sites with declaration-owned accessors. Existing test bodies were left alone.
- Updated the canonical vocabulary: `S` means Sentence; use `Shape` explicitly.
- Moved ownership implementations to `transforms/mir/ownership`. Production
  package dependencies point there. Old frontend ownership files are forwarding
  imports. The old `mlir.pipeline` path is a compatibility symlink; the Rust
  backend and Severian backend now read the MIR-owned pipeline.
- Extracted executable CFG builder state and mutation functions into
  `transforms/cfg`. MIR imports that builder instead of a universal builder.
- Added source block layouts with identities, parents, children, kinds, callable
  owners and spans. Parser body spans survive common typed-body reconstruction
  paths. Layouts are assigned before executable lowering, rather than rebuilt
  afterward by the dependency graph projection.
- Added operation origins independent of debug output. CFG insertion and storage
  promotion preserve parallel operation/origin tables. Origins include lexical
  block identity, source span, callable identity and compiler callable paths.
  Dependency edges and Agent IR operation output now retain origin information.
  Agent IR also exposes block kinds, parents, children, owners and source spans.
- Added MIR scalar-slot definite-initialization analysis over CFG predecessor
  intersections and loop fixed points. MIR checks stack escape/alias information
  and attaches storage and callable locations using existing diagnostic records.
  Ownership authorization is repeated after storage promotion and by lowering.
- Removed the older driver-level HIR ownership-validation calls; the check entry
  now invokes MIR construction. Existing semantic effect/loan/place checks remain
  until their complete MIR replacement exists.
- Updated frontend archive generation to discover syntax-owned and re-exported
  source contracts. The generator has not been executed.

Diagnostics have not been relocated. No diagnostic package dependency split or
rendering redesign was made. Existing diagnostic records carry the new context.

## Still required

1. Finish phase separation. Semantic analysis still calls effect/loan/place
   helpers and elaborates some cleanup. Moving their files under MIR does not
   move every decision to the MIR phase. A complete MIR memory-event/place model
   must preserve moves, loans, projection initialization and cleanup obligations
   before those HIR checks can be removed.
2. Finish primitive migration. Numeric `.index` comparisons and reserved ranges
   still exist; legacy buffers still use the `+ 200` archive encoding. Remove
   these through structural type registration and an archive compatibility plan.
   The Rust primitive catalog has not been generated from the new declarations.
   Conversion edges and behavior implementations still need a single authority.
3. Complete block coverage. Expression-owned blocks, lambda/capture scopes,
   specializations and every generated body need consistent source ownership.
   The new layout covers common statement bodies and callable/class/trait/test/
   global roots. Review shared source bodies and cloned declarations carefully.
4. Finish the HIR/MIR model split. `universal.Module` still carries HIR declarations
   and CFG bodies, and the CFG carrier types are still in universal. The new
   builder package is separate; the carrier/serialization split remains.
5. Integrate sentence declarations fully with parsing and HIR construction. The
   existing sentence composition implementation moved into syntax, but arbitrary
   `sentence [...]` declarations are not newly supported by this change. Replacing
   expression/declaration/statement wrappers with the agreed callable/operation/
   sentence contracts remains work.
6. Finish provenance coverage and presentation. Agent IR includes operation
   origins, dependency origin records and block layouts/spans; all generated and
   expression-owned bodies still need to be covered consistently. A dependency edge currently retains
   its first invocation origin. Compiler callable paths are not runtime stacks.
7. Review and compile the changed package graph, archive generator and bootstrap
   integration. No build has validated new contracts or constructor syntax.
8. Audit tests separately when authorized. No tests were edited or added for this
   migration, and no test command was run.

## Purge candidates

Do not delete compatibility entries until their remaining import/archive users
have been migrated and the replacement is known to work.

| Candidate | Replacement / prerequisite |
| --- | --- |
| Forwarding files under `universal/id`, `universal/type`, `universal/literal`, `universal/operator` | Syntax and primitive package APIs; migrate remaining import paths first. |
| `frontend/sentence/src/lib.sev` and its package | `syntax/sentence.sev`; redirect parser and package clients before deletion. |
| `frontend/ownership` forwarding package and pipeline symlink | `transforms/mir/ownership`; migrate direct source and tooling/test path users first. |
| `universal/literal/literal.sev` | Incomplete bool-only literal enum; syntax literal payload and primitive construction contracts cover the intended replacement, but audit public callers first. |
| `universal/sentence/sentence.sev` | Unintegrated sentence sketch; canonical executable composition now lives in syntax. |
| `FunctionDeclaration.cfg`, `Module.initializer_cfg` | Existing comments identify seed ABI sentinels. Require coordinated bootstrap/archive migration; not safe to delete just because unused. |
| `Block.operations` / `Block.lowered_operations` | Separate HIR and MIR carriers; migrate legacy lowering/backend users first. |
| `primitives/catalog.sev` scalar adapter and legacy buffer offsets | Direct descriptor/structural-type consumers plus archive migration. |
| `transforms/mir/ownership/src/validate/mod.sev` | Older HIR declaration-availability walk; audit remaining direct callers before deletion. |
| Old semantic ownership state/checks | Complete MIR move/loan/place and cleanup implementation, not just relocation. |
| `universal/tests/scalar_lookup.sev` | Its expected catalog is stale. Preserve relevant lookup/identity behavior when later revising it. |
| Success-path `Unimplemented` throws in primitive tests | Manual behavior audit; do not mechanically delete the tests. |

## Pre-existing source issues observed

These were present before this edit and have not been repaired:

- `universal/src/lib.sev` imports `universal/source/assertion_origin.sev` and
  `universal/source/assertion.sev`, which are absent from this checkout.
- `universal/module/module.sev` imports the absent
  `universal/source/quality.sev`.
- `transforms/mir/src/passes.sev` contains malformed `PassError`/`PassManager`
  declarations with inserted `throw Error(...)` text inside identifiers.
- Several existing compiler branches still throw `Unimplemented` for normal
  operations. In particular, CFG optimization paths need a separate review.
- `sev_compiler/target/host/dev/bin/sev_compiler` was already deleted in the working
  tree before this task. That user change was left untouched.

## Build command

From the repository root:

```sh
sev_rust build sev_compiler --bin sev_compiler
```

This command has not been run. The pre-existing source issues above may prevent
it from reaching the new code.

## Exact stopping point

The last work was a source review of lexer normalization/metadata ordering,
archive namespace discovery and operation/block provenance preservation. The
scanner now supplies normalized values before constructing literal metadata.
Agent IR now emits the retained block layouts. The next step is compiler-level
validation and completing the remaining HIR-to-MIR ownership/event migration;
this work has not established that the current tree builds.
