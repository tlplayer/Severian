# SIP-0008 review: semantic graph V2 implementation and retirement

Reviewed 2026-09-22 against checkout `1614db3ae9d9f681cd744788c3e66745893530d0` and
the [hierarchy proposal](0008-Complier-Structure.md). Paths below are repository
relative. **This is a proposed implementation plan, not a claim that V2 is
implemented. No compiler code is deleted by this review.**

The architecture is feasible if Universal owns typed semantic identities and
relationships while retaining explicit stages, verification, and the existing
canonical executable CFG. The largest missing pieces are stable cross-build
identity, semantic submodules, typed `with` obligations, safe predicate dispatch,
and serialized interfaces sufficient for downstream specialization.

“V2” here means this proposal applied to `sev_compiler` and its package libraries.
Retiring the Rust seed is a separate self-hosting milestone. Follow the repository
rule: **implement and validate the replacement before removing its predecessor.**

## 1. What the checkout actually contains

| Finding | Evidence and consequence |
| --- | --- |
| Executable CFG already exists | `sev_compiler/universal/cfg/{cfg,block,branch}.sev`, `universal/module/module.sev`, and `transforms/mir/src/callable.sev` store bodies in `Module.cfg_bodies`. Extend these contracts; do not introduce a second executable topology. |
| MLIR and Agent IR use that path | `sev_compiler/boundaries/driver/src/pipeline/source.sev` calls `mir.build_callables`, then either `agent_ir.emit` or `mlir.lower_callables`. Both inspect the lowered module. Agent IR still needs graph/identity completeness checks. |
| Universal is shared models, not yet the proposed graph database | `sev_compiler/universal/module/module.sev` has items, functions, types and CFG tables. There is no integrated Library/Submodule/ObjectUnit/constraint graph matching this SIP. |
| IDs are not yet stable under all proposed moves | `sev_compiler/frontend/semantic/semantic/src/definitions.sev::definition_id` uses declaration ordinals; trait registration also uses source index and span start. Merely having `DefId` does not provide file-independent persistent identity. |
| Sentence matching is a partial foundation | `sev_compiler/frontend/sentence/src/lib.sev` has structural filtering and exactly-one matching, but constraint references are strings. Its README explicitly leaves sentence declaration parsing and semantic lowering as integration work. `universal/sentence/sentence.sev` is illustrative syntax, not proof of an active pipeline. |
| `with` has several existing representations | Generic and field constraints exist in `universal/declaration/{language,function}.sev`; test policies and grammar metadata have other owners. These must converge through typed attachment contracts without changing each construct's meaning. |
| Scalar descriptor lowering is still active | `frontend/semantic/src/grammar_scalar.sev`, `universal/operator/scalar.sev`, and `transforms/mlir/src/emit/callable.sev` carry `ScalarOperation`. The retirement regex in `tests/sev_compiler/migration.py` does not cover this whole path. |
| Source-shaped package interfaces remain | `frontend/modules/src/source.sev::read_source` reconstructs `.sevi` declaration contracts as source text. `boundaries/driver/src/pipeline/package_interface.sev::scalar_interface` and `library/package/src/interface_metadata.sev` emit scalar-oriented contracts. This is not yet the typed semantic interface proposed here. |
| Historical lowering paths remain alongside the active path | `boundaries/driver/src/pipeline/mod.sev`, `compile/src/planner.sev`, `transforms/lowering/src/lib.sev`, and `transforms/mlir/src/emit/mod.sev` refer to older operations/block storage. Audit callers and preserve their compiler-provider/backend functionality before retirement. |
| Two empty fields are documented seed ABI workarounds | `Module.initializer_cfg` and `FunctionDeclaration.cfg`, documented in `universal/cfg/README.md`, cannot safely be removed solely because no executable consumer reads them. |
| Migration documentation is stale in places | `tests/sev_compiler/migration/RESULTS.md` is dated September 8 and says CFG inspection is missing. Current source and `INVENTORY.md` describe later work. Do not report that old 16/30 tally as current results. |

## 2. Corrections needed before accepting the SIP

### Blocking semantic issues

1. **A dependency graph is not necessarily a DAG.** Recursive functions,
   mutually dependent declarations and loop-carried values produce cycles.
   Compute strongly connected components (SCCs) for scheduling; their condensation
   graph is a DAG. Distinguish legal recursion, unsupported import cycles, cyclic
   initialization, and invalid predicate-prerequisite cycles. Reject only the
   cycles forbidden by the relevant contract. CFG and loop dataflow stay cyclic.

2. **Cheap-first dispatch cannot stop at its first matching implementation.**
   In the example, `rank == 2` and `expensive_check(x)` may both be true. Selecting
   the first would contradict the ambiguity rule and SIP-0007. Select only once
   all other candidates are eliminated or proven disjoint; otherwise evaluate
   enough remaining predicates to establish uniqueness. Cost orders legal work,
   never candidate precedence. Diagnose proven ambiguity statically; detect
   value-dependent ambiguity at runtime unless the language rejects unresolved
   overlap conservatively. Define no-match behavior as well.

3. **Arbitrary predicates cannot safely be reordered or shared.** A test can
   mutate, throw, diverge, perform IO, read volatile state, or observe concurrent
   mutation. Reordering or memoizing it can change behavior. Initial dispatch
   predicates should be an effect-checked, non-throwing, terminating subset with
   stable reads; classify unproven calls as unsafe for reordering. Other `with`
   operations retain declared order and control dependence. Cost annotations
   cannot grant purity or termination. Resource budgets on compile-time execution
   diagnose exhaustion; they do not prove a runtime predicate terminates.

4. **The proposed `defer = fix` alias conflicts with implemented syntax.**
   `Statement.Defer`, the parser's “defer requires a call” diagnostic, and
   `tests/sev_compiler/deferred_cleanup.py` define delayed cleanup on scope exits.
   Preserve that behavior. Prefer `fix` alone for invariants. A context-specific
   `with defer` meaning would require an explicit language decision and cannot
   silently reuse ordinary `defer` semantics.

5. **The contract example does not maintain its own invariant.** With `x = 95`,
   entry `x > 0` and initial `fix x < 100` hold; `x += 10` breaks the invariant.
   For integer `x`, use entry `0 < x and x < 90` if the example is intended to
   succeed, and state overflow behavior. Specify when `fix` is established and
   rechecked: entry, writes through aliases, calls that may mutate, loop edges,
   exits, and suspension/re-entry where applicable. “Throughout” is not a
   realizable promise of continuously polling arbitrary shared memory.

6. **`Static/Specialization/Runtime` and `Entry/Fix/Exit` are different axes.**
   Use an evaluation stage and a check site/policy. An exit obligation can be
   discharged statically or checked at runtime. Define normal return, error
   propagation, break/continue crossing the owning region, cancellation, and
   cleanup ordering separately. Do not promise checks on process abort. Also
   separate dispatch guards from contracts: a failed body invariant is not a
   request to try a different implementation after side effects occurred.

7. **General automatic proof and overlap detection are not achievable.**
   Arbitrary user code does not admit a complete terminating solver for
   implication, predicate overlap, invariants, or exact execution cost. Use
   `Proven / Refuted / Unknown`, a bounded supported predicate language, and
   conservative behavior for Unknown. Runtime checks can validate executions;
   they cannot prove all future executions. Complexity and selectivity are hints,
   not correctness evidence. Define whether Unknown requires a runtime check,
   rejects a static-only requirement, or requires an explicit dispatch policy.

### Structural and analysis issues

8. **Lexical regions and CFG basic blocks are different.** One source block can
   lower to several basic blocks; one basic block can carry locations from several
   sentences. Class/trait bodies are declaration scopes, not necessarily runtime
   execution regions. Use distinct `RegionId` and `BlockId`, with explicit links.
   Async work also creates a task execution/lifetime region even if its call
   sentence has no nested lexical block.

9. **The hierarchy must admit recursion and references.** The draft's linear
   containment test omits its own `Sentence → Block → Sentence` example and omits
   declarations/callables. Define allowed owner kinds and cardinalities. Symbol
   uses reference a declaration identity; the declaration must not acquire a new
   containment owner for every use. `Node.id: Symbol` also conflicts with the
   repository vocabulary: use a typed node ID, and reserve Symbol/`Y` for resolved
   declaration identity. Values, literals, types, and syntax uses remain distinct.

10. **A projection is usually a query, not merely an edge filter.** Predicate
    scheduling needs data prerequisites, guard facts, effect dependencies, and
    candidate membership as well as `Requires`. Artifact planning needs targets,
    substitutions and linkage. Define each projection's input relations,
    verification rules and invalidation dependencies. Retain structured syntax
    for diagnostics, macros and grammar resolution; graph unification does not
    eliminate the unresolved-to-resolved-to-lowered stage distinctions.

11. **Refinement belongs to a value version at a program point.** `x > 0` cannot
    permanently refine a mutable symbol after assignment or an aliasing call.
    Attach facts to SSA values or memory versions and dominating edges, kill them
    when invalidated, and merge by a sound join. Widening is for convergence at
    loops when needed, not the ordinary operation at every branch join. The
    Positive/NonPositive complement example is valid for integers; NaN makes it
    invalid for unrestricted floating-point comparisons.

12. **Several example predicates need repair.** `sorted(x)` returns a sorted
    sequence in the canonical prelude; it is not an `is_sorted(x)` boolean test.
    `rank(x)` being defined does not prove `x.shape[1]` is valid: require rank at
    least 2, and prove `y` has rank at least 1 before `y.shape[0]`. Define
    `unchanged(y)` as identity, shallow state or deep reachable state, including
    snapshots and alias/concurrency rules. A local `view` alone is not a proof
    that no other alias changes the observed storage.

### Packaging and performance qualifications

13. **File-independent identity needs explicit naming rules.** Specify how
    files declare/join a semantic submodule, how top-level declarations are
    assigned, overload disambiguators, private/anonymous identities, and ordered
    initialization. Start with one file mapped to one submodule for compatibility,
    then support explicit multi-file membership. Moving a declaration preserves
    identity only if its semantic owner and declaration key stay the same; source
    maps and debug artifacts may still need rebuilding.

14. **An unchanged interface does not always spare every consumer.** Inlining,
    generics, constant evaluation, macros, imported grammar, specialization and
    LTO can depend on implementation bodies. Track actual query dependencies,
    including the candidate set of a dispatch family; adding an implementation
    can introduce ambiguity even if old function signatures stay unchanged.
    Namespace hashes by schema/compiler, target/ABI where relevant, features,
    optimization/overflow policies and dependency identities. Preserve portable
    paths in existing package caches.

15. **`.sevi + objects` only suffices if the interface contains all needed
    semantics.** Downstream generic instantiation needs a typed body or an
    available compatible realization; imported grammar/macros need their compiler
    semantics. Predicates need executable implementations or serialized supported
    expressions. Export visibility, effects, ownership, substitutions and
    constraints, and reject unsupported source-free cases explicitly. Do not
    label scalar declaration stubs complete semantic serialization.

16. **ObjectUnit is useful, but native objects are one realization kind.**
    Support initializers, duplicate specialization ownership, symbol visibility,
    linker reachability and deterministic splitting/merging. Keep target-specific
    plan data outside target-independent semantic identity. CPU objects, device
    code, JIT outputs and compile-provider artifacts need tagged realization
    kinds. Planning can precede MIR/MLIR; materialization follows lowering. Graph
    storage and extra analyses do not automatically make compilation faster:
    measure time, peak memory, cache hits and emitted code before claiming gains.

Editorial follow-up: the main file is numbered 0008 but titled SIP-0000, and its
Markdown headings are bold literal `##` text. Normalize those when adopting the
revised SIP; update SIP-0007's overlap policy at the same time.

## 3. Concrete V2 contract to implement

Use typed arenas/tables and typed references, not `list[Node]` containing arbitrary
objects. Existing concrete contracts remain meaningful; `X` is a vocabulary role,
not a reason to erase them.

| Contract | Required information |
| --- | --- |
| Identity | Package/module/submodule semantic keys; persistent declaration key; revision-local typed indices; a mapping between the two. Distinguish identity from content fingerprint. |
| Provenance | Source ID and span, grammar definition, containing sentence/region, and synthetic-origin reason. Establish one span coordinate contract with explicit conversions; sentence docs currently mention Unicode scalars while migration IR specifies UTF-8 byte offsets. |
| Containment | One semantic owner per owned entity; acyclic owner edges; ordered sentence/member lists; symbol-use references separate from ownership. |
| With attachment | Resolved interface identity and typed payload: declaration/setup, guard, contract, ownership requirement or execution context. Preserve attached syntax until resolution; `list[Constraint]` cannot represent all these forms. |
| Constraint | Typed predicate, value inputs, prerequisite facts, effect/read summary, stage, check site, failure behavior, origin, optional cost/selectivity estimates and proof status. |
| Refinement | Domain, value/memory version, establishing program point, invalidation dependencies, join and optional widening operations. |
| Dispatch plan | Closed candidate-set revision, substitutions, shared safe checks, guard-dependent edges, exact-one-match/no-match/ambiguity outcomes. |
| Executable state | Existing `CfgBody`/`BasicBlock` and typed operations/terminators, plus region links, explicit exceptional/cleanup edges and verified capabilities. |
| Object plan | ObjectUnit ID, realization kind, target/ABI/options, member realizations, required symbols, initializer order, fingerprints and artifacts. |
| Graph revision | Append/update through checked builders; freeze before passes; derived facts carry revision/dependency stamps. No independently mutable duplicate facts in graph edges and model fields. |

Keep MIR as a named lowered stage/view with explicit legal operations and a
verifier. It may share Universal storage; it must not become an unverified alias
for every node kind. Keep MLIR as the backend IR and preserve source origin maps.

## 4. Ordered implementation plan and files

Paths marked **add** are proposed new files, not existing implementations. Brace
lists enumerate files. Extend the relevant `package.json` dependencies and
`src/lib.sev` exports when adding modules; regenerate lockfiles through the normal
package tooling, rather than hand-editing resolution data.

### P0 — Freeze a reproducible baseline and normative decisions

**Edit:** `docs/sip/0008-Complier-Structure.md`,
`docs/sip/0007-interface-with-dispatch.md`, `sev_compiler/GRAMMAR.md`,
`sev_compiler/README.md`, `tests/sev_compiler/migration/{README,INVENTORY,RESULTS}.md`.

Adopt the corrections above, especially predicate admissibility, runtime overlap,
failure behavior, `fix` check points, membership and initialization order. Record
the source revision, compiler hash, sysroot and tools with fresh baseline results.
Add a capability ledger for all public entry points, including library/provider
APIs reached through the older driver, so cleanup cannot silently lose features.

**Gate:** every supported behavior has a test owner; stale results remain labeled
historical. Existing failures are recorded separately from V2 acceptance failures.

### P1 — Typed identity, hierarchy and graph verification

**Edit:** `sev_compiler/universal/id/compiler.sev`,
`universal/module/module.sev`, `universal/declaration/{language,function}.sev`,
`universal/src/lib.sev`, `frontend/semantic/src/definitions.sev`,
`frontend/modules/src/source.sev`, `frontend/source/source.sev` (all under
`sev_compiler/`); `library/package/build/src/model.sev` and
`library/package/src/{manifest,resolve}.sev` for explicit membership metadata.

**Add:** `sev_compiler/universal/graph/{model,verify,queries}.sev` and
`sev_compiler/universal/module/hierarchy.sev`.

Implement stable definition keys before caching them. Introduce Library,
Submodule, source memberships, Region and Sentence records; retain the existing
module facade while consumers migrate. Provide one checked graph construction
API, endpoint-kind/cardinality validation, duplicate-ID checks, owner-cycle
diagnostics and SCC scheduling. Use adapters temporarily, with one authoritative
owner for every fact. Specify cross-revision anonymous-ID matching conservatively.

**Gate:** recursion works; illegal initialization/owner cycles diagnose; moving
files or changing import discovery order preserves exported semantic keys;
duplicate overload keys diagnose; non-ASCII source provenance round-trips.

### P2 — Integrate sentences and typed `with` attachments

**Edit:** `sev_compiler/frontend/sentence/src/lib.sev`,
`frontend/parser/src/statement/mod.sev`, `frontend/lexer/src/syntax/mod.sev`,
`frontend/semantic/src/{definitions,callable,test_modes}.sev`,
`universal/{sentence/sentence,statement/statement,expression/expression}.sev`,
`universal/grammar/{grammar,contracts}.sev`,
`universal/declaration/{language,function}.sev`.

**Add:** `sev_compiler/universal/constraint/model.sev` and
`sev_compiler/frontend/semantic/semantic/src/with.sev`.

Parse an unresolved attachment once, then resolve its interface and typed payload
in the owning grammar. Replace string-only sentence constraint identities with
resolved requirements. Keep setup declarations, context/ownership attachments,
dispatch guards and contracts distinct. Retain legacy syntax adapters until every
existing producer/consumer has moved. Preserve declaration bootstrap and CFG
capability enforcement; an extensible interface does not grant arbitrary effects.

**Gate:** all SIP sentence examples have the expected lexical nesting and source
origins; grammar edits change behavior with an unchanged executable; invalid
attachments diagnose; ordinary `defer` and test policies retain their semantics.

### P3 — Constraint solving and ambiguity-preserving dispatch

**Edit:** `sev_compiler/frontend/semantic/semantic/src/{callable,definitions}.sev`,
`frontend/sentence/src/lib.sev`, `universal/grammar/contracts.sev`,
`universal/cfg/effects.sev`, `transforms/mir/src/callable.sev`.

**Add:** `sev_compiler/universal/constraint/{graph,proof,dispatch}.sev` and
`sev_compiler/frontend/semantic/semantic/src/constraints.sev`.

Start with type/trait facts, integer intervals, static rank and explicit
prerequisites. Implement three-way proof outcomes and stable candidate-set
identity. Build a deterministic topological schedule over guarded checks; merge
only equivalent safe checks with the same substitutions and value versions.
Use cost/selectivity only among legal ready nodes. Emit runtime exact-one-match
logic for admissible dynamic predicates; static-only Unknown is a diagnostic.

**Gate:** rank guards dominate indexing; overlapping true predicates always
diagnose regardless of costs or declaration order; no-match is defined;
effectful/unproven predicates cannot be reordered; a budget cannot convert Unknown
into Proven. Dispatch IR records candidates as well as evaluated predicates.

### P4 — Contracts, refinement, ownership and cleanup on the active CFG

**Edit:** `sev_compiler/universal/cfg/{cfg,block,branch,dominance,availability}.sev`,
`transforms/mir/src/{callable,cfg,ownership,verify,agent_ir}.sev`,
`frontend/semantic/src/{loans,parameter_effects}.sev`,
`frontend/ownership/src/{cfg,loans,places,effects,plan}.sev`,
`transforms/mlir/src/emit/callable.sev`.

**Add:** `sev_compiler/universal/constraint/refinement.sev` and
`sev_compiler/transforms/mir/src/contracts.sev`.

Insert entry/invariant/exit checks at specified CFG sites and record discharged
obligations. Attach facts to versions and kill them on affected writes/calls.
Resolve authority between existing semantic loan checks and MIR ownership; share
facts rather than duplicating mutable ownership graphs. Route all required exits
through appropriate cleanup/check sequences, including typed error propagation.
Extend Agent IR from those same tables with obligations and provenance; verify
every emitted call identity resolves to its declared callable.

**Gate:** negative invariant example fails at the intended boundary; valid one
passes; early return, loop exits, errors and deferred cleanup execute exactly
once; alias writes invalidate facts; loop analysis converges; malformed CFGs
reject. Stage-labelled Agent IR exposes the graph actually used for emission.

### P5 — Semantic interfaces and incremental dependency queries

**Edit:** `library/package/interface/src/{model,codec,query,store}.sev`,
`library/package/metadata/src/{model,codec,validate}.sev`,
`library/package/src/{interface_metadata,reuse,dependency,pipeline,stages}.sev`,
`sev_compiler/frontend/modules/modules/src/source.sev`,
`sev_compiler/boundaries/interface/src/package_metadata.sev`,
`sev_compiler/boundaries/driver/src/pipeline/{package_interface,frontend_archive,mir_archive,prelude_package}.sev`,
`sev_compiler/build/frontend_codec.py`.

**Add:** `sev_compiler/boundaries/interface/src/semantic_graph.sev` and
`library/package/src/semantic_dependencies.sev`.

Version the interface and archive schemas. Serialize typed definitions,
constraints, effects, required generic/compiler-semantic bodies and realization
requirements; validate references on decode. Import into semantic tables directly.
Track query dependencies separately for signatures, predicates/candidate sets,
bodies and realization inputs. Keep versioned readers during migration, then
remove them only after the compatibility policy permits it. Inspect generated
archive output from `frontend_codec.py`; do not manually patch generated codecs.

**Gate:** move the producer source out of the consumer environment and compile
against only interface/artifacts; exercise generics, imported grammar, predicates
and target mismatch. Body edits invalidate inline/const/generic users; ordinary
unrelated clients stay cached. Adding an overlapping implementation invalidates
dispatch consumers. Malformed/dangling/version-incompatible interfaces reject.

### P6 — ObjectUnit planning and realization

**Edit:** `library/package/build/src/model.sev`,
`library/package/artifact/src/model.sev`,
`library/package/src/{build,native,cross_native,link,realization,payload}.sev`,
`sev_compiler/boundaries/driver/src/{package_compiler,package_session}.sev`,
`sev_compiler/boundaries/driver/src/pipeline/source.sev`,
`sev_compiler/compile/compile/src/planner.sev`.

**Add:** `library/package/src/object_units.sev` and
`sev_compiler/universal/module/realization.sev`.

Start with one semantic submodule per ObjectUnit while allowing many-to-many
membership in the data model. Then implement specialization splitting and
compatible-unit merging. Plan exports/imports, initializer ordering and symbol
ownership before emission. Include target/device/ABI/schema/compiler/options in
realization fingerprints. Migrate compile-provider region extraction to canonical
CFG inputs rather than dropping provider routing with the old planner.

**Gate:** one-to-one, one-to-many and many-to-one produce identical observable
behavior; no duplicate/missing symbols or initializers; generic realizations
deduplicate; CPU/device units never merge incompatibly; incremental relinking
includes changed objects without forcing unrelated semantic analysis.

### P7 — Remove scalar semantic alternatives, then old pipeline storage

**Edit:** `sev_compiler/universal/primitive/{int,float,bool}.sev`,
`universal/grammar/contracts.sev`, `frontend/semantic/src/callable.sev`,
`transforms/mir/src/callable.sev`, `transforms/mlir/src/emit/callable.sev`, and
the public exports/manifests of files listed in the retirement ledger below.

Make source implementations own numeric semantics through resolved callable/
operation contracts, with typed backend primitives as the implementation boundary.
Only then remove descriptor fallback selection. Redirect every supported driver
and provider entry to the canonical path. Do not equate source-defined semantics
with removal of compiler-known primitive representations or backend operations.

**Gate:** remove a numeric/source grammar declaration in an isolated sysroot;
the unchanged compiler must reject its use. Mutation of its semantic body must
change native behavior. All public APIs in P0's ledger preserve their supported
features through the new path. Proceed to each deletion only after its gate.

### P8 — Final retirement and clean verification

Add `tests/sev_compiler/v2_architecture.py`, extend `tests/sev_compiler/migration.py`,
and update `migration/{README,INVENTORY,RESULTS}.md`. The new suite should cover
P1–P6 contracts, not merely search for new filenames. Keep negative and historical
regression cases. Record the exact removal manifest and post-removal test results.

Perform a clean build in an isolated output/cache location with retired paths
absent; no source-subject fallback to the seed, old interface stubs or stale
archives. Rebuild the compiler after model/layout changes. Validate cold and warm
builds, then record actual timing/memory/cache behavior against P0.

## 5. Conditional purge ledger

These are **end-state candidates**, not authorization to delete them now. Whole
files and selected fields/branches have different gates. A search result is not
proof of dead code; inspect imports, exports, generated codecs, package manifests,
tests and provider entry points before removing anything.

### Whole-file candidates after replacement

| File | Replacement required before purge | Evidence required |
| --- | --- | --- |
| `sev_compiler/frontend/semantic/semantic/src/grammar_scalar.sev` | General resolved source implementation/constraint selection in P3/P7 | Numeric families, conversions, overflow, comparisons and generic source-body/removal tests pass without its resolver. |
| `sev_compiler/universal/operator/scalar.sev` | Typed primitive/backend operation contracts selected by source implementations | No `ScalarOperation` fields, map producers, codecs or backend consumers remain. |
| `sev_compiler/universal/primitive/numeric/operators.sev` | Real `int`/`float` and family implementations preserving every supported operator | Native primitive, numeric, conversion and generic suites pass; imports/prelude exports moved. Keep neighboring conversion/power files unless separately replaced. |
| `sev_compiler/boundaries/driver/src/pipeline/mod.sev` | Public Compiler/provider APIs routed to the active source/CFG/ObjectUnit path | All public callers migrated; custom compilation, tests and backend selection preserved. |
| `sev_compiler/transforms/lowering/src/lib.sev` | Every supported consumer lowered from verified CFG with equivalent ABI/target behavior | Old lowering package has no remaining supported callers. Remove package/manifests only after auditing other files in it. |
| `sev_compiler/transforms/mlir/src/emit/mod.sev` | Canonical CFG emitter owns all still-used emission helpers/contracts | Move shared symbol/type/ABI helpers first; native and provider emission pass. |
| `sev_compiler/universal/sentence/sentence.sev` | Executable resolved sentence model from P2, with its design explanation retained in docs | Remove the illustrative file only if it is not converted into that implementation in place. Do not delete the new sentence model. |

### Purge selected code or generated artifacts, not entire owner files

| Owner | Retire after its replacement gate |
| --- | --- |
| `universal/operator/syntax.sev`, `universal/grammar/grammar.sev`, `universal/declaration/language.sev`, `universal/grammar/contracts.sev` under `sev_compiler/` | `scalar_operation`, `scalar_syntax`, `scalar_operations`, descriptor maps/imports and analogous migration-only registrations after P7. Preserve grammar metadata and source implementations. |
| `sev_compiler/frontend/semantic/semantic/src/callable.sev`, `transforms/mlir/src/emit/callable.sev` | Descriptor lookup/emission branches after their source implementation replacement. Both files remain central to the active compiler. |
| `sev_compiler/universal/module/module.sev::initializer_cfg`, `universal/declaration/function.sev::cfg` | Empty ABI sentinels only after an isolated seed record/list layout regression is fixed and passes without them. Identify the actual Rust layout defect before naming Rust files to delete; the current comments do not locate it. |
| `sev_compiler/universal/statement/statement.sev::Block.operations` and `Block.lowered_operations` | Old executable storage after planner, ownership, lowering and backend consumers migrate. Keep syntax `Block.statements`. `BasicBlock.operations` is active CFG storage and must remain. |
| `sev_compiler/compile/compile/src/planner.sev` | Nested-structured-operation walkers, replaced with canonical CFG/provider-region analysis in P6. Keep compile routing and artifact handling. |
| `sev_compiler/frontend/ownership/src/{cfg,plan}.sev`, `boundaries/backend/src/lib.sev` | Old block-storage traversal/emission branches after supported ownership/backend behavior moves. Do not delete resource checks or alternate backend support without replacements. |
| `sev_compiler/frontend/modules/modules/src/source.sev`, `boundaries/driver/src/pipeline/package_interface.sev`, `library/package/src/interface_metadata.sev` | `.sevi`-to-source reconstruction and scalar-only compiler-interface generation after P5 source-free import parity. Keep source loading and foreign ABI adapters that remain necessary. |
| `sev_compiler/build/frontend_codec.py`, driver archive modules, `library/package/src/{realization,payload}.sev` | Stale schema fields and legacy reader/writer branches after compatibility gates. Keep archive generation, relocation and cache validation mechanisms. |
| Generated `package.pkgi/severian/<build-id>/scalar.sev`, `interface.bin`, old frontend/MIR archives and cache receipts | Retire their producers/readers first, then invalidate the corresponding old format namespace. Remove obsolete outputs only from isolated test/build roots for validation; do not blanket-delete user caches or supported historical package formats. |

Paths abbreviated within a row share its explicitly stated `sev_compiler/` root.
After each purge, remove dangling exports/dependencies and regenerate codecs and
lockfiles. Retain an explicit compatibility reader if published packages still
require it, and label that exception; do not claim complete format retirement.

### Keep unless a separate replacement project proves otherwise

- `rust_compiler/`, `Cargo.toml`, the seed CLI and bootstrap tests: V2 source
  architecture does not by itself establish reproducible self-hosting.
- `sev_compiler/universal/cfg/`, active MIR transforms/verifiers and
  `transforms/mlir/src/ir/`: these enforce the executable contracts V2 needs.
- Lexing, parsing, source storage/maps, definition resolution, structured syntax
  and diagnostic provenance. AST/HIR names are not evidence of duplication.
- Primitive representations, operation/effect/type contracts, foreign ABI and
  compile-provider boundaries. Audit prototype-looking files individually; do not
  delete `universal/operator/registry.sev` merely because its design is unfinished.
- All behavioral regression suites. Update `grammar_scalar.py` to test the new
  ownership of semantics or migrate its cases before retiring a mechanism-specific
  assertion; do not delete failing tests to make migration green.

## 6. Validation that actually establishes migration

The existing 30 migration cases are a floor, not a completeness proof. Require:

| Gate | Required observable evidence |
| --- | --- |
| Single authoritative pipeline | All supported compile/check/test/library/provider paths reach the verified canonical representation; Agent IR and backend expose matching identities/CFG, not a reconstructed demonstration graph. |
| Source semantic authority | With one compiler executable hash, change and remove real grammar/numeric/predicate implementations; observe changed behavior or a resolution error, never hidden fallback. |
| Graph validity | Endpoint kinds, containment, recursion/SCCs, stable IDs, source origins, CFG dominance/terminators, dispatch prerequisites and dangling-reference rejection. |
| Contract soundness | Mutation/alias invalidation, ordinary deferred cleanup, all specified exits, ambiguity and no-match, integer overflow and floating NaN edge cases. |
| Incrementality | Build logs show the expected changed query/ObjectUnit closure for body, exported constraint, candidate addition, target and file-move changes. Compare outputs with a cold build. |
| Interface independence | Consumer has no producer `.sev` files or old source adapter; generic and grammar requirements still resolve from the supported interface, or diagnose an explicitly unsupported case. |
| Artifact correctness | Split/merged units preserve symbols, ABI, initialization, link reachability and execution; incompatible targets/schemas reject rather than reuse stale objects. |
| Physical retirement | The manifest's old implementations are absent, every import/export is updated, and a fresh compiler build plus full relevant suites pass without old cache state. |
| Performance | Fixed corpus, tools, target and resource budgets; report compiler wall time/peak RSS, cache work, artifact size and dispatch behavior. No assumed improvement from graph shape alone. |

Suggested existing checks, after a guarded rebuild of the candidate compiler:

```sh
python3 tests/sev_compiler/resource_guard.py --timeout 180 --memory-bytes 6000000000 --report /tmp/v2-compiler-build.json -- sev_rust build sev_compiler --bin sev_compiler --build-profile release -o sev_compiler/package.pkg/host/dev/bin/sev_compiler
python3 tests/sev_compiler/migration.py -v
python3 tests/sev_compiler/cfg.py -v
python3 tests/sev_compiler/source_contracts.py -v
python3 tests/sev_compiler/deferred_cleanup.py -v
python3 tests/sev_compiler/explicit_ownership.py -v
python3 tests/sev_compiler/artifact_layout.py -v
sev test sev_compiler
```

Also run the affected callable/diagnostic/primitive/numeric/generic/library and
package-interface suites named by each phase, plus the new V2 tests. Use
`tests/sev_compiler/RESOURCES.md` budgets and serial compiler builds. A failed or
timed-out check is not a pass. Capture Agent IR via the source compiler's
`--emit agent-ir` boundary and preserve the selected executable hash with results.

## 7. Review verification record

This change is documentation only. Repository paths, active call sites, interface
formats and retirement candidates were inspected; proposed new paths are marked
as additions. Any targeted checks run for this review are recorded below and do
not establish fresh-build or full-suite migration status.

Targeted checks against the existing source compiler binary (not rebuilt during
this review), run on 2026-09-22:

| Check | Result |
| --- | --- |
| `migration.py Gate1Baseline.test_05_active_pipeline_is_inspectable` | **FAIL:** Agent IR emitted 1,492 definition entries but only 1,490 unique IDs. Fix definition registration/serialization identity in P1 before claiming graph validity. This check does not isolate the collision cause. |
| `migration.py Gate6LibraryAndRetirement.test_05_legacy_dispatch_and_parallel_topology_are_retired` | **PASS:** the existing forbidden-pattern check passes despite the remaining descriptor/older paths documented above. It is insufficient evidence of complete retirement. |
| `git diff --check` | **PASS** for the tracked documentation edit; no compiler implementation changed. |

Compiler SHA-256:
`2b5a4ca0a6d33cece42445a67c3b7a417df4cce965c0511510f785d543c94e46`.
The two migration checks took 55.449 seconds in total. The full suite and a fresh
compiler build were not run. The failing identity check is a concrete baseline
blocker, not evidence that the proposed hierarchy is impossible.
