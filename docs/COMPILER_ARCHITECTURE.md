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

## Provenance-aware semantic traits

Trait composition has two compiler views with different jobs:

```text
direct trait graph ──┬──> expanded requirements for conformance checking
                    └──> provider graph for semantic resolution
```

The requirements view may deduplicate identical inherited signatures. The
provider graph never infers ownership from that flattened list: it traverses
the direct composition edges and records members as `Trait::member` or
`Trait::operator`. Consequently, two traits may both declare `@`, `matmul`, or
`broadcast` without losing their identities.

A decorator declared inside a trait defines a semantic marker. When that marker
decorates a function, HIR receives a `SemanticContext` containing:

- the capability trait and explicitly selected composed traits;
- every operator and operation candidate with its provider;
- ordered, provider-qualified `with`/`without` scoped behavior;
- a selected provider when explicit context or a single candidate proves one;
- named policy values inherited from the trait marker and overridden at use.

Resolution follows the same boundary for every semantic member: zero candidates
is an error, one selects automatically, and multiple candidates remain legal
until the source uses the member. An unresolved use reports `E000210` and asks
for a selector such as `@tensor(xla)`. Policy-driven `auto` planning may choose
a unique legal candidate later without changing this source or HIR shape.

Function-header entries such as `with { metric }` are resolved against the same
trait marker registry as `@metric`; the latter is syntax sugar and both produce
the same HIR decorator and `SemanticContext`. Other expressions in the set stay
ordinary Boolean contract clauses.

Composed behaviors enter in trait declaration order and exit in reverse order.
HIR preserves the structured scope, while MIR emits explicit provider-qualified
entry and exit operations. Return and loop-control terminators receive cleanup
operations for every scope they leave. This establishes stack semantics before
backend lowering and prevents decorators from becoming arbitrary wrapper
functions.

The current vertical slice resolves the trait-owned tensor `@` operator,
validates paired lifecycle declarations, and preserves lifecycle sequencing
through MIR. Backend cost planning and execution of compiler-context lifecycle
bodies remain subsequent lowering stages.

## Compile-time trait implementation registries

A trait `property` is required, typed provider metadata rather than a mutable
global or an extensible enum variant. Semantic analysis examines the complete
reachable package interface set, validates every provider, rejects missing,
non-constant, mistyped, and overlapping contributions with `E000212`, and emits
a deterministically ordered `TraitRegistryDefinition` in HIR metadata.

Traits remain open while packages are composed, but the implementation set is
closed for each executable compilation. This gives later lowering enough
information to synthesize static lookup and dispatch tables without package
initializers, runtime reflection, or service discovery. HIR currently preserves
the complete table; backend dispatch synthesis is the next consumer boundary.

## Foreign ABI boundary

Package-owned native providers cross one typed compiler boundary:

```text
package manifest + private ABI declarations
            │
            ▼
HIR ForeignCall { ForeignSymbol, ABI signature, arguments }
            │
            ▼
MIR ForeignCall
            │
            ▼
generic ABI shim and native symbol link
```

`library/abi` owns source-facing calling conventions, layouts, pointer and
buffer shapes, nullability, and ownership vocabulary. `library/ffi` owns
foreign library and symbol identities. Package manifests own provider sources,
target selection, include paths, and link-library requirements. The compiler's
closed `severian-abi` descriptors are validation and lowering data, not public
domain APIs.

Generic lowering may inspect `AbiType`, `CallingConvention`, and ownership, but
must not branch on a standard-library package or provider symbol. It emits the
same conversion and call machinery for every package. Architecture tests scan
the compiler lowering/backend boundary for migrated provider names to keep that
dependency direction intact.

The first `file` slice moves text-family reads to
`library/file/native/posix/file.c`; JSON, YAML, and CSV reader adapters share
that path. Binary handles, writes, directories, mapping, and locks remain on
the legacy platform bridge until separately migrated. The old compiler-owned
text-read implementation has been removed.

File-format selection is now a closed `Reader` trait registry. Reachable reader
implementations contribute `extensions` and `document_class` metadata; lowering
generates runtime dispatch, and semantic analysis uses the same metadata for
literal-path refinement. JSON and CSV parsing/encoding are ordinary source code
in their owning packages. The compiler bridge retains only format-independent
dynamic map/list/value representation primitives.

Tensor operations are not modeled as ordinary foreign calls. Tensor HIR/MIR,
shape and dtype analysis, fusion, legalization, and XLA/MLIR lowering remain
compiler responsibilities; only a genuine external ABI call uses this path.

The broader ownership audit and ordered migration ledger live in
[`DOMAIN_IMPLEMENTATION_MIGRATION.md`](DOMAIN_IMPLEMENTATION_MIGRATION.md).

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
