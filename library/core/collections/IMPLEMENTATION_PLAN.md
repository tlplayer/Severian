# Library collections through ordinary generics

The implementation target is `sev_compiler`. The Rust compiler is the seed;
change it when a seed limitation prevents building the source compiler.
Source-free `.pkgi` imports are outside this work.

Current delivery: source-compiler named type arguments, record type-constructor
bindings, generic alias forwarding, applied trait conformance, explicit generic
function applications and typed union arguments. Eight native/diagnostic test
groups exercise these mechanisms in
[`collection_generics.py`](../../../tests/sev_compiler/collection_generics.py).
The real set/map providers, provider defaults, variadic collection construction
and removal of compiler-hosted collection algorithms are not yet migrated.

## Public application syntax

```sev
ordered = set[T:int, C:BTree](8, 2, 5)
hashed = set[T:int, C:Hash](8, 2, 5)
lookup = map[K:string, V:int, C:BTree](entries...)
```

These are destination collection APIs, not claims that the present set/map
packages already implement them. At an application, `T:int` binds a named
parameter. At a declaration, `T:Ordered` remains a capability constraint.
Names such as `Element` and `Backend` work identically; the canonical compiler
vocabulary does not reserve parameter spellings.

`C` binds a type constructor with resolved declaration identity. Applying it
to element/key/value arguments produces an ordinary concrete type. A set owns
one mapping specialization with a membership payload, rather than two copies
of its owned element. `BTree` and `Hash` are library declarations, not enum tags
interpreted by the compiler. Qualify or alias the hash provider where it would
otherwise clash with the existing hashing capability named `Hash`.

Named arguments normalize into declaration order before specialization. Reject
unknown names, duplicate bindings, missing required parameters, wrong argument
kinds and wrong arities. A positional argument cannot follow a named argument.
One parameter binding has the same identity regardless of its position in the
application. Default provider arguments belong to generic declarations; the
compiler must not assign a special default to the spelling `set`.

Prelude names are reserved call identities. Collection extensions implement
polymorphic operators or use a package's explicit `[prelude].exclude` selection;
they must not rely on silently shadowing an unrelated prelude function. Qualified
package exports retain their own identities. The current source compiler tests
this policy in `tests/sev_compiler/prelude.py`. Exclusion of compiler-handled
intrinsics remains an explicit unsupported diagnostic pending extraction.

## Trait and class division

The small source-compiler contract to establish first is:

```sev
trait Readable[T]:
    def get() -> T

class Box[T]: Readable[T]:
    value: T
    def get() -> T:
        return value

class Collection[T, C]:
    storage: C[T]
    def Collection(value: T):
        storage = C[T](value)
    def get() -> T:
        return storage.get()
```

The same machinery must support a mapping provider's key/value methods,
associated iterator state and ownership modes. Contracts describe required
operations. Classes supply fields and algorithms. Applying `Readable[int]`
substitutes method parameters, results, properties, inherited contracts and
operators; an implementation returning `bool` is a type error.

For full collection contracts:

| Contract | Responsibility |
| --- | --- |
| `Collection[T]` | Cardinality, truth and iteration; no storage choice |
| `Sequence[T]` | Checked indexing and sequence operations |
| `Map[K,V]`, `MapIndex[K,V]` | Mapping semantics and keyed access |
| Ordered mapping | Ordering, extrema, bounds and half-open ranges |
| Hashed mapping | Equality-compatible hashing and hash-table operations |
| `SetLike[T]` | Unique representatives and membership |

Reuse one canonical equality, ordering, hashing and ownership contract identity.
The existing `collections.traits.HashKey` and `Ordered` declarations must be
reconciled with their universal counterparts during extraction. Do not make all
maps require hashing, or all stored values require `Default` or `Copy`.

Optional lookups borrow from the owner; removal transfers ownership. Choose the
exact absence/index-error spelling in a separate API migration with callers.
`size(values)` counts elements. Do not silently change current `.size()`,
`reserve`, `get`, or `pop` signatures while moving their implementation.

## Typed input and output

* An explicit element type checks every supplied value against that type using
  the ordinary conversion rules. It does not widen on subsequent insertion.
* A union element/value type remains that union in storage and at lookup.
  Refinement must follow the same pattern and conversion rules as any user type.
* Heterogeneous packs retain the individual types of their positions. A tuple
  pack is not an erased homogeneous collection.
* `array[T,N]` has an element type and a constant element count. `S` denotes
  shape; neither `S` nor `N` is a container implementation selector.
* A runtime length does not create a compile-time specialization. Static type,
  constructor and constant arguments do; runtime values remain normal inputs.
* Algorithms changing element type must declare a distinct result parameter.
  A callable mapping `T` to `R` produces storage for `R`, not storage for `T`
  with a cast at its boundary.
* Borrow/move/drop, mutation and throwing remain effects of ordinary calls.
  Their checks must survive specialization and early exits.

These rules apply to user-defined classes as well as compiler records. Universal
types, declarations, symbols, conversions, callables, patterns, constraints and
effects retain their compiler meaning. Only collection mechanisms move out.

## Package ownership and extraction order

Make this directory a `collections` package whose entry point re-exports leaf
declarations with their original identities. Keep leaf packages independently
publishable. Internal storage packages are dependencies, not public exports.

| Order | Package work | Actual compiler dependency being removed |
| --- | --- | --- |
| 1 | Shared initialized-slot storage, then list and array | Active buffer-list construction, access, copying and growth in `universal/primitive/collections.sev` |
| 2 | Map contracts, hash table and insertion-ordered dict; B-tree provider and set | `universal/collections/{map,dict,btree,set}.sev` algorithms and runtime set selector |
| 3 | Deque, queue, stack, heap and count | Remaining compiler-hosted algorithms; adapters reuse list/deque |
| 4 | Fixed-lane vector | Numeric vector behavior, independent of ordinary collection storage |
| 5 | Bitset, arena and interning for demonstrated consumers | Compiler-local storage; preserve scope, symbol, node and scheduling semantics |
| 6 | Re-exporting facade and published external consumers | Relative compiler-source imports and obsolete bootstrap adapters |

The current inventory has three different sources: package classes, the
algorithm implementations in `universal/collections`, and the active
`type list[T] = buffer[T]` provider. A file move alone does not reconcile their
APIs or establish that they execute through the source compiler.

Storage tracks live slots explicitly. A contiguous list, wrapped deque and
sparse table have different live layouts. Spare capacity never constructs
dummy owned values. Checked growth, relocation, partial initialization cleanup
and destruction belong to shared library storage and memory operations.
MLIR lowering handles typed layouts and memory primitives; collection algorithms
and string/character manipulation do not become C helpers.

## Compiler milestones and deletion gates

1. Named applications and generic record/type-constructor identity. Test nested
   applications, declaration scopes, aliases, reordered arguments and failures.
2. Applied trait substitution and conformance, including generic inheritance,
   method/operator results, generic body checks and provider-specific constraints.
3. General provider defaults, function/method inference of constructor arguments,
   variadic construction and static-value parameters. Never special-case `set`.
4. Owned aggregate storage and borrowing iteration. Reject growth/removal while
   a live reference could be invalidated; test destruction exactly once.
5. Migrate one real compiler consumer per collection and run it through `sev`.
   Remove old providers only after those consumers use the package implementation.

Do not treat these milestones as complete merely because parsing or a Rust-seed
build succeeds. Record native source-compiler results separately.

## Tests and publication

Generic language diagnostics belong beside parser/semantic tests. Storage and
algorithm tests belong beside their packages. Use adversarial collisions,
deque wrapping, B-tree split/merge/root contraction, allocation failure and
owned-element counters, rather than tests that only mirror method bodies.

Publish leaf dependencies before the facade into an isolated registry. Test an
external consumer defining a new element type after removing the checkout from
its dependency search path. Source-inclusive immutable snapshots are sufficient.
Re-exporting must preserve declaration identity; it must not copy definitions.

Run the documentation examples before and after the changes with the respective
source compilers. Compare build, native execution and test status independently.
Keep pre-existing failures and exact diagnostics in the report. A stale installed
compiler failing manifest parsing is not a useful language-regression baseline.
