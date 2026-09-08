# Collections implementation plan

Status: bootstrap source algorithms have been implemented; see [CAPABILITIES.md](CAPABILITIES.md) for coverage, API decisions, validation and outstanding language dependencies. C0–C9 are not complete. The present-state inventory below records the original 2026-09-08 planning baseline.

This plan follows the [primitive plan](../primitive/IMPLEMENTATION_PLAN.md). Generic layouts and contract dispatch (P1), scalar operations (P2), memory/ownership (P5), and arrays/slices/ranges/iteration (P6) are prerequisites. Core strings (P7) enable realistic keyed and owned-element tests. Full string completion (P8) then consumes `list[string]`.

## Coverage and present state

| File | All declared types/contracts | Present source and required outcome |
|---|---|---|
| `map.sev` | `Map[K,V]`, `MapIndex[K,V]` | Semantic contracts and `mapping_size`; no concrete `map` class. Reconcile copy-return signatures with documented borrow/ownership end state. |
| `list.sev` | `List[T]`, `list[T]` | Contiguous growable allocation, initially `Copy + Default`; make executable and support owned elements. |
| `vector.sev` | `VectorScalar`, `Vector[T,N]`, `vector[T,N]` | Fixed-length mathematical vector backed by `array`; verify lane operations, reductions, callable `map`, and shape constraints. |
| `deque.sev` | `Deque[T]`, `deque[T]` | Ring-buffer bodies; validate wrapping, growth, logical indexing, and lifetime behavior. |
| `heap.sev` | `Heap[T]`, `heap[T]` | List-backed minimum heap; validate ordering and all replacement/empty cases. |
| `dict.sev` | `Dict[K,V]`, `dict[K,V]` | Insertion-ordered parallel lists with a deliberate linear probe; establish correctness, then add source-owned hashing. |
| `btree.sev` | `btree_node[K,V,N]`, `btree[K,V]` | Fifteen placeholder bodies, inconsistent `ptr` naming/node arity, and missing storage/ordering details. Implement the ordered engine. |
| `set.sev` | `set_storage` (`BTree`, `Hash`), `set[T]` | Duplicate constructor signatures and tree-only operations despite the storage selector. Implement both providers and complete iteration/algebra. |
| `count.sev` | `Count[T]`, `count[T]` | Dictionary-backed frequencies; validate counters, zero-removal, iteration, copying, and overflow. |

## C0 — Resolve contracts before storage implementation

Keep a method-level coverage ledger for every file, including operators, truth behavior, constructors, iteration, copy/clone, cleanup, and existing tests. Start with seed/source/native baseline results rather than assuming method bodies are executable.

- Preserve the distinction between semantic constraints (`Ordered`, `Hash`, `Equatable`) and temporary storage constraints (`Copy`, `Default`). Methods such as `list.find` need an equality capability even though unrelated list operations do not.
- Reconcile `Map.get`'s `(V, bool)` bootstrap form with its documented eventual `borrow V | None`; similarly define borrowing lookup/indexing and ownership transfer on removal/replacement across collections. Define absence without fabricating `V.default()` once arbitrary owned elements are supported.
- Keep `Map` a contract. Its opening B-tree claim conflicts with the insertion-ordered concrete `dict`; correct the documentation and avoid introducing an unsolicited second concrete `map` type.
- Decide whether `Map` itself needs `Hash` or whether hashing belongs only to `Dict`. Recommend `Equatable` for the shared identity contract, `Hash + Equatable` for `Dict`, and a compatible total order for B-tree keys. Update generic examples and conformance tests together if adopted.
- Provide the referenced `KeyError`; unify `pointer`/`*[T]` with B-tree's `ptr`; define node capacity and constant arguments consistently. Adopt the primitive plan's range and iterator identities.
- Specify borrow invalidation under growth/removal, mutation during iteration, aliasing in `extend`, and failure cleanup. Use compile-time loans to reject invalidation while a reference is live; bounds checks alone are insufficient.
- Resolve set storage semantics without changing `set[T]` into separate public types. Recommended initial contract: BTree remains sorted/default; Hash iteration has a documented deterministic order (insertion order if reusing dict). Ordered queries require `Ordered` and may scan the hash provider. Runtime storage selection requires capabilities for both branches; statically selected providers should require only their own capabilities where specialization permits.

Exit: consistent API signatures, provider constraints, ordering/absence semantics, and diagnostic fixtures. These decisions should be encoded before implementing a second storage engine.

## C1 — Shared storage and iteration

- Consume the P5/P6 allocation, layout, initialized-slot, move/drop, borrow, range, and iterator mechanisms. Collections must not add their own compiler-recognized type-name dispatch.
- Share checked growth arithmetic and allocation/relocation helpers. Track initialized elements separately from capacity; unused capacity must not require constructing dummy `T` values.
- Support iterator state machines generated from `yield`, early-exit cleanup, and borrowing elements from stable owner storage. `items()` must not borrow a temporary pair; model an owned pair of references or another explicit iteration result.
- Preserve safe failure behavior: an allocation or element-construction failure leaves existing contents valid and destroys every newly initialized element exactly once.
- Make literals, indexing, slicing, iteration, truth, and compound indexed assignment resolve through ordinary source contracts. Evaluate keys/indexes/receivers once.

Exit: generic storage works for a scalar, a nontrivial record, and an owned value with a counted destructor. Mutation invalidating a live slice/iterator is rejected; completed and abandoned iterators release loans.

## C2 — List and fixed-size vector

### List

Make all declared operations executable: empty/array/slice construction; indexing and slicing; `len`/`cap`; `reserve`, `append`, `extend`, `insert`; both `pop` forms; `find`, `remove`, `swap`, `truncate`, `reverse`, `clear`, explicit copy, iteration, and drop.

Preserve contiguous storage and amortized constant-time append with geometric growth. Specify `reserve(additional)` as room for at least `len + additional`. Validate middle shifts, zero capacity, overflow, self/overlapping extension, and old-slice invalidation. Deep copying the container must allocate independent storage.

### Vector

Implement `VectorScalar`/`Vector` conformance over `array[T,N]`: lane access, pairwise arithmetic, scalar multiplication/division, `sum`, `dot`, `length_squared`, callable `map`, copy, and iteration. Preserve compile-time lane count and reject mismatched shapes.

Use scalar loops first. SIMD lowering is a later optimization with identical semantics, particularly for floating reductions and fused operations. Clarify the additive identity used by `sum`: `Default` alone does not promise numeric zero for a user-defined scalar.

Exit: native list model tests covering operation sequences; allocation independence and capacity invariants; vector tests for zero/one/multiple lanes, integer and floating scalars, custom scalar contracts, return-type-changing `map`, and shape rejection.

## C3 — Owned elements and compiler list migration

The initial `Copy + Default` specialization is a milestone, not the final public restriction.

- Enable `list[string]`, `list[record]`, and nested collections using initialized-slot storage and explicit ownership transfer. Restrict fill/copy/default construction only where each operation needs the corresponding capability.
- Distinguish an explicit deep container copy from cloning element values. Move-only elements remain usable through operations that do not copy; reject copying when unsupported.
- Change borrow-return APIs from their bootstrap forms atomically with callers/tests and diagnostics. Verify removal transfers ownership, replacement returns or destroys the old value as specified, and clear/drop clean up exactly once.
- Migrate actual compiler list consumers from `primitive/collections.sev` buffer aliases to the canonical provider. Keep the seed adapter isolated until the source compiler builds and runs with the new list.

Exit: native nested-container and destructor-count tests, compiler build/consumer regressions, and safe `list[string]`. This unlocks primitive P8 and owned keys/values in later collections.

## C4 — Deque and heap

For `deque`, finish empty/capacity construction, reserve, both-end push/pop/peek, logical indexed access, clear, copy, iteration, truth, and drop. Grow a wrapped buffer into logical order and reset its head correctly. Test wraparound followed by growth, repeated empty/refill, middle indexes, and destructor behavior.

For `heap`, finish push, peek, pop, replace, push-pop, sift-up/down, clear, and truth over the canonical list. Guard unsigned parent arithmetic at the root. Preserve the minimum-heap invariant, duplicate values, and specified empty behavior. Reuse list ownership cleanup, including when comparisons can fail.

Exit: randomized deque comparison with a sequence model; heap pops match a sorted reference model; replacement/push-pop and duplicate-key cases pass; unsupported ordering is diagnosed. Count comparisons to catch accidental linear insertion/pop behavior.

## C5 — Map conformance and insertion-ordered dictionary

1. Make the existing linear implementation fully executable as a correctness reference. Preserve unique keys under equality, replacement position, remove/reinsert-at-end, strict indexing, nonthrowing lookup, set-default, reserve, copy, iteration, and truth.
2. Validate `Map`, `MapIndex`, and `Dict` generic dispatch, including expected-type-directed dictionary literals. Supply borrowed reads and owned insertion/removal after C3.
3. Add a Severian hash index over stable logical entries, retaining insertion order independently of bucket placement. Implement collisions, resizing, deletion/tombstones or equivalent bookkeeping, reuse, and compaction.
4. Preserve `equal => same hash`; hash equality never establishes key equality. Define stability requirements for stored keys and disallow safe mutation that breaks indexing. Do not accidentally admit IEEE NaN as an equivalence relation.

Exit: differential operation sequences against the linear reference; forced collisions and constant hashes; rehash/order preservation; absent/default/replace cases; owned key/value drop accounting. Average-case lookup/insert/delete should be constant-time under a well-distributed hash; collision worst cases remain explicit.

## C6 — B-tree engine

- Define one internal node capacity/minimum degree and enforce its bounds. Propagate `N` correctly through recursive node pointers, roots, and helper signatures. Keep partially occupied keys/values/children in initialized-slot storage.
- Implement empty construction, search/get, split-child, insert-non-full, insertion/replacement, and root splitting.
- Implement deletion from leaves and internal nodes, predecessor/successor replacement, borrowing from siblings, merge-children, rebalancing, and root contraction.
- Implement first/last, lower/upper bounds, bounded traversal, ordered iteration, clear, and destruction. Define endpoint inclusivity through the shared range contract.
- Tie traversal references to tree loans. A split/merge must not leave safe references pointing to moved entries. Specify key-order/equality compatibility and failure behavior of comparison.

Exit: validate occupancy bounds, sorted keys, child counts, subtree ranges, uniform leaf depth, and stored count after every mutation in randomized tests. Compare lookup/removal/bounds/traversal to a sorted reference. Exercise multiple small node capacities so every split/merge/root case is forced; verify complete destruction.

## C7 — Both set storage choices

- Replace the duplicate constructors with unambiguous default/explicit-provider forms. Initialize the selected provider and cardinality before insertion.
- Dispatch add/get/remove, membership, clear, truth, and iteration to real BTree and Hash storage. Preserve the existing stored representative when an equivalent value is added.
- Implement first/last, lower/upper bounds, pop-first/pop-last, and range according to the C0 contract. Hash-provider complexity must be documented when ordered queries scan or sort.
- Complete `size`, union, intersection, difference, symmetric difference, subset, superset, and disjoint for both providers and mixed-provider operands. Define the result provider explicitly; preserving the left operand's provider is the proposed default.
- Avoid duplicating ownership by storing an owned value as both key and value in `btree[T,T]`. Use a membership payload or another representation with one owning copy of each element.

Exit: identical membership/algebra results across providers; sorted default traversal; explicit hash iteration behavior; representative retention; mixed-provider and empty cases; generic capability diagnostics and owned-element cleanup.

## C8 — Count

Implement all `Count` methods over canonical `dict`: constructors, frequency/indexing, membership, add/subtract/update/remove/clear, unique and total cardinalities, key/item/expanded-element iterators, and copy.

Maintain `unique == number of positive entries` and `total_count == sum(frequencies)`. Zero additions do nothing; subtraction floors at zero and deletes exhausted entries. Check overflow before updating either the dictionary or totals, so failure cannot leave inconsistent counters. Expanded iteration must stream repetitions rather than allocate `total_count` elements.

Exit: randomized multiset model comparison; zero/excess subtraction, maximum counters, absent removal, overflow rollback, copy independence, deterministic underlying order, and iterator cleanup.

## C9 — Integration and completion

- Run every declared API through source checking and native execution, including compile-failure ownership/constraint cases. Extend `tests/sev_compiler` with focused primitive/collection migration fixtures and retain executable specifications beside each type.
- Reuse migration harness temporary sysroots and binary digests. Editing a collection method in `.sev` must change behavior without recompiling the compiler. Removing a required provider must fail visibly rather than fall back to a compiler builtin.
- Run allocation/destruction counters and sanitizer-backed native cases where the toolchain supports them. Check representative complexity with operation/allocation counts rather than timing-only assertions.
- Rebuild the source compiler using the checkout seed; execute compiler workloads using the migrated collections. Use `sev build --emit agent-ir` to inspect generic method targets, element layouts, bounds checks, and cleanup edges. Keep native and Agent IR validation distinct.
- Retire the primitive list alias and obsolete collection-specific lowering only after actual compiler consumers migrate. Run source-contract, lexical-rule, diagnostics, CFG, and compiler-build regressions affected by that switch.

Completion means all nine files' contracts and concrete types work through Severian definitions, both set providers work, B-tree has no placeholder bodies, dictionary hashing preserves insertion order, owned-element semantics are validated, and the compiler consumes the canonical collections.

## Recommended delivery order

Land small, reviewable changes with the exit gate from the corresponding section:

1. P0/C0 coverage ledger and contract corrections.
2. P1 generic declaration/layout/conversion support; P2 Boolean/integer core.
3. P3 floating formats and P4 character support, with textual edges tracked until P7.
4. P5 memory/ownership; P6 arrays/slices/ranges/iteration.
5. P7 owned string core and scalar conversions; C1/C2 storage, list, vector.
6. C3 owned elements and compiler list migration; P8 complete string API.
7. C4 deque/heap; C5 dictionary; C6 B-tree.
8. C7 set and C8 count.
9. P9/C9 prelude replacement, fallback removal, and integration acceptance.

This is a dependency order, not a requirement to delay independent work: fixed-size vectors can proceed after arrays, count after dictionary, and B-tree after ownership/storage. Exotic floating formats do not need to block initial scalar collection development, but remain required before the complete primitive plan is done.
