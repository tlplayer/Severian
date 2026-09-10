# Collections implementation and validation

The source algorithms now cover all nine original modules. This is the
`Copy + Default` bootstrap implementation, **not completion of C0–C9** in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Owned elements, compiler
consumer migration and native collection acceptance remain open.

## Implemented behavior

| Module | Source implementation / algorithm checks |
| --- | --- |
| `list` | Constructors, indexing/slicing, checked reserve, append, snapshot-before-growth extension, insert, both pops, equality search/removal, swap, truncate, reverse, clear, independent copy, iteration, allocation release. Random operation sequences and aliased extension exercise storage bounds. |
| `vector` | Fixed lanes, arithmetic and scalar broadcasts, sum/dot/squared length, callable map, copy and iteration. Zero/one/four-lane algorithm checks; inline tests include float-returning map. `VectorScalar.default()` must be additive zero. |
| `deque` | Both constructors, checked growth, wrapping without overflowing head arithmetic, both-end push/pop/peek, indexing, clear, copy, iteration and allocation release. Random sequences include non-power-of-two capacities and wrapped growth. |
| `heap` | Push/pop/peek, replace, push-pop, sift operations and clear. Child arithmetic runs only for parent indices. Differential tests include empty cases and duplicate minima. |
| `map` | Equatable identity contract independent of hashing, strict indexing contract, `KeyError`, and `mapping_size`. This remains a trait, not another container. |
| `dict` | Open-addressed buckets, cached hashes, tombstones, resizing, stable reusable slots, insertion-order links, lookup/indexing, replacement, set-default, removal, reserve, clear, all iterators and copy. Forced collisions, order/link invariants and rehash-count tests. |
| `btree` | Search, replacement, splitting, non-full insertion, leaf/internal deletion, predecessor/successor substitution, both sibling rotations, merging, root contraction, extrema, bounds, bounded/whole traversal, clear, copy. Random mutations at degrees 2, 3 and 8 check occupancy, key/subtree order, leaf depth, cardinality and free-row accounting after every mutation. |
| `set` | Unambiguous provider-first construction, actual BTree/Hash dispatch, stored representative retention, membership, extrema/bounds/pops, sorted ranges, copy, iteration, clear/truth/cardinality, all seven algebra/predicate helpers. Tests include both providers and mixed operands. |
| `count` | Constructors, frequency/indexing, membership, add/subtract/update/remove/clear, unique/total cardinality, key/item/expanded iteration and copy. Overflow is checked before mutation; random multiset comparison and overflow rollback checks. |
| `storage` (new helper) | Shared checked size addition and geometric growth with a non-overflowing final step. Slot-count checks do not replace allocator byte-size checks. |

Each module retains source tests next to its implementation. Parsed method,
constructor, operator, property and test inventories are generated in
`declarations.json` by the compiler measurement runner; they are not inferred
from method-looking text inside documentation.

## API decisions

- `Map` and `MapIndex` require `Equatable`; `Dict` additionally requires `Hash`.
  The previously empty canonical Ordered trait now declares equality and the
  four comparison operators, and the storage helper imports scalar capabilities.
  B-tree keys require a compatible total order. Equal hash keys must retain
  equal, stable hashes throughout storage. IEEE NaN is not an equivalence key.
- Bootstrap lookup/removal/replacement returns `(value, found)`. Tree extrema
  and bounds return `(key, value, found)`; set extrema and bounds return
  `(value, found)`. The old unimplemented tree/set `None` signatures were
  reconciled with the executable algorithm surface. Borrow-return/owned
  removal APIs still depend on primitive storage and loan support.
- `btree_node` is now a non-generic, copyable metadata record. Keys/values and
  child indices live in parallel arena rows owned by `btree[K,V]`. There are no
  recursive owning node pointers. Default degree is 8; an explicit degree
  constructor permits invariant testing. Released rows are reused; clear
  releases logical slots and retains the backing list capacities.
- Construct `set[T](values...)` or `set[T](set_storage.Hash, values...)`.
  Runtime selection requires both ordering and hashing capabilities. Only the
  chosen provider receives entries; the other starts empty. Each element is
  stored once as a key, with a bool payload. Algebra preserves the left provider.
- Default set iteration is sorted; Hash iteration is insertion ordered. Both
  providers return sorted half-open ranges. Hash ordered queries scan, and
  Hash range builds a temporary tree; these are not constant-time operations.
- `reserve(additional)` makes room for at least current length plus additional.
  Dictionary buckets stay at most half live. Rehashing at three-quarter used
  occupancy leaves a tombstone budget, avoiding repeated full rehashes during
  steady-size removal/insertion workloads.

## Reproduce validation

Run from the repository root:

```sh
python3 tests/sev_compiler/collection_algorithms.py
python3 tests/sev_compiler/collection_capabilities.py --record-only
# Strict acceptance: currently fails on missing language capabilities.
python3 tests/sev_compiler/collection_capabilities.py
(cd sev_compiler && ../package.pkg/debug/sev build --emit agent-ir)
```

The 12 algorithm tests execute the actual `.sev` method bodies through
[collection_source_adapter.py](../../../tests/sev_compiler/collection_source_adapter.py).
The adapter erases annotations and provides Python primitive storage; it does
not supply a second implementation of the tree, hash table or other algorithms.
It validates algorithms only: **it is not a compiler, native run, trait check,
integer-width proof, or ownership/destructor test**. It cannot establish the
language's call/overload resolution. Its instrumented raw memory detects bounds,
uninitialized reads and explicit double frees in list/deque operations.

The compiler runner independently records seed parse/check/test, source
check/test, source Agent IR and native execution stages. It preserves exact
commands, source/compiler hashes and raw diagnostics under
`sev_compiler/package.pkg/collection-ledger`. Unsupported stages fail strict mode;
`--record-only` explicitly requests an inventory rather than acceptance.
[MEASUREMENTS.md](MEASUREMENTS.md) records the measured compiler boundary.

The compiler package builds with the checkout seed and emits Agent IR. This
proves the existing compiler path still builds, **not that it consumes these
collections**; that path still uses `primitive/collections.sev` buffer aliases.

## Remaining language dependencies

Native collection execution is blocked, not marked passed by algorithm tests.
The source compiler currently rejects imported canonical primitive declarations
(`expected macro identifier`). The shared size helper uses explicit lossy conversion of -1 to compute the
target-width unsigned maximum. Its seed native test passes, including overflow
rejection. Collection generic/memory paths still require further work. The original list failed native seed compilation
before these edits; tree/set originally failed parsing.

P1/P5/P6 still need generic/constant layouts, canonical capability resolution
(including canonical trait conformance), checked typed allocation,
initialized slots, per-element move/drop, borrow-return indexing, and iterator
state/cleanup. List equality operations need method-specific Equatable checking;
vector shape and return-type-changing map need real compiler validation.

Mutation invalidating a live borrow/iterator must be rejected by the compiler.
The list extension snapshot addresses storage aliasing inside the algorithm;
it does not establish language loan safety. Arbitrary owned values, nested
containers, allocation-failure rollback and throwing comparisons need their
native ownership tests before removing Copy/Default restrictions. Set provider
fields/shape metadata need compiler-enforced encapsulation. Migration of actual
compiler consumers and deletion of the primitive list alias remain C3/C9 work.
