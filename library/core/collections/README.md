# Core collections

The [implementation plan](IMPLEMENTATION_PLAN.md) defines the migration to a
re-exporting `collections` package with static provider arguments such as
`set[T:int,C:BTree](values...)`. The inventory below describes the current
packages, whose APIs have not yet completed that migration.

The root package provides the collection facade. Individual providers have
their own manifests and can evolve independently.

The root `.sev` files contain the collection contracts and algorithms relocated
from the compiler: list, fixed-lane vector, deque, heap, map, dict, B-tree, set,
count, and shared size/growth helpers. They are imported directly by source path
and retain their existing APIs and inline tests. Their primitive and operator
contracts still come from `sev_compiler/universal`.

These source modules and the provider packages below have different APIs;
reconciling them remains part of the [implementation plan](IMPLEMENTATION_PLAN.md).
The [source algorithm plan](SOURCE_IMPLEMENTATION_PLAN.md),
[capability inventory](CAPABILITIES.md), and [measurements](MEASUREMENTS.md)
record their current implementation and compiler support.

```text
core.memory
    ↑
internal.contiguous_storage
    ↑          ↑          ↑          ↑
  list       array      deque      stack/heap
                         ↑
                       queue

internal.contiguous_storage
    ↑
internal.hash_table
    ↑                 ↑
   set               map
```

Rules:

- Public collections never allocate or manipulate pointers directly.
- `list`, `array`, and `deque` do not depend on one another.
- `set` and `map` share private associative storage but never wrap each other.
- Queue depends on deque because FIFO behavior fundamentally uses a deque.
- Internal packages are not user-facing APIs.
- Generic reads borrow elements; removal transfers ownership to the caller.

Current packages:

- `list[T]`: growable indexed sequence
- `array[T]`: bounded fixed-capacity sequence
- `deque[T]`: double-ended sequence
- `stack[T]`: LIFO adaptor over private storage
- `queue[T]`: FIFO adaptor over deque
- `heap[T: Ordered]`: binary min-heap
- `set[T: HashKey]`: unique-value collection
- `map[K: HashKey, V]`: keyed collection
