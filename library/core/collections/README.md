# Core collections

This directory is an organizational namespace, not one aggregate package.
Every public collection has its own manifest and can evolve independently.

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
