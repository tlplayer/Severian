# Collections

Start with storage and views, then growable ownership, then specialized collections.

| Example | Collection | Covers |
| --- | --- | --- |
| [01-array.sev](01-array.sev) | `array[T, N]` | Fixed-size storage, indexing, copying |
| [02-slice.sev](02-slice.sev) | `slice[T]` | Borrowed views and changes to the original storage |
| [03-list.sev](03-list.sev) | `list[T]` | Growth, capacity, insertion, removal, iteration |
| [04-deque.sev](04-deque.sev) | `deque[T]` | Adding and removing at both ends |
| [05-vector.sev](05-vector.sev) | `vector[T, N]` | Numeric lanes, arithmetic, scaling, dot product |
| [06-map.sev](06-map.sev) | `map[K, V]` | Lookup, replacement, ordered keys |
| [07-set.sev](07-set.sev) | `set[T]` | Uniqueness, membership, ordered values |
| [08-dict.sev](08-dict.sev) | `dict[K, V]` | Key/value lookup and defaults |
| [09-heap.sev](09-heap.sev) | `heap[T]` | Priority order, peeking, empty removal |
| [10-count.sev](10-count.sev) | `count[T]` | Frequencies, distinct and total counts, subtraction |

Each example includes a `main`, named test blocks, and expected output comments.
Types are inferred from constructor values or subsequent uses; the array example
also shows an explicit element type and length.
Tests include inferred mixed values, explicit unions such as `int | string`,
and `any` elements or mapped values, as well as strings and floating point lanes.
Methods returning `(value, found)` report whether a lookup or removal succeeded.
These examples cover the collection APIs being built and planned; compiler
support is still in progress.

Additional examples:

- [11-lists-tuples.sev](11-lists-tuples.sev)
- [12-maps-sets.sev](12-maps-sets.sev)
- [13-comprehension-style.sev](13-comprehension-style.sev)
- [14-shape-stable-indexing.sev](14-shape-stable-indexing.sev)
- [15-expressive-collections.sev](15-expressive-collections.sev)

See the [examples guide](../README.md) for validation commands.
