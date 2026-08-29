# Collections

API ID: `library.collections`

The catalogue covers array, deque, heap, list, map, queue, set, stack, builders, iterators, operations, and capability traits. `export_sources` is checked against exact top-level Severian symbols.

```sev
def collection_count(values: list[int]) -> usize:
    return size(values)
```

Current weakness: method-level contracts need generated pages even though the package symbol inventory is exhaustive.
