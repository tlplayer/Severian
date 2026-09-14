# list[T]

`list[T]` is the public, source-owned contiguous sequence. It owns list
semantics—construction, growth requests, indexing, insertion, removal,
iteration, and algorithms—but owns no pointers or allocator calls.

The required dependency boundary is:

```text
list[T]
   |
vector[T]
   |
array[T]
   |
core.memory
```

These are distinct collection types. `list[T]` must not alias `array[T]`.
The current list package still delegates to `ContiguousStorage[T]`; migrating
its operations and ownership behavior to `vector[T]` remains unfinished.
