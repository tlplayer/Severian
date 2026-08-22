# list[T]

`list[T]` is the public, source-owned contiguous sequence. It owns list
semantics—construction, growth requests, indexing, insertion, removal,
iteration, and algorithms—but owns no pointers or allocator calls.

The dependency boundary is:

```text
list[T]   array[T]   deque[T]
    \         |         /
     core.collections.internal.contiguous_storage[T]
                         |
                    core.memory
                         |
              allocator/runtime/platform
```

Concrete collections do not depend on one another. The internal storage
package is private and contains no list-specific API.
