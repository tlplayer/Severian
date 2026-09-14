# box[T]

`box[T]` owns one heap-allocated `T`. Its declaration lives in the primitive
prelude alongside `pointer[T]`; its behavior is written in Severian.
It does not require a default value or copyable payload.

| Operation | Contract | Rust counterpart |
| --- | --- | --- |
| `box[T](value)` or `box(value)` | Move `value` into a new allocation; infer `T` when omitted | `Box::new(value)` |
| `value.clone()` or `clone(value)` | Clone the payload into independent storage | `Box::clone` / `Clone` |
| `value.take()` | Consume the box, free its storage, return the owned payload | Moving out of a `Box` |
| `value.replace(replacement)` | Move in a replacement and return the old payload | `mem::replace(&mut *value, replacement)` |
| `drop(value)` or scope exit | Destroy the payload and release its allocation | `drop` / `Drop` |

```sev
first = box[int](42)
second = clone(first)
assert(second.replace(7) == 42)
assert(first.take() == 42)
assert(second.take() == 7)
```

A move transfers ownership. It never clones the allocation. `take()` consumes
the owner and subsequent use is rejected. A resource-owning payload must
provide its own `clone()` method to make its containing box cloneable.
The allocation pointer and ownership flag are private implementation fields.

## Compiler coverage

`python3 tests/sev_compiler/boxes.py` checks native execution with the Rust seed,
including destructor order, nested boxes, large payload layouts, cloning,
extraction, and compile-time rejection of invalid ownership operations.

The source prelude selects this declaration as well. Source-compiler execution
still depends on the unfinished prelude collection and ownership migration;
seed execution does not establish source-compiler parity.

Borrowed payload access will use Severian's `view` (shared) and `borrow`
(exclusive) names. These operations are not exposed yet: the seed currently
materializes a temporary for pointer loads and cannot return a checked loan
into that allocation. Raw ownership conversion, leaking, pinning, custom
allocators, and uninitialized boxes are also outside this initial API.
