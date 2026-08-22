# Contiguous storage

Private move-aware storage shared by `list[T]`, `array[T]`, and `deque[T]`.
It owns growth and element relocation, but delegates raw allocation and pointer
access to `core.memory`.

It is not a user-facing collection and must not acquire list-specific APIs.
