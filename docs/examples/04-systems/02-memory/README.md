# Memory

Ordinary values use managed, ownership-checked storage. Raw allocation exists
for allocators, foreign interfaces, device buffers, and similar systems work,
and requires an unsafe scope.

This section covers:

- managed construction and deterministic destruction;
- raw allocation and exactly-once release;
- target-derived size, alignment, field offsets, and padding;
- resource cleanup on normal return and error propagation;
- compile-time rejection of raw allocation from safe code.

Layout queries describe the selected target. They do not let semantic layers or
backends reinterpret a source type independently.
