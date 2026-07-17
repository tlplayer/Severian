# Aliasing mutation

An immutable borrow prevents conflicting mutation until its last use. This lets
the compiler preserve stable references without hidden locking or copying.

Mitigation: **compile-time**. End the borrow before mutating, create a snapshot
with `clone()`, or use an explicitly synchronized shared type.
