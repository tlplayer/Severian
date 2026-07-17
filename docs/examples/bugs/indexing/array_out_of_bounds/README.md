# Array out of bounds

The compiler rejects a statically known bad index. Dynamic indexing remains
bounds-checked; APIs that may miss should return `Option[type]` instead of exposing
unchecked indexing.

Mitigation: **compile-time** for provable failures and **runtime** for dynamic
indices.
