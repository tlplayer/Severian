# Core utilities

`core.utils` owns general prelude utilities that apply across value types, such
as assertion reporting. It is also the intended home for utilities such as
`zip`, `map`, and `sum` when their implementations are extracted. This package
currently implements assertion reporting and hashing; it does not yet implement
those other utilities.

The prelude selects these declarations; compiler lowering only arranges the
assertion call and termination operation. String-specific behavior belongs to
`core.string`, including `core.string.format` and `core.string.regex`.

Assertion reporting uses the shared IO text writer. IO owns UTF-8 byte transfer,
partial writes, and descriptor errors; the utility does not inspect storage.
Reporting is best effort and returns the original condition even if IO fails.

Collection constructors remain in collection packages, numeric functions in
`core.math`, and stream operations in IO. Borrowing, type resolution, syntax,
and intrinsic lowering remain compiler responsibilities. The prelude's function
catalog is not a claim that every listed function has a library implementation.

## Hash dispatch

`hash[F: HashFunction = StableHashFunction](value: string) -> u128` selects a
hashing strategy by type. Implement `HashFunction` on a default-constructible
class and call `hash[YourHashFunction](value)` to select it.

The default is FNV-1a over UTF-8 bytes with modular 128-bit multiplication,
matching the Rust bootstrap stable IDs. It includes embedded zero bytes.
This is deterministic, non-cryptographic hashing.

An ordinary `hash(value)` overload selects the same strategy for the Rust seed,
which currently discards function type-parameter defaults. The hash algorithm
and custom dispatch remain ordinary library code.
