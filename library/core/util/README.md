# Core utilities

`core.util` owns general prelude utilities that apply across value types, such
as assertion reporting. It is also the intended home for utilities such as
`zip`, `map`, and `sum` when their implementations are extracted. This package
currently implements assertion reporting; it does not yet implement those
other utilities.

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
