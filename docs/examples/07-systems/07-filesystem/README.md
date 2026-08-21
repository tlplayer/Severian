# Filesystem

Filesystem operations are separated from lexical path manipulation. The `path`
library can join, normalize, and inspect paths without touching the host. The
filesystem provider performs directory traversal, metadata queries, copying,
renaming, and removal.

Mutating examples are integration tests. They must create unique temporary
directories and clean them up even when an assertion fails; fixed shared names
under `/tmp` are illustrative only and must not be used by the executable test
harness.

Errors preserve the operation, path, and provider error category. Boolean
helpers such as `exists` are conveniences and do not erase errors from
operations that need diagnostic detail.
