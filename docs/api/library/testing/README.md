# Testing library

Stable ID: `library.testing`.

The testing package provides reusable assertions, fixtures, generators, and
test utilities. Language `test`, `compiler accept/reject`, property, benchmark,
and differential syntax is owned by the parser/AST contract. The library
supplies ordinary functions used inside those declarations.

Compiler diagnostics, runtime assertions, and test-runner failures are distinct
error channels. A helper should preserve that distinction rather than turning
all failures into `panic`.
