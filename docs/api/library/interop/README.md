# Interop libraries

Stable IDs: `library.interop.abi` and `library.interop.ffi`.

ABI contracts define versioned data layouts and calling conventions. FFI
contracts bind external symbols using those layouts. Neither may pretend an
opaque pointer is a typed MLIR tensor. Storage crosses through a descriptor;
compute kernels receive specialized ranked values or explicit pointer/shape/
stride arguments.

Ownership, nullability, alignment, lifetime, unwind, and error propagation must
be explicit at every exported boundary. A successful symbol lookup is not
evidence that these obligations are satisfied.
