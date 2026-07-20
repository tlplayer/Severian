# math

Portable numeric algorithms live in Severian source. Operations that map to a
hardware or MLIR intrinsic, such as `sqrt`, must be declared by a typed
compiler/runtime interface rather than recognized by a spelling check.

The initial source API provides `square`, `cube`, and `clamp` for floats.

