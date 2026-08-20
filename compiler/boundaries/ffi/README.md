# FFI Boundary

Semantic validation and marshalling contracts for values crossing between
Severian and foreign implementations.

This crate owns external function/type declarations after source resolution,
ownership and lifetime contracts, parameter modes, nullability, semantic-to-ABI
conversion plans, and ABI selection. It consumes universal `TypeId`s and emits
concrete `severian-abi` signatures.

It does not parse `@c`/`@rust` attributes and does not implement target calling
conventions. Those belong to XXI and ABI respectively.
