# Ownership authority

This library owns the source contract for views, mutation, transfers,
initialization, lifetimes, and destruction. Its executable analyses are written
in Severian. `effects.sev` classifies parameter effects, `loans.sev` checks access and authorizes explicit release,
`places.sev` validates initialization and selects initialized cleanup,
`cfg.sev` verifies SSA availability, and `plan.sev` authorizes lowering of the
current graph. Semantic analysis and MIR are clients of this authority.

## Contract

- Ordinary arguments begin as views. Reading an argument grants no authority to
  release it. Exclusive loans permit mutation; consuming boundaries transfer
  responsibility explicitly. Existing source effect inference is represented in
  `effects.sev`, rather than inferred again by an emitter or native runtime.
- Initialization belongs to places, not allocation capacity. Uninitialized and
  moved fields must not be read, retained, or destroyed. Cleanup visits the
  remaining initialized contents, after any custom hook.
- A view carries the identity of its owner through aliases and control flow. Its
  owner cannot be released or replaced while that view remains live. A view of
  local stack storage cannot escape the function.
- Lowering must obtain a fresh plan after transformations. Diagnostics retain
  source spans. A function name or native pointer is not ownership authority.
- Source-owned buffers stay as memrefs through the ownership pipeline. Allocation,
  copying and aliasing use standard MLIR operations. The standard deallocation
  pass determines physical releases; source checking determines whether the
  requested lifetime and transfer are legal.

`mlir.pipeline` is the shared executable pass contract consumed by both compiler
backends. It structures control flow, resolves buffer ownership, and lowers
bufferization operations before memrefs become LLVM descriptors/pointers. The
canonicalization between control-flow lifting passes simplifies exit switches
introduced by lifting, so the ownership pass receives supported control flow.
The Rust backend runs this after tensor bufferization.

The [MLIR ownership ABI](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/)
keeps argument ownership with the caller and transfers returned buffer ownership
to the caller. Private functions may use MLIR's dynamic ownership extension.
Neither C adapters nor hand-written reference counts replace this buffer model.

`core.storage` owns initialized contents above `core.memory`'s physical
allocation interface. The hosted C implementation remains as a compatibility
provider while equivalent source implementations are validated and callers
migrate. See [the audit](AUDIT.md) for remaining boundaries; passing the buffer
pipeline does not verify ownership hidden in opaque bootstrap runtime pointers.
