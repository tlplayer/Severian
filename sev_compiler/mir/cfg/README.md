# Executable compiler CFG

`cfg.sev` exposes the compiler-owned execution context and block construction
operations. `block.sev` and `branch.sev` define graph storage, typed value IDs,
successor arguments and terminator provenance. `lazy.sev` carries source operands
without executing them; `effects.sev` and `control_flow.sev` provide compiler
contract vocabulary.

The source executable resolves a G declaration before entering its semantic
frame. The frame distinguishes source expressions/bodies, lowered values, block
handles and absence. CFG access requires a control-flow capability. Ordinary
runtime bodies cannot acquire the implicit cfg/lower context, including through
an operator's G constraint or a raw MLIR terminator declaration.

The current evaluator supports semantic bindings and typed compiler calls,
including block creation/selection, branches, jumps, loop contexts, returns,
block parameters and lowering source operands. It is not yet a general evaluator
for arbitrary Severian `valid`/`semantic` bodies or compiler-context helpers.

Executable bodies live in `Module.cfg_bodies`; each function has a `cfg_index`.
The initializer occupies index zero. Two old optional fields remain empty as
layout sentinels for the Rust seed's record/list ABI. No executable consumer
reads them. Removing them currently corrupts declaration records in the seed.

Construction rejects a second terminator or instructions after termination.
Verification checks block identity, all terminators, resolved grammar origins,
branch/return types, successor arity/types and value dominance. An unused block
created by a semantic body still requires a terminator. String storage is
promoted to SSA block parameters before verification and inspection.

MLIR and Agent IR iterate this exact table. The active storage-alias analysis
also uses it; full resource ownership and cleanup are still migration work.
