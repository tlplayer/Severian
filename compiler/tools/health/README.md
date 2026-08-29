# Compiler-stage health integration

Repository analysis cannot reconstruct compiler invariants reliably from Rust
source. Each compiler stage owns its verifier and reports failures through the
shared health vocabulary.

## Current enforcement

MIR exposes `PassContract` with:

```rust
pub struct PassContract {
    pub requires: InvariantSet,
    pub preserves: InvariantSet,
    pub establishes: InvariantSet,
    pub may_remove: EntitySet,
    pub may_introduce: EntitySet,
}
```

The MIR pass manager checks requirements before execution, rejects preservation
claims for unavailable invariants, snapshots entity counts, rejects undeclared
additions/removals, invalidates analyses not explicitly preserved, and invokes
MIR verification after a pass claims to establish or preserve well-formedness.
The required pipeline establishes well-formed MIR, elaborated drops, valid
ownership, and lowering readiness in that order.

MLIR already parses and verifies generated artifacts, validates legal dialects
for the target, checks entry signatures, verifies composed modules, and verifies
GPU launcher modules. Those checks remain in the MLIR boundary because a
repository lint cannot replace dialect verification.

## Required stage contracts

Future stage integrations must report the same structured finding fields and
must verify these facts at the owner:

- AST: source spans, unique declaration IDs, and no leaked recovery nodes.
- HIR: resolved names/types/operators/effects and retained provenance.
- Ownership: moves, borrows, resource lifetimes, and stable identities.
- MIR: CFG targets, terminators, use-before-definition, call signatures,
  ownership state, registered operations, and operation verification.
- Compile planning: exactly one route, target capability support, retained
  shape bounds, and no premature ABI decisions.
- LIR/MLIR: source locations, symbols, signatures, legal dialects, custom
  operation verification, and no illegal operations after lowering.

Tests belong beside each verifier. Transform tests must include invalid input,
valid output, deterministic repeated execution, and idempotence where the pass
claims canonicalization.
