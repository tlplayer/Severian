# Ownership compatibility package

Ownership implementations and the MLIR ownership pipeline now live in
[`transforms/mir/ownership`](../../transforms/mir/ownership/README.md).
This package forwards old source imports. `mlir.pipeline` links to that owner.

Semantic construction still consumes MIR-owned effect/place/loan helpers during
the migration. The remaining phase separation is recorded in
[`PIPELINE_MIGRATION.md`](../../PIPELINE_MIGRATION.md).
