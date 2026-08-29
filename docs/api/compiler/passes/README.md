# Pass contracts

Stable IDs: `compiler.pass.*`.

Every MIR pass declares required, preserved, and established invariants plus
the entity classes it may add or remove. The runner checks the precondition,
runs the pass, checks undeclared structural changes, invalidates analyses not
preserved, and verifies MIR after a pass claims well-formedness.

An `IrStage` is a pipeline position. An `Invariant` is a fact. They are related
but not synonyms: stage checking prevents gross ordering errors, while the
invariant set explains why the next pass is legal. `AnalysisId` identifies
cached derived information and is separate from both.

Current limitation: entity auditing records class counts, so a future stable
entity-delta representation is still needed to prove a remove-and-replace pair
within one class.
