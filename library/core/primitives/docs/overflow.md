# Overflow

Fixed-width integer operations use the overflow behavior selected by the
language contract and compilation mode. Backend lowering must preserve that
decision and must not infer overflow policy from a source type name.
