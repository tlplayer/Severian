# Type identity

Every primitive has the stable identity of its bootstrap declaration. Source
annotations and inferred literals resolve through the same catalog and produce
the same `PrimitiveId` and interned `TypeId`. HIR, MIR, and lowering preserve
those identities; downstream passes do not reconstruct them from names.
