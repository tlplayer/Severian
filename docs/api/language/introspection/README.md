# Introspection

API ID: `introspection.type`

Compile-time type, shape, and capability inspection belongs here. Introspection must query structural metadata without turning dtype, rank, or shape into source-name dispatch.

```sev
def introspection_subject[T](value: T) -> T:
    return value
```

The intended boundary is the semantic type/constraint context. Current weakness: this area is specified but does not yet expose a complete stable source-level query API.
