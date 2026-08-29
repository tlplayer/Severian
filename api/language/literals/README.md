# Literals

API ID: `literal.kind`

The surface mirrors `universal::LiteralKind`: integer, float, boolean, character, string, bytes, `None`, and unit. Spelling survives parsing until semantic analysis selects a representation-compatible type; dtype is not encoded in the literal operation identity.

```sev
def literal_probe() -> bool:
    return 42 > 0 and 3.5 > 0.0 and "text" != "" and true
```

Lowering creates a representation-correct constant only after type selection. Current weakness: measured literal suffix algebra remains documented by primitive pages rather than one generated dimension table.
