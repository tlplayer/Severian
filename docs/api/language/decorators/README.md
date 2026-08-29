# Decorators

API ID: `decorator.compile_policy`

Decorators attach compile policy and metadata. `compile`, `mlir`, `stablehlo`, `xla`, and `triton` are compile policies, never foreign ABI attributes; they select a route only after legality is established.

```sev
@compile("mlir")
def decorated_probe(value: i32) -> i32:
    return value + 1
```

The AST keeps decorator name and structured arguments. Current weakness: policy compatibility diagnostics need a generated target-capability reference.
