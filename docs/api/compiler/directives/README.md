# Directives

API ID: `compiler.directive.compile`

Compile policies and target directives select legal lowering routes only after semantic and backend legality. They cannot silently change source semantics or ABI types.

```sev
@compile("mlir")
def directed_add(left: i32, right: i32) -> i32:
    return left + right
```

Current weakness: target-policy compatibility is not yet rendered as a generated table.
