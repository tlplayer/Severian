# Control

API ID: `control.conditional`

Conditional expressions, `if`, `while`, `for`, `match`, `select`, `break`, `continue`, return, assertions, and deferred execution form the structured control API.

```sev
def control_probe(value: int) -> int:
    if value < 0:
        return -value
    return value
```

MIR lowers this to explicit blocks and edges while retaining source spans. Current weakness: select/limit scheduling has less conformance coverage than ordinary branches and loops.
