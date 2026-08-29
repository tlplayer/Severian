# Errors

API ID: `error.preserve`

Error unions, `throw`, propagation, preservation with `?=`, fallback, `try`/`catch`, and fallible `else` form one typed error surface. Preservation retains the complete union instead of propagating its error member.

```sev
def error_subject(value: int) -> int:
    return value
```

HIR preserves union and control-flow identity through ABI lowering. Current weakness: exhaustive error-layout documentation is not generated for every union.
