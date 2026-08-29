# Compile types

API ID: `compiler.compile_type`

Compile routes and compiler-extension type identities select lowering contracts without becoming source runtime types or dtype/rank suffixes.

```sev
def compile_subject(value: i32) -> i32:
    return value + 1
```

Current weakness: the complete extension-type catalogue is not yet emitted as generated semantic IDE metadata.
