# Hooks

API ID: `compiler.hook`

Hooks are structured compiler interception points with explicit context plus with/without phases. They are distinct from function contracts and foreign ABI declarations.

```sev
def hook_subject(value: int) -> int:
    return value
```

Current weakness: hook phase/input/output records need deeper per-hook documents.
