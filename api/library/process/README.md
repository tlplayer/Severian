# Process

API ID: `library.process`

The module exposes run, spawn, kill, wait, and process arguments. Source/API exhaustiveness is checked directly against `library/system/process/src/lib.sev`.

```sev
import process

def argument_count() -> usize:
    return size(process.arguments())
```

Current weakness: process handles are integers rather than opaque owned resources.
