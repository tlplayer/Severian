# File

API ID: `library.file`

The public module includes file types, overloaded writes, raw text, handle reads, mappings, and locks. File I/O and resource lifecycle are explicit effects.

```sev
import file

def load_text(path: string) -> string:
    return file.raw(path)
```

Current weakness: native handles are integers rather than opaque owned resources.
