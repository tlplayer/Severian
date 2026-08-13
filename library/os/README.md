# OS

`os` owns filesystem namespace operations and path metadata. Use `file` for
contents, typed documents, streams, mappings, and locks.

```sev
import os

if os.exists(path):
    information = os.stat(path)
    print(information.size, information.modified_seconds)

entries = os.ls(directory)
```

`stat()` is the metadata boundary. Callers should not assemble metadata from
separate size and timestamp functions.

The concise command vocabulary and descriptive names are aliases over the same
operations: `ls`/`list`, `mkdir`/`make_directories`, `rm`/`remove`,
`mv`/`rename`, and `cp`/`copy`.
