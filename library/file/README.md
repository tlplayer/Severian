# file

`file.read(path)` is Severian's single path-reading entry point. It selects a
reader by extension and returns the document type owned by the corresponding
format package:

```sev
import file

people = file.read("people.csv")
print(people.headers())
print(people.rows()[0].get("name"))

settings = file.read("settings.json")
print(settings.get("voice"))
```

Use `file.load(path)` when only the decoded value is needed. It uses the same
extension dispatch without exposing the path-backed document wrapper:

```sev
settings = file.load("settings.json")
rows: list[list[string]] = file.load("people.csv")
```

For JSON this is the parsed JSON value (an object, array, scalar, or null), not
the source text. Use `file.read()` when document methods such as `write()`,
`raw()`, or format-specific mutation are required.

Literal paths refine statically from the same reader registry metadata used at
runtime. Dynamic paths retain the `file.File` interface and use trait dispatch
for its common methods; the compiler contains no built-in extension catalog.

## Ownership

The packages do not duplicate one another:

- `file` owns content objects, the `File` and `Reader` contracts, extension
  dispatch, and binary/text/audio adapters.
- `os` owns namespace operations and path metadata through `os.stat()`.
- `csv` owns `CSV`, `CSVRow`, parsing, quoting, mutation, and encoding.
- `json` owns `JSON`, typed/in-memory decoding, mutation, and encoding.
- `yaml` owns `YAML`, mapping access, mutation, and encoding.
- `file_csv.sev`, `file_json.sev`, and `file_yaml.sev` are thin reader adapters
  which connect extensions to those document packages.

Text-family reads are implemented by the `file` package's own `c-v1` native
provider. `abi` defines the stable types and ownership descriptors at that
boundary; `ffi` defines library and symbol lookup. The compiler only validates
and lowers the resulting typed foreign call.

In-memory and file-backed values therefore share one type:

```sev
import csv
import file

memory = csv.document("name,note\nAda,compiler\n")
disk = file.read("people.csv")

assert(memory.kind() == disk.kind())
```

Every path-backed object remembers its source path, so writes are symmetric:

```sev
config = file.read("settings.json")
config := config
config.set("voice", "belinda")
_saved = config.write()
```

## Extension readers

A format package implements its document and a small `file.Reader` adapter.
Reachable readers contribute extensions to the closed trait registry:

```sev
import file

class PlaylistReader: file.Reader
    extensions = {".m3u", ".m3u8"}
    document_class = "Playlist"

    def read(path: string) -> Result[file.File, IOError | file.FormatError]:
        content = file.source_text(path)
        return Playlist(path, content.split("\n"))

```

The `Playlist` document structurally implements `file.File`; its parsing and
domain methods remain owned by the playlist package. `file.source_text()` goes
directly to the package-owned text provider, preventing recursive
`file.read()` calls.

`extensions` drives generated runtime dispatch, while `document_class` lets a
literal path retain the provider's concrete result type during semantic
analysis. Both values are closed compile-time trait properties; no reader mutates
a global table at startup.

`read_text` and `parse_csv` remain compatibility shims. New code uses
`file.load()` for decoded values, `file.read()` for path-backed documents, and
the relevant format package for in-memory data.
