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

Literal paths refine statically: `.csv` is `csv.CSV`, `.json` is `json.JSON`,
`.yaml` is `yaml.YAML`, and `.txt` is `file.Text`. Runtime paths retain the
`file.File` interface and use dynamic trait dispatch for its common methods.

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
Readers advertise their extensions and can be registered together:

```sev
import file
import platform

class PlaylistReader: file.Reader
    def extensions() -> list[string]:
        return [".m3u", ".m3u8"]

    def read(path: string) -> Result[file.File, IOError | file.FormatError]:
        content = platform.file_read(path)
        return Playlist(path, content.split("\n"))

def install():
    file.register_reader(PlaylistReader())
```

The `Playlist` document structurally implements `file.File`; its parsing and
domain methods remain owned by the playlist package. Low-level reads used by a
reader adapter stay in `platform`, preventing recursive `file.read()` calls.

`read_text` and `parse_csv` remain compatibility shims. New code uses
`file.load()` for decoded values, `file.read()` for path-backed documents, and
the relevant format package for in-memory data.
