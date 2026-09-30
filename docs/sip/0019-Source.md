SIP-0019: 

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

## Context

Source is simply the interface to raw .sev files. It carries span, paths, etc. Which is needed for knowing where errors occur and finding suitable fixes. 

## Data model

```sev
class SourceFile:
    id: SourceId
    name: string
    path: string
    text: string
    characters: list[string]
    # Derived lazily, shared by repeated tooling span queries, never archived.
    byte_offsets: list[int] = []

    def SourceFile(id: SourceId, path: string, text: string):
        self.id = id
        self.path = path
        self.text = text
        characters = text.characters()

class Source:
    files: 
    spans: string

class SourceId:
    index: u32

class Span:
    # Internal coordinates are half-open Unicode scalar indices. Serialized
    # tooling interfaces use UTF-8 bytes; convert explicitly at that boundary.
    source: SourceId
    start: u32
    end: u32

class SourceMap:
    files: list[SourceFile] = []

    def add_virtual(path: string, text: string) -> SourceId:
        id = SourceId(u32(len(files)))
        files.append(SourceFile(id, path, text))
        return id

    def load(path: string) -> SourceId | Error:
        return add_virtual(path, file.read_checked(path))

    def get(id: SourceId) -> SourceFile:
        return files[id.index]

```
