# file

`file.read(path)` selects a registered decoder from the file extension and
returns an object implementing `file.File`. The package includes `Text`,
`CSV`, `WAV`, `MP3`, and `Binary`; `read_text` remains the explicit API when a
caller wants an unwrapped string rather than format dispatch.

The interface and implementations are deliberately separate compilation
units:

```text
src/file_interface.sev  File and Reader traits
src/file_text.sev       Text and TextReader
src/file_csv.sev        CSV and CsvReader
src/file_wav.sev        WAV and WavReader
src/file_mp3.sev        MP3 and Mp3Reader
src/file_binary.sev     Binary and BinaryReader
src/lib.sev             public dispatch and general file operations
```

```sev
import file

document ?= file.read("people.csv")
print(document.kind())
print(document.rows)
```

Applications can extend the same interface without creating a parallel file
API. Define a class implementing `file.File`, implement decoding as the
`read()` member of a `file.Reader`, then register the reader object:

```sev
import file

class Playlist: file.File
    source_path: string
    tracks: list[string]

    def path() -> string:
        return source_path

    def extension() -> string:
        return ".m3u"

    def kind() -> string:
        return "playlist"

    def media_type() -> string:
        return "audio/x-playlist"

    def bytes() -> int:
        return size(tracks.join("\n"))

class PlaylistReader: file.Reader
    def read(path: string) -> Result[file.File, IOError | file.FormatError]:
        content ?= file.read_text(path)
        return Playlist(path, content.split("\n"))

def main():
    file.register(".m3u", PlaylistReader())
    playlist ?= file.read("music.m3u")
    print(playlist.get("tracks"))
```

Native operations remain isolated in `platform`, and failures stay explicit
through `Result`.
