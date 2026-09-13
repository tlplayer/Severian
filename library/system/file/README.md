# File

`system.file` depends on `system.io`. File operations own paths, open modes,
permissions and metadata. IO owns descriptor transfers; XXI owns the conversion
from caller-owned storage to foreign ABI views.

`file.read(path)` dispatches `.txt` and `.sev` to `system.file.text`, and registered
data formats to their codecs. JSON, CSV and YAML codecs obtain text from the
same text module before decoding. `file.map(path)` returns bytes, including NUL.
`file.write` accepts text or byte lists. `metadata`, `permissions`, and
`set_permissions` expose filesystem metadata and the rwx mode bits.

`src/text.sev` contains the UTF-8 algorithm, with no C bindings. `src/streams.sev`
opens the path, calls IO, and closes the descriptor on success and failure.
`extern/posix/file.c` supplies the platform open-mode and permission operations.
There is no whole-text C provider or separate bootstrap string adapter.

The source compiler's typed buffer path uses `src/path_buffer.sev` for path
marshalling and `system.io/src/buffer.sev` for bytes. Its public `@file` dispatch
parser limitation remains separate; the Rust seed runs dispatch and codec tests.
