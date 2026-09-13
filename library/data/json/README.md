# JSON

`json` owns JSON documents and codecs for values from memory, HTTP, tools, or
other streams. File-path dispatch is integrated through `file.read("data.json")`,
which delegates parsing and document behavior to this package.

`decode(source)` returns `Json | Error`. The current Data adapter accepts an
object or an array of objects. Nested arrays/objects retain their JSON text in
table cells; numbers and booleans retain the table's existing textual cell
representation. Invalid syntax and malformed Unicode escapes return an error.
The hosted NUL-terminated string ABI rejects an escaped NUL rather than
silently truncating it.

The native provider is `native/json.c`. A document owns decoded keys, cell text,
and rows through core.storage; output accessors retain the requested owners and
the decoder releases the temporary document before returning. Column discovery
and row decoding share one record traversal. `string(document)` and
`print(document)` use the document's JSON representation.

`tests/native_json.c` checks decoding, Unicode, malformed input, and zero retained
storage after repeated parses. The library performance runner writes its JSON
allocation/timing report under `package.pkg/debug/profile/`.
