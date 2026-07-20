# file

Typed file operations backed by the runtime. The API will distinguish owned
handles from borrowed views and make I/O failures explicit through `Result`.
Path manipulation belongs in the future `path` package rather than this one.

