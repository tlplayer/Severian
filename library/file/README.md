# file

Typed file operations backed by the runtime. The initial `write` declaration
makes `IOError` explicit through `Result`. Owned handles and borrowed views are
still pending. Path manipulation belongs in the future `path` package.
