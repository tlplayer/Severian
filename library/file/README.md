# file

Typed file operations backed by the explicit `platform` package. Native read
and write are covered by a compile-link-execute round-trip test, and `IOError`
is explicit through `Result`. Path manipulation and owned handles remain
future work.
