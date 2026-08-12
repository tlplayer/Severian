# Compiler error catalog

Each diagnostic has a stable code and a Severian source file under its primary
category. Filenames use `EXXX-description.sev` for errors and
`WXXX-description.sev` for warnings. The source itself is the documentation:
`#` comments explain why the compiler reports it, where detection happens, and
the smallest normal repair.

These files intentionally demonstrate rejected or warned-about programs. They
are documentation fixtures, not runnable examples.

Package boundaries may progressively require concrete types with:

```toml
[package]
type-safe = true
```

That mode rejects declarations which would silently default to `Any`. Explicit
`Any` remains available when a dynamic boundary is intentional.
