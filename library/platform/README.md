# platform

`platform` is the explicit typed boundary between Severian packages and native
operating-system services. Its declarations name their linker symbols; there is
no implicit `runtime` identifier and no source-level compiler escape hatch.

Application packages normally import `file`, `json`, `log`, `network`, or
`regex`. Those packages depend on `platform`, keeping native ABI details below
their public APIs while preserving ordinary package resolution and type checks.
