# Build policy

`sev build` is the release-quality path. It always executes every mandatory
gate before emitting artifacts:

```text
compile -> architecture -> test -> profile -> coverage -> memory -> integration
```

The CLI has no `--skip-*` or `--through` option. Individual commands such as
`sev test`, `sev coverage`, and `sev memory` remain useful diagnostics, but are
not weakened substitutes for a successful build. Incremental work should come
from reusable compiler artifacts and gate caches, never from changing which
checks count.

The complete policy is declarative in `package.toml`:

```toml
[build]
pipeline = [
    "compile",
    "architecture",
    "test",
    "profile",
    "coverage",
    "memory",
    "integration",
]

[coverage]
minimum = 75
branches = 60

[memory]
leaks = "deny"

[architecture.files]
soft_lines = 500
hard_lines = 800
include = ["src/**/*.sev", "tests/**/*.sev"]
```

A soft limit is visible debt; a hard limit fails the build. A temporary hard
limit exception must name the exact path and give a reviewable reason. Owners
and expirations make migration debt explicit:

```toml
[[architecture.files.exception]]
path = "src/lowering/expression.sev"
hard_lines = 1100
reason = "typed match lowering is being split by expression family"
owner = "compiler-lowering"
expires = "2026-10-15"
```

An expired exception fails the architecture gate even when the file remains
below its exceptional hard limit.

The compiler repository currently pins its own line floor to 55%, just below
the measured 57.50% baseline, while new packages inherit the 75% default. This
is progressive pressure: the existing workspace cannot regress, and the floor
can rise as coverage is added without weakening or skipping the gate.
