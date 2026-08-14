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

Successful gates are recorded under `target/build-gates`. Their fingerprint
includes all Severian sources and manifests in the policy root, the selected
input, the compiler executable identity, and the host platform. An unchanged
gate prints `CACHED`. When a gate must run, its old result and every downstream
result are invalidated before execution; an interrupted build can therefore
resume only from gates whose inputs and successful result still match. Build
outputs, virtual environments, editor metadata, and dependency directories are
excluded from the fingerprint.

The complete policy is declarative in `package.toml`:

```toml
[build]
# Source-oriented diagnostics are the default. Use `internal` when debugging a
# compiler or backend failure; `--diagnostics` overrides this per invocation.
diagnostics = "user"
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
the measured 57.50% baseline. When a manifest omits coverage policy, the parser
still uses a 75% fallback. `sev init` and `sev new` now write every available
control and explicitly start new applications at 0% with permissive memory and
file-size policy. This keeps a fresh project usable while making the intended
ratchets visible in `package.toml`; teams can raise them over the project's
lifetime without researching hidden keys.
