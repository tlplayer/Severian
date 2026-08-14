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
minimum = 99
regions = 99
branches = 99
functions = 99
per_file = true

[memory]
leaks = "deny"

[architecture]
enforce = true
deny_cycles = true
deny_unknown_layers = true
deny_layer_violations = true

[architecture.layers]
include = ["compiler/*"]
order = ["syntax", "semantic", "hir", "mir", "backend"]

[[architecture.rule]]
from = "compiler/backend/**"
allow = ["compiler/mir/**"]
deny = ["compiler/syntax/**", "compiler/semantic/**"]

[architecture.files]
soft_lines = 500
hard_lines = 800
include = ["src/**/*.sev", "tests/**/*.sev"]
```

The architecture pass builds dependency graphs from local `Cargo.toml`
production/build dependencies and `package.toml` dependencies. Tarjan strongly
connected components identify cycles and diagnostics include the concrete
dependency path and declaration line. Layer order is dependency order: a later
layer may depend on an earlier one, while the reverse edge fails even when it
does not form a cycle. The optional `include` patterns scope layer names when a
repository contains more than one package family. Matching
`[[architecture.rule]]` tables then apply deny lists and allow-list boundaries
to the resolved edges.

`sev check` and `sev build` enforce the same pass. `sev architecture` prints a
package/edge summary and high fan-out packages; `sev architecture --graph`
emits the resolved graph as standalone DOT on standard output.

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

The compiler repository requires 99% line, region, branch, and function
coverage from the aggregate report and from every individual Severian source
file. Per-file enforcement prevents a large, well-tested module from hiding a
regression in a smaller module. A metric with no executable regions is treated
as fully covered.

When a manifest omits coverage policy, the parser uses a 75% aggregate line
fallback. `sev init` and `sev new` write every available control and explicitly
start new applications at 0% with permissive memory and file-size policy. This
keeps a fresh project usable while making the intended ratchets visible in
`package.toml`; teams can raise them without researching hidden keys.
