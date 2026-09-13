# Severian packages

This library owns package discovery, resolution, editing, builds, and
publication in ordinary Severian source. Its design follows
[SIP-0003: Package Interfaces, Realizations, and Incremental Dependency Builds](../../docs/sip/0003-sip-packages.md).
The callable API and compilation-unit protocol are documented in [API.md](API.md).

SIP-0003 is a draft. The contracts below describe its intended package model;
the implementation-status section records the remaining migration work.

## Package model

| Object | Role |
| --- | --- |
| `package.toml` | User-authored package identity, declared targets, dependencies, exports, and policy |
| `package.lock` | Generated exact dependency resolution, identities, and content verification data |
| `package.pkg/` | Generated realization of the package |
| `package.pkg/package.pkgi` | Generated public semantic interface consumed by imports |

A package is defined by one manifest. A declared target is a library or binary
produced by that package; a platform is a compilation destination. An artifact
is a compiled realization for a platform, backend, and build profile.

An illustrative source package has this layout:

```text
geometry/
├── package.toml
├── package.lock
├── src/
│   ├── lib.sev
│   └── math.sev
└── package.pkg/
```

The manifest's package name must be present and nonempty. Dependency keys are
local import aliases; published package identity comes from the declared
package name and resolved version. Path dependencies resolve relative to the
manifest that declares them. Source paths must stay within their package root.
Root-package tests may use development dependencies in addition to ordinary
dependencies.

The manifest in [this directory](package.toml) defines the `package` library
itself. The `geometry` examples describe a consumer package.

## Canonical generated layout

The SIP's summary and local-publication contract define these top-level roles:

```text
package.pkg/
├── package.pkgi
├── metadata/
├── artifacts/
├── build/
├── cache/
├── bin/
├── debug/
├── container/
└── source/
```

| Entry | Contents |
| --- | --- |
| `package.pkgi` | Exported declarations, stable symbols, types, callable signatures, constraints, effects, public layouts, ABI information, and implementation references |
| `metadata/` | Generated realization and publication metadata |
| `artifacts/` | Completed, reusable compiler and linker outputs, organized by platform, profile, or backend format |
| `build/` | Mutable incremental state for the working checkout; never published |
| `cache/` | Disposable intermediate and temporary execution data |
| `bin/` | Runnable package executables, with platform/profile subdivisions where needed |
| `debug/` | Test, coverage, profile, symbol, source-map, and compiler-mapping information |
| `container/` | Optional container realizations or construction metadata |
| `source/` | Optional source for rebuilding, specialization, debugging, or inspection |

For example:

```text
package.pkg/
├── artifacts/
│   ├── linux-x86_64/release/object/
│   └── portable/mlir/
├── bin/
│   └── linux-x86_64/release/geometry-tool
├── debug/
│   ├── test/<platform>/<build-profile>/<invocation>/
│   ├── coverage/<invocation>/
│   └── profile/<invocation>/
└── cache/
    ├── native/<invocation>/
    └── run/<invocation>/bin/
```

Test, coverage, and profile outputs belong beneath `debug/`. Standalone build executables belong
in `bin/`; temporary run executables and compiler intermediates belong in
`cache/`. Executables and intermediate files do not belong directly in the
`package.pkg/` root. Platform and backend classifications belong inside their
artifact category, rather than introducing additional top-level categories.

The published realization excludes `build/`. Source inclusion is optional;
an ordinary installed import must be usable without the dependency's source
or mutable build state. The complete source-free import path remains migration
work, as recorded below.

### `debug/test/`

Contains test invocation results: pass/fail summaries, diagnostics, captured
output, and failure details. Each invocation owns its directory so repeated or
concurrent runs do not overwrite one another. Retained test executables belong
in that invocation's `bin/` subdirectory.

### `debug/coverage/`

Contains coverage measurements and reports, including execution counters and
their mappings to source lines, branches, or functions. Reports identify the
source revision, build settings, and test invocation they describe so coverage
from incompatible builds is not combined.

### `debug/profile/`

Contains CPU, memory, and elapsed-time measurements, compilation-stage timings,
and detailed profiling traces such as sampled call stacks or allocation
records. Each invocation owns its reports and records the measured executable,
arguments, and build identity. This directory describes performance
measurements; build profiles such as `dev` and `release` remain build settings.

## Interfaces, exports, and imports

`package.pkgi` is generated from the canonical export model. It carries public
semantic information and stable declaration identities; consumers must not
need a dependency's private AST to type-check an import.

The SIP defines three export forms that converge on the same export set:

```sev
package.export(foo)

with package.export:
    class Point:
        x: float
        y: float
```

```toml
[package]
name = "geometry"
version = "1.2.0"
export = ["Point", "distance"]
```

These are design examples; convergence of the export forms and general package
interfaces is not yet complete. Duplicate identical exports collapse, and an
export referring to a missing declaration is an error.

The intended import flow is:

```text
import alias
  → package.lock
  → resolved installed package
  → package.pkg/package.pkgi
  → semantic namespace
```

An import establishes visibility. It does not fetch an undeclared dependency,
edit the manifest, or compile every implementation in the imported package.
Only reachable declarations require implementation lookup. A compatible cached
implementation is reused; missing or invalid implementations are compiled.

## Resolution and incremental builds

`package.lock` records exact identities, versions, sources, dependency edges,
and verification hashes. Explicit dependency operations update the manifest
and lock together. Normal compilation must not rewrite dependency resolution.
`sev add` resolves and records a dependency without compiling it.

Interface identity and implementation identity are separate. A private helper
change in dependency C must not force semantic rebuilds of consumers B and A
when C's consumed interface remains unchanged. An exported type change causes
its consumers to be reconsidered; invalidation propagates further only where
relevant interfaces or implementation dependencies change.

Implementation cache keys include implementation semantics, consumed interface
hashes, compiler version and ABI, platform, backend, profile, and relevant
options. Package timestamps alone do not establish cache validity.

The current compilation-unit cache verifies actual source input hashes, the
resolved graph, compiler identity, native tool versions, build settings, and
output digests. Its records and reusable outputs live in
`package.pkg/build/units/`. A unit is keyed by its own transitive graph, so
building another consumer or editing consumer code does not rebuild the
dependency unit. Temporary test/run destinations are restored from the same
cache. Cache lookup precedes prelude preparation, including standalone source
invocations. Missing, damaged, or changed inputs invalidate the entry; failed
compilation never marks an entry reusable. `build.incremental = false` bypasses
reuse. Concurrent invocations serialize writes to each cache entry.

Published packages remain immutable: local unit state for these dependencies
lives under `${SEVERIAN_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/severian}`
in `packages/build/units/`, shared by consumers. This implements compilation
unit reuse; declaration-level interface invalidation and source-free dependency
consumption remain separate steps.

## Package operations

The package standard library is the semantic authority for package operations.
The CLI, compiler, publisher, and tooling must use the same resolution and
export rules. The native source compiler already delegates package operations
to this library; the Rust seed still has bootstrap implementations to converge.

The current API can be used directly:

```sev
import package

project = package.open(".")
graph = package.resolve(project)
print(package.tree(graph))

compiler = package.Compiler("/path/to/sev_compiler", "/path/to/Severian")
result = package.build(project, compiler, package.Build(profile="release"))
report = package.test(project, compiler)
```

A compilation unit receives a versioned snapshot of resolved identities,
manifests, roots, and alias edges through `__compile-unit`. That compiler
boundary performs the requested compilation without resolving the graph again.
See [API.md](API.md) for request fields and result types.

Common native compiler commands are:

```sh
sev check
sev build --build-profile release
sev test
sev test --profile
sev test --profile cpu
sev test --profile memory
sev run
```

`--build-profile` selects build settings; `--profile` measures the invocation.
Default profiling reports use `package.pkg/debug/profiles/`. Test invocations
use `package.pkg/debug/tests/`, with their executables beneath `bin/`.
Standalone source builds use `package.pkg/bin/`; temporary runs and native
intermediates use `package.pkg/cache/`. Explicit build and profiling output
options can select another location.

In a directory without `package.toml`, `sev test` batches `.sev` files in that
directory and its subdirectories. A package directory uses its declared and
conventional package tests.

## Local registry and publication

SIP-0003 places local package storage under:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/severian/
├── registry/
│   ├── index/
│   └── packages/<name>/<version>/
│       ├── package.toml
│       ├── package.lock
│       └── package.pkg/
└── git/
    ├── checkouts/
    └── db/
```

`SEVERIAN_HOME` can select an isolated Severian root. Registry releases are
immutable published realizations; Git checkouts are independent package
sources. The publication command `sev publish <package> --local` accepts the current
manifest name, and `sev publish` publishes the current package. The current
release contents use the archive compatibility layout described below; the
SIP interface and storage migration remain incomplete.

Publication must preserve package identity, validate indexed content, and
exclude mutable `package.pkg/build/` state. Current publication stages a
snapshot and commits it without replacing an existing release. An identical
publication is a no-op; changed content at an existing version produces
`PackageVersionConflict`. Consumption verifies published content before use.

The current archive compatibility boundary is `SEVPKG`: version 1 contains
reachable library source; version 2 contains indexed metadata, optional source,
and compatible native artifacts. Archive entries have normalized relative
paths, deterministic ordering, content checksums, and executable flags.
Readers validate bounds, duplicate entries, and paths before extraction or
execution. Existing archive paths such as `metadata/sev.lock` and
`artifacts/<platform>/<profile>/bin/` are compatibility details pending
migration, rather than the canonical working-package layout above.

The current registry runner selects a compatible native binary or builds from
included source. General interface consumption, additional backend selection,
and container fallback must satisfy the SIP's compatibility rules before being
presented as implemented. Optional source, containers, and debug data do not
replace the semantic interface.

## Implementation status and migration

| Contract | Current status |
| --- | --- |
| Package operations in ordinary Severian source | Implemented; native source CLI delegates to this library |
| One package implementation for every consumer | Partial; Rust bootstrap package semantics remain separate |
| Manifest discovery, dependency aliases, and transitive resolution | Implemented |
| Transactional dependency edits and lock generation | Implemented using legacy `sev.lock` |
| Canonical `package.lock` name | Pending migration of APIs, compilers, archives, examples, and tooling |
| Lock enforcement | Implemented when requested through `--locked` or `Resolve(locked=true)` |
| Canonical `package.pkg/` top-level layout | Partial; package builds still write `<platform>/<profile>/bin` and `pkg` beneath `package.pkg/` |
| Test and profile output placement | Under `package.pkg/debug/`; current writers use plural `tests/` and `profiles/`, pending migration to canonical `test/` and `profile/` |
| Coverage output placement | Canonical destination is `package.pkg/debug/coverage/`; producer convergence remains migration work |
| Standalone source build/run and intermediate placement | Implemented beneath `bin/` and `cache/` |
| General `package.pkgi` and source-free installed imports | Pending; primitive interface records exist |
| Canonical export model across all SIP export forms | Pending convergence |
| Compilation-unit cache reuse | Implemented under `package.pkg/build/units/` |
| Declaration-level interface hashing and incremental invalidation | Pending |
| Local registry publish/consume and transitive dependency validation | Implemented with existing publication layout |
| SIP registry index and independent Git storage layout | Pending; current Git cache is under the registry cache |
| `SEVPKG` v1/v2 archive compatibility | Implemented; archive migration remains separate |
| Registry execution and machine-level installation | Implemented |
| Remote registry transport and authentication | Not implemented |
| General backend/container fallback and policy enforcement | Design contract |

The canonical names above define the destination of the migration. Legacy
paths remain implementation compatibility details until their replacements
work and all consumers have migrated. SIP-0003 requires replacement interfaces,
implementation lookup, and shared APIs before removing the corresponding
legacy paths or duplicate implementations.

## Validation and native boundaries

Hosted filesystem and process operations come from `os`, `file`, and `process`.
Their POSIX providers are declared in `library/system/package.toml` and selected
through package manifests. String processing and archive decoding remain in
Severian source.

Run the adjacent contracts with:

```sh
python3 library/system/tests/native.py
python3 library/package/tests/workspace.py
SEVERIAN_SOURCE_COMPILER=/path/to/sev python3 library/package/tests/publication.py
python3 tests/sev_compiler/artifact_layout.py
sev_rust test library/package
```

End-to-end publication is covered by
[registry publish/consume validation](../../test/validation/packages/registry_publish_consume.sh)
and [transitive dependency validation](../../test/validation/packages/registry_transitive_tensor.sh).
Changes to package semantics or output placement should update the relevant
contracts and this implementation-status table together.
