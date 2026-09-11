# SIP-0000: Package Interfaces, Realizations, and Incremental Dependency Builds

Status: Draft

Type: Package

Authors:

Created: 2026-09-11

Target: Package system

Supersedes:

Superseded by:

## Summary

Severian packages use three primary package files:

```text
package.toml
package.lock
package.pkg/
```

`package.toml` is the user-authored package definition.

`package.lock` records the exact resolved dependency graph.

`package.pkg/` is the generated realization of the package.

Its top-level structure is fixed:

```text
package.pkg/
├── package.pkgi
├── metadata/
├── artifacts/
├── build/ #incremental builds 
├── bin/
├── debug/
├── container/
└── source/
```


`package.pkgi` is the semantic interface consumed by imports.

```sev
import tensor
```

resolves `tensor` through `package.lock`, then loads the installed package's:

```text
package.pkg/package.pkgi
```


Implementation artifacts are loaded only when reachable code requires them.

The package system  supports:

```text
sev publish package_name --local
sev add package_name
```

`sev add` updates `package.toml` and `package.lock` but does not compile the added package.

Packages may declare exports through Severian code:

```sev
package.export(foo)

with package.export:
    class Tensor:
        ...

    def zeros(...):
        ...
```

or through `package.toml`:

```toml
[package]
export = ["Tensor", "zeros"]
```

Both forms update one canonical package export model.

The package standard library SHALL be the source of truth for package operations. CLI commands delegate to the same APIs.

---

# Motivation

Severian needs package boundaries that support:

* fast incremental builds;
* source-free dependency consumption;
* local package installation;
* portable and native artifacts;
* containers;
* executables;
* debugging information;
* optional source;
* stable package interfaces;
* declaration-level reachability.

The compiler should not rebuild eight packages because one application package changed.

Likewise:

```sev
import foo
```

should not imply:

```text
compile foo
```

An import only establishes that the compiler may refer to `foo`'s public interface.

Compilation should occur only when implementation is required and no compatible cached artifact exists.

---

# Goals

This SIP must:

1. Define the canonical package layout.

2. Make `package.pkgi` the semantic import boundary.

3. Separate package resolution from package compilation.

4. Separate package interface changes from implementation changes.

5. Avoid compiling unused packages.

6. Avoid compiling unused declarations within used packages.

7. Reuse previously compiled artifacts whenever their semantic inputs remain valid.

8. Support local package publication and installation.

9. Make `package.name` mandatory and non-empty.

10. Make `sev add` update dependency state without compiling dependencies.

11. Make the standard `package` library own package semantics.

12. Support exports declared in source and manifest.

13. Make those export mechanisms converge on one representation.

14. Support packages whose source is unavailable.

15. Support native, GPU, portable, container, and executable realizations without adding arbitrary top-level directories.

---

# Non-goals

This SIP does not define:

* a public package registry protocol;
* registry authentication;
* package signing;
* remote mirrors;
* container runtime implementation;
* the exact binary encoding of `.pkgi`;
* the exact internal format of every artifact;
* whole-program LTO;
* automatic dependency downloading during ordinary compilation.

---

# Canonical package layout

A source package:

```text
geometry/
├── package.toml
├── package.lock
├── src/
└── package.pkg/
```

The generated package realization:

```text
package.pkg/
├── package.pkgi
├── artifacts/
├── bin/
├── debug/
├── container/
└── source/
```

These six entries form the top-level package realization contract.

## `package.pkgi`

Contains the public semantic package interface.

Used for:

* import resolution;
* exported names;
* types;
* function signatures;
* traits;
* constraints;
* effects;
* ABI information;
* public layouts;
* generic interface information;
* dependency identities;
* implementation references.

It does not contain unrelated private implementation state.

---

## `artifacts/`

Contains compiler/linker-consumable implementation artifacts.

Examples:

```text
artifacts/
├── linux-x86_64/
│   └── release/
│       ├── object/
│       └── library/
├── rocm-gfx1101/
│   └── release/
│       └── hsaco/
└── portable/
    ├── mlir/
    ├── llvm/
    └── stablehlo/
```

`native`, `gpu`, and `portable` are artifact properties.

They SHALL NOT become additional package-level directories.

`artifacts/` may contain:

* `.o`;
* `.a`;
* `.so`;
* LLVM;
* MLIR;
* StableHLO;
* HSACO;
* CUBIN;
* generic implementation material;
* cached declaration implementations.

---

## `bin/`

Contains runnable package executables.

Example:

```text
bin/
└── linux-x86_64/
    └── release/
        ├── sev
        └── server
```

Executables are separated from general artifacts so tooling can locate runnable outputs without understanding backend artifact formats.

---

## `debug/`

Contains optional development information.

Examples:

* symbols;
* source maps;
* debug metadata;
* profiling metadata;
* compiler mappings;
* IR mappings.

Debug information does not participate in normal import resolution.

---

## `container/`

Contains container realizations or container metadata produced for the package.

Example:

```text
container/
└── linux-x86_64/
    └── release/
```

Containers are package realizations.

They are not the semantic definition of the package.

---

## `source/`

Contains optional source bundled with the package.

Possible uses:

* rebuilding;
* generic specialization;
* source debugging;
* inspection;
* fallback implementation generation.

A consumer MUST NOT require `source/` merely to import a normal published package.

---

# Rejected top-level names

The following SHALL NOT be top-level package directories:

```text
target/
build/
native/
gpu/
portable/
metadata/
cache/
```

`target/` and `build/` do not describe what they contain.

`native`, `gpu`, and `portable` classify artifacts and therefore belong under `artifacts/`.

Package metadata belongs in:

```text
package.toml
package.lock
package.pkgi
```

rather than another metadata hierarchy.

---

# `package.toml`

Example:

```toml
[package]
name = "geometry"
version = "1.2.0"
edition = "2026"

export = [
    "Point",
    "distance",
]

[dependencies]
math = "1.4"
```

## Package name

A package name SHALL NOT be blank.

Invalid:

```toml
[package]
name = ""
```

Diagnostic:

```text
PackageNameError: package name cannot be empty
```

A missing package name SHALL also be rejected once this SIP is fully implemented.

---

# `package.lock`

`package.lock` is generated from `package.toml`.

It records exact resolution.

Example logical representation:

```toml
lock-version = 1

[[package]]
name = "geometry"
version = "1.2.0"
source = "local"
content-hash = "..."
interface-hash = "..."

dependencies = [
    "math@1.4.2",
]

[[package]]
name = "math"
version = "1.4.2"
source = "local"
content-hash = "..."
interface-hash = "..."
```

The lockfile records enough information to reproduce package identity and verify installed realizations.

Normal compilation SHALL NOT change `package.lock` unless dependency resolution is intentionally requested.

---

# Package interface

A logical `.pkgi` contains:

```text
PackageInterface {
    format_version
    package_identity
    interface_hash

    exports[]
    dependencies[]
}
```

Each exported declaration contains information such as:

```text
Export {
    symbol_id
    name
    namespace
    kind

    type
    generic_parameters
    constraints
    effects

    layout
    abi

    interface_hash
    implementation_key
}
```

Example:

```text
package geometry@1.2.0

export class Point
    x: float
    y: float

export def distance(
    left: Point,
    right: Point
) -> float
```

The interface is generated.

Users do not manually edit `.pkgi`.

---

# Import semantics

Given:

```sev
import geometry
```

the compiler performs:

```text
current package
      ↓
package.lock
      ↓
geometry@1.2.0
      ↓
installed package
      ↓
package.pkg/package.pkgi
      ↓
semantic namespace
```

The import SHALL NOT:

* parse dependency source;
* compile dependency source;
* execute dependency code;
* rebuild dependency artifacts;
* automatically modify `package.toml`.

If the package is not declared:

```text
PackageDependencyError:
'geometry' is not a declared dependency

help: sev add geometry
```

---

# Reachability

Importing a package does not imply using its implementation.

Example:

```sev
import tensor
import collections

def main():
    value = tensor.zeros[float](1024)
```

Only the required implementation graph is reachable:

```text
main
 ↓
tensor.zeros[float]
 ↓
required tensor operations
 ↓
required allocator/collection operations
```

The compiler SHALL NOT compile unrelated exported declarations merely because their package is imported.

---

# Incremental compilation

Severian must distinguish:

```text
package changed
```

from:

```text
package interface changed
```

and:

```text
used implementation changed
```

Suppose:

```text
A → B → C
```

and C changes a private helper.

Expected:

```text
C implementation changes
C package.pkgi unchanged

B semantic rebuild: no
A semantic rebuild: no
```

If C changes an exported type used by B:

```text
C interface changes
 ↓
B is reconsidered
```

If B's public interface remains unchanged, A need not necessarily be invalidated.

---

# Artifact build key

Compiled implementation artifacts SHOULD be keyed by semantic inputs rather than package timestamps.

Conceptually:

```text
BuildKey =
    hash(
        implementation identity,
        implementation semantics,
        consumed interface hashes,
        compiler version,
        compiler ABI,
        target platform,
        backend,
        profile,
        relevant options
    )
```

If the key already exists:

```text
reuse
```

Otherwise:

```text
compile
```

---

# Export semantics

## Direct export

```sev
package.export(foo)
```

Example:

```sev
def foo() -> int:
    return 42

package.export(foo)
```

---

## Export block

```sev
with package.export:

    class Point:
        x: float
        y: float

    def origin() -> Point:
        return Point(0.0, 0.0)
```

Declarations introduced directly inside the block become public package exports.

---

## Manifest exports

```toml
[package]
export = [
    "Point",
    "origin",
]
```

---

# Canonical export model

All export mechanisms feed one representation:

```text
source package.export(...)
          │
source with package.export
          │
package.toml export list
          │
          ↓
      ExportSet
          ↓
    package.pkgi
```

No consumer may independently determine package visibility.

Duplicate identical export declarations collapse.

Invalid names are errors.

Example:

```text
PackageExportError:
package.toml exports 'Triangle',
but no package declaration named 'Triangle' exists
```

---

# Package API

Package operations SHALL be available through Severian code.

Example:

```sev
import package

project = package.open(".")
package.add(project, "tensor")
```

Possible package API surface:

```sev
package.open(...)
package.add(...)
package.remove(...)
package.update(...)
package.resolve(...)
package.lock(...)
package.build(...)
package.publish(...)
package.install(...)
package.export(...)
```

The CLI wraps these operations.

There SHALL NOT be separate CLI and standard-library package semantics.

---

# `sev add`

Command:

```text
sev add tensor
```

Flow:

```text
tensor
 ↓
resolve package metadata
 ↓
select package version
 ↓
update package.toml
 ↓
update package.lock
 ↓
done
```

`sev add` SHALL NOT compile the dependency.

It SHALL NOT compile the current package.

It SHALL NOT execute dependency code.

After:

```text
sev add tensor
```

the manifest contains:

```toml
[dependencies]
tensor = "..."
```

and the lock records the exact version.

---

# `sev publish`

Local publication:

```text
sev publish geometry --local
```

If the manifest contains:

```toml
name = "geometry"
```

the name must match.

If an old/default package has no valid name and the command provides one, publication tooling may update the manifest during migration.

A conflicting explicit package name is an error.

Example:

```text
PackageIdentityError:
publish name 'geometry2'
does not match package.toml name 'geometry'
```

---

# Local package store

Default Linux user package location:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/severian/packages/
```

Example:

```text
~/.local/share/severian/packages/
└── geometry/
    └── 1.2.0/
        ├── package.toml
        ├── package.lock
        └── package.pkg/
            ├── package.pkgi
            ├── artifacts/
            ├── bin/
            ├── debug/
            ├── container/
            └── source/
```

Published versions SHOULD be immutable.

The store root may be overridden for development and testing.

---

# Source of truth

Source of truth:

```text
library/package
```

Author inputs:

```text
package.toml
source export declarations
```

Resolved dependency state:

```text
package.lock
```

Public semantic representation:

```text
package.pkg/package.pkgi
```

Compiled implementation representation:

```text
package.pkg/artifacts/
```

Executable representation:

```text
package.pkg/bin/
```

Optional realizations:

```text
package.pkg/debug/
package.pkg/container/
package.pkg/source/
```

Consumers:

```text
compiler
CLI
LSP
build system
publisher
installer
debugger
```

No consumer should recreate package semantics independently.

---

# Compiler impact

## Lexer

Changes:

None required if package operations use ordinary identifiers, calls, imports, and `with`.

Removed behavior:

Any package-export-specific tokenization should be avoided.

---

## Parser

Changes:

No special package grammar is required for:

```sev
package.export(...)
```

or:

```sev
with package.export:
```

Removed behavior:

Any duplicated special package export syntax.

---

## Semantic

Changes:

* imports resolve against `.pkgi`;
* exported declarations produce canonical package symbols;
* package exports feed `ExportSet`.

Removed behavior:

* dependency source parsing for installed imports;
* duplicated package visibility rules.

---

## HIR

Imported declarations should carry stable identities:

```text
PackageSymbolId
InterfaceHash
ImplementationKey
```

HIR should not need the dependency's original AST.

---

## MIR

Only reachable implementations enter MIR.

Import alone MUST NOT materialize the implementation of every exported declaration.

---

## MLIR / backend

The backend requests implementation by stable implementation identity.

It should:

```text
lookup cached artifact
 ↓
reuse if valid
 ↓
compile only if unavailable or invalid
```

---

## Runtime

No package runtime work occurs merely because an import was resolved.

---

## ABI

Public ABI-relevant information must be represented in `.pkgi`.

An ABI-affecting change changes the relevant interface hash.

---

# Tooling

## CLI

Required:

```text
sev add <package>
sev remove <package>
sev update <package>
sev publish [package] --local
sev package tree
sev package interface
```

`sev package interface` SHOULD provide a human-readable view of `.pkgi`.

---

## LSP

The LSP may use `.pkgi` for:

* completion;
* signature help;
* imported type information;
* go-to-definition fallback;
* documentation.

Source is optional.

---

## Debugger

Debug information resolves through:

```text
package.pkg/debug/
```

---

# Compatibility

## Source compatibility

Existing:

```sev
import foo
```

remains valid.

---

## Package compatibility

Old:

```text
sev.lock
target/
```

New:

```text
package.lock
package.pkg/
```

The migration should be mechanical.

---

# Migration plan

## Stage 0 — Define behavior

Add tests for:

* package names;
* package lock;
* export semantics;
* `.pkgi` generation;
* `.pkgi` import;
* source-free imports;
* no-codegen imports;
* dependency cache reuse.

---

## Stage 1 — Canonical package model

Introduce:

```text
PackageIdentity
PackageLock
PackageInterface
PackageSymbolId
ExportSet
ImplementationKey
```

Exit criteria:

All can be constructed and serialized independently.

---

## Stage 2 — Generate `.pkgi`

Every importable library package generates:

```text
package.pkg/package.pkgi
```

Exit criteria:

The interface contains enough information to semantically consume the package without source.

---

## Stage 3 — Consume `.pkgi`

Installed dependency imports transition from:

```text
dependency source
```

to:

```text
package.pkgi
```

Exit test:

Delete the installed dependency's `source/`.

Consumer still compiles.

---

## Stage 4 — Standardize `package.pkg/`

All package outputs move beneath:

```text
package.pkg/
```

with exactly:

```text
package.pkgi
artifacts/
bin/
debug/
container/
source/
```

at the top level.

---

## Stage 5 — Replace `sev.lock`

Rename:

```text
sev.lock
```

to:

```text
package.lock
```

Update package APIs, examples, compiler logic, and tooling.

---

## Stage 6 — Package API convergence

Move:

```text
add
remove
update
resolve
lock
publish
install
```

to the canonical `package` library.

CLI implementations become wrappers.

---

## Stage 7 — Reachability compilation

Implement:

```text
import
 ↓
interface only
 ↓
reachable declaration
 ↓
implementation lookup
 ↓
cache reuse or compile
```

Exit criteria:

Imported-but-unused packages generate no implementation work.

---

## Stage 8 — Incremental invalidation

Add declaration/package interface hashing.

Prove:

```text
private dependency change
```

does not rebuild unrelated consumers.

---

## Stage 9 — Remove legacy paths

Delete:

* `sev.lock`;
* top-level `target/`;
* dependency source import resolution;
* whole-package codegen triggered by import;
* duplicate package resolvers;
* duplicate package manifest editing.

---

# Deprecation and cleanup plan

| Deprecated                    | Replacement              | Removal condition              |
| ----------------------------- | ------------------------ | ------------------------------ |
| `sev.lock`                    | `package.lock`           | All consumers migrated         |
| top-level `target/`           | `package.pkg/artifacts/` | All build producers migrated   |
| source-based installed import | `package.pkgi`           | Interface complete             |
| package-wide codegen          | reachability codegen     | implementation lookup complete |
| CLI package resolver          | package library          | CLI delegates fully            |
| duplicate export logic        | `ExportSet`              | all export forms migrated      |

---

# Files / flows to delete

After migration, remove any implementation whose sole purpose is:

```text
parse dependency source to discover exports
```

Remove old build-path handling for:

```text
target/
```

Remove compatibility handling for:

```text
sev.lock
```

once the migration window ends.

Remove duplicate CLI-specific:

* resolver;
* manifest editor;
* lock generator;
* publisher.

---

# Tests

## Compile-positive

Package:

```sev
with package.export:
    def square(value: int) -> int:
        return value * value
```

Consumer:

```sev
import math_package

test:
    assert(math_package.square(4) == 16)
```

Expected:

Passes through `.pkgi`.

---

## Compile-negative

```sev
import missing
```

Expected:

```text
PackageDependencyError:
'missing' is not a declared dependency

help: sev add missing
```

---

## No-source test

1. Publish a library.
2. Remove its `source/`.
3. Keep `.pkgi` and compatible artifacts.
4. Compile consumer.

Expected:

Pass.

---

## No-use test

```sev
import huge_package

def main():
    print("hello")
```

Expected:

```text
interface reads: >= 1
implementation compilations: 0
linked symbols: 0
```

---

## Private-change test

Dependency:

```sev
with package.export:
    def public() -> int:
        return helper()

def helper() -> int:
    return 1
```

Change:

```sev
return 1
```

to:

```sev
return 2
```

Expected:

```text
dependency implementation changed: yes
dependency interface changed: no
consumer semantic rebuild: no
```

---

## Export equivalence

These:

```sev
package.export(foo)
```

```sev
with package.export:
    def foo():
        ...
```

and:

```toml
[package]
export = ["foo"]
```

must generate equivalent canonical export information.

---

# Performance requirements

| Metric                                                         | Required |
| -------------------------------------------------------------- | -------: |
| Imported unused dependency implementation compilations         |        0 |
| Unchanged dependency recompilations                            |        0 |
| Dependency source parses during installed import               |        0 |
| Consumer semantic rebuild after private dependency-only change |        0 |
| `sev add` compiler invocations                                 |        0 |

Add benchmark package graphs of:

```text
1
8
32
128
```

dependencies.

Repeated builds should scale primarily with changed reachable work, not total package count.

---

# End-to-end example

Library:

```text
geometry/
├── package.toml
└── src/
```

```toml
[package]
name = "geometry"
version = "0.1.0"
```

```sev
import package

with package.export:

    class Point:
        x: float
        y: float

    def origin() -> Point:
        return Point(0.0, 0.0)
```

Build:

```text
package.pkg/
├── package.pkgi
├── artifacts/
├── bin/
├── debug/
├── container/
└── source/
```

Publish:

```text
sev publish geometry --local
```

Application:

```text
sev add geometry
```

Application manifest becomes:

```toml
[dependencies]
geometry = "0.1.0"
```

Application:

```sev
import geometry

def main():
    point = geometry.origin()
    print(point.x)
```

Resolution:

```text
application/package.lock
        ↓
geometry@0.1.0
        ↓
geometry/package.pkg/package.pkgi
        ↓
geometry.origin
        ↓
implementation_key
        ↓
geometry/package.pkg/artifacts/...
        ↓
link
```

Expected:

```text
0.0
```

---

# Implementation plan

## Change 1

Scope:

Define canonical package model and fixed `package.pkg/` layout.

Tests:

Layout validation.

---

## Change 2

Scope:

Generate complete `.pkgi`.

Tests:

Public/private symbol tests.

---

## Change 3

Scope:

Consume `.pkgi` for installed imports.

Tests:

No-source dependency test.

---

## Change 4

Scope:

Implement canonical export set.

Tests:

Source/manifest export equivalence.

---

## Change 5

Scope:

Replace `sev.lock` with `package.lock`.

Tests:

Lock migration and deterministic regeneration.

---

## Change 6

Scope:

Move build outputs into:

```text
package.pkg/artifacts/
package.pkg/bin/
package.pkg/debug/
```

Tests:

Artifact discovery.

---

## Change 7

Scope:

Implement:

```text
sev publish <package> --local
sev add <package>
```

against the package library.

Tests:

Publish from one package and add/import from another.

---

## Change 8

Scope:

Add reachability-driven implementation lookup.

Tests:

Unused package and unused exported declaration tests.

---

## Change 9

Scope:

Add semantic fingerprints and artifact reuse.

Tests:

Private implementation changes do not invalidate consumers.

---

## Change 10

Scope:

Delete compatibility architecture.

Tests:

Search for old names and full regression suite.

---

# Alternatives considered

## No `.pkgi`

Rejected because dependency source would remain necessary for semantic import resolution.

## Compile complete package on import

Rejected because import does not imply implementation reachability.

## `target/`

Rejected because it communicates only that the compiler produced something.

The contents have useful categories and SHALL instead live under:

```text
artifacts/
bin/
debug/
container/
source/
```

## `native/`, `gpu/`, `portable/` at package root

Rejected because these describe kinds of artifacts, not package-level concerns.

They belong under `artifacts/`.

## Separate `.pkgi` beside `package.pkg`

Rejected because the interface is part of the package realization and should not have independent lifecycle management.

---

# Risks

| Risk                                           | Impact                            | Mitigation                                                   |
| ---------------------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `.pkgi` lacks required semantics               | Source fallback remains necessary | No-source test                                               |
| Interface hash changes unnecessarily           | Rebuilds remain excessive         | Canonical hashing                                            |
| Artifact cache becomes too granular            | Complexity                        | Start declaration-level                                      |
| Generic specialization needs source            | Missing implementation            | Portable implementation representation or optional `source/` |
| CLI duplicates package semantics               | Split architecture                | CLI/library equivalence tests                                |
| Package layout starts accumulating directories | Unclear contract                  | Freeze six top-level entries                                 |

---

# Open questions

1. Binary or textual `.pkgi` representation?

2. Exact placement of generic implementation templates beneath `artifacts/`.

3. Exact distinction between portable generic implementation and target-specialized implementation.

4. Whether remote publication includes `debug/` and `source/` by default.

These do not change the fixed package root.

---

# Completion criteria

This SIP is implemented when:

* `package.name` cannot be blank.
* `package.lock` replaces `sev.lock`.
* `package.pkg/` has the fixed top-level contract.
* `.pkgi` is generated for importable packages.
* installed imports resolve exclusively through `.pkgi`.
* dependency source is unnecessary for normal semantic import.
* `sev add` performs no dependency compilation.
* `sev publish --local` installs a reusable package realization.
* source and manifest exports converge on one `ExportSet`.
* imports compile only reachable implementation.
* unchanged implementation artifacts are reused.
* private dependency changes do not rebuild consumers unnecessarily.
* CLI package operations delegate to the standard package library.
* legacy package flows are removed.
* full regression suite passes.

---

# Decision record

2026-09-11 — Draft

Defined the package interface, package realization, fixed package directory layout, local package publication, source-defined exports, and reachability-driven package compilation.

---

# Final architecture

```text
package.toml
      │
      ├───────────────┐
      ↓               ↓
 dependency intent   exports
      │               │
      ↓               ↓
 package.lock      ExportSet
      │               │
      └───────┬───────┘
              ↓
         package.pkg/
              │
      ┌───────┼──────────────────────────────────────┐
      ↓       ↓          ↓        ↓          ↓       ↓
 package.pkgi artifacts/ bin/    debug/   container/ source/
      │
      ↓
    import
      │
      ↓
reachable exported symbols
      │
      ↓
implementation keys
      │
      ↓
cached artifact or compilation
      │
      ↓
link / execute
```

## Fixed top-level realization

```text
package.pkg/
├── package.pkgi
├── artifacts/
├── bin/
├── debug/
├── container/
└── source/
```

## Removed

```text
sev.lock
target/
source parsing for installed imports
compile-whole-package-on-import
duplicate package resolution
duplicate package export semantics
duplicate CLI package mutation logic
```

## Core invariant

> Import reads the interface. Reachability requests implementation. Fingerprints decide whether compilation occurs.
