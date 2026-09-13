# SIP-0003: Package Interfaces, Realizations, and Incremental Dependency Builds

Status: Draft

Type: Package

Authors: Timothy Player

Created: 2026-09-11

Target: Package system


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
├── package.pkgi/ # Interop files 
├── metadata/ # Information needed to run the crate and link objects 
├── artifacts/ # completed reusable compiler outputs
├── build/ # incremental compiler state NOT EXPORTED TO CLIENTS
├── cache/ 
├── bin/ #runables 
├── debug/ #debug is not published
├── debug/profile/
├── debug/test/
├── debug/coverage/
├── container/
└── source/ # Source is published
```



```sev
import tensor
```

resolves `tensor` version through `package.toml`, then loads the installed package's:

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


# Versions

the package's version: 0.1.0:commit:hash:time
The hash is the hash of the package so that if the files change/the source is different we can do a diff of
the commits for when it breaks without constantly tweaking/poluting the version number
Or something like that

# Local package registry

Severian stores locally published packages in the user's XDG data directory:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/severian/
```

Canonical layout:

```text
~/.local/share/severian/
├── registry/
│   ├── index/
│   └── packages/
│       └── geometry/
│           └── 1.2.0/
│               ├── package.toml
│               ├── package.lock
│               └── package.pkg/
│                   ├── package.pkgi/
│                   ├── metadata/
│                   ├── artifacts/
│                   ├── cache/
│                   ├── bin/
│                   ├── debug/
│                   ├── container/
│                   └── source/
└── git/
    ├── checkouts/
    └── db/
```

`registry/` is the package source used by:

```text
sev init #create new package with package.toml filled with all options and their defaults/other values as comments
sev build #populates the values int he package.pkg/
sev publish <package> --local
sev add <package>
sev update <package>
```

`registry/index/` contains package/version discovery information.

`registry/packages/` contains immutable published package realizations.

`git/` contains packages resolved directly from Git repositories and is independent from registry-published packages.

## Local publication

```text
sev publish geometry --local
```

publishes:

```text
geometry:1.2.0
```

to:

```text
~/.local/share/severian/registry/packages/geometry/1.2.0/
```

The published package contains:

```text
package.toml
package.lock
package.pkg/
```


```text
package.pkg/build/
```

`build/` exists in the working package:

```text
geometry/
├── package.toml
├── package.lock
├── src/
└── package.pkg/
    ├── package.pkgi/ #artifacts for interop 
    ├── metadata/
    ├── build/
    ├── cache/
    ├── artifacts/
    ├── bin/
    ├── debug/
    ├── container/
    └── source/ #mandatory
```


`build/` describes how the current checkout incrementally produced its outputs. It is not a package artifact and is never published.

## Package resolution

Given:

```text
sev add geometry
```

the package system searches:

```text
registry/index/
```

selects the requested version, records it in:

```text
package.toml
package.lock
```

and resolves imports against:

```text
registry/packages/geometry/1.2.0/package.pkg/package.pkgi/
```

Normal import resolution does not load the package's source or build state.

## Overrides

`SEVERIAN_HOME` MAY override:

```text
~/.local/share/severian/
```

For example:

```text
SEVERIAN_HOME=/tmp/severian-test
```

produces:

```text
/tmp/severian-test/
├── registry/
└── git/
```

This is intended for tests, containers, CI, and isolated development environments.


# Updates

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
---

# Goals

This SIP must:

1. Define the canonical package layout.

2. Builds are cached and do not recompile if no files have changed in their modules

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

2. Make `package.pkgi` the semantic import boundary for non severian consumers
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
├── metadata/
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

Contains source of the program src/ just goes there and is needed for compiling



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

A logical `.pkgi` contains roughly:

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
package geometry:1.2.0

export class Point
    x: float
    y: float

export def distance(
    left: Point,
    right: Point
) -> float
```

The interface is generated.


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
package.toml
      ↓
geometry:1.2.0
      ↓
installed package
      ↓
package.pkg/package.pkgi
      ↓
semantic namespace
```


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
