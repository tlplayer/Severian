# SIP-0003: Package Interfaces, Realizations, and Incremental Dependency Builds

Status: Draft
Type: Package
Authors: Timothy Player
Created: 2026-09-11
Updated: 2026-09-13
Target: Package system

## 1. Contract and compilation boundary

This SIP specifies required behavior; it does not assert implementation status.
It supersedes source-optional publication and older XDG registry layouts.

- `package.json` defines the package, targets, exports, dependencies, and options.
- `package.lock` pins exact dependency identities and the resolved graph.
- `package.pkg/` is the generated package realization directory.
- `package.pkg/package.pkgi/` is a DIRECTORY containing compiler interfaces/interop files.
- Source MUST ship in every published package, including native provider sources.
- Compatible dependencies MUST reuse metadata and compiled implementations.
- Source availability MUST NOT cause dependency recompilation on every import.
- `library/package` owns these semantics; CLI commands delegate to its APIs.

A package  (library, binary, test, or host macro) is a compilation boundary.
Source modules organize declarations inside that target. Importing a module does
not create another dependency build. Compiler-selected codegen units may produce
multiple objects; they need not correspond one-to-one with source files.

Exports declared with `package.export(foo)`, `with package.export:`, or
`[package].export` MUST populate one canonical export model. Published metadata
includes supporting private declarations needed for public layouts and generics.

### Registry, publication, and commands

| Command | Required behavior |
| --- | --- |
| `sev init` | Create package definition with all supported options, defaults and commented alternatives; provide the standard package pipeline |
| `sev build` | Resolve existing locked graph, reuse/build targets, populate working realization |
| `sev check` | Load/produce metadata without requiring native codegen; mark metadata-only results as non-linkable |
| `sev test` | Reuse dependencies, build test targets locally, run with explicit resource/time bounds |
| `sev publish <name>:version --local` | Validate name, source completeness and selected completed artifacts; exclude build/cache/debug; install atomically |
| `sev add <name>:version` | Resolve and atomically update manifest/lock without compiling the dependency |
| `sev update <name>` | Intentionally refresh selected dependency resolution; subsequent build evaluates freshness |
| `sev file.sev` | Use explicit dependencies or compatible published packages/prelude from any directory |
| `sev clean` | Remove selected local generated state; preserve authored source and registry payloads |

Export packages MUST NOT contain a metadata-only check result masquerading as a
linkable library. Native artifacts may be omitted for an unsupported target only
when the publication explicitly declares source-rebuild availability for it.

## 2. Working package and local build layout

```text

#Build/ABI Id is in the format YYYY-MM-DD-HH-min-commit(a12c2a30)-hash-source
The ID is used to determine if recompilation is needed/not from a quick glance

geometry/
├── package.json
├── package.lock
├── src/
└── package.pkg/
    ├── package.pkgi/
    │   ├── index              # Available interfaces and binding sets
    │   ├── severian/<build-id>/interface.bin
    │   ├── c/<abi-id>/geometry.h
    │   ├── rust/<abi-id>/Cargo.toml
    │   ├── rust/<abi-id>/src/lib.rs
    │   └── python/<abi-id>/geometry.pyi
    ├── metadata/
    │   └── realizations/<build-id>
    ├── artifacts/<target-triple>/<profile>/<build-id>/
    │   ├── object/
    │   ├── archive/
    │   ├── dynamic/
    │   └── ir/
    ├── build/<build-id>/
    │   ├── dep-info/
    │   ├── incremental/
    │   └── staging/
    ├── cache/
    ├── bin/<target-triple>/<profile>/<build-id>/
    ├── debug/
    │   ├── symbols/<build-id>/
    │   ├── profile/<build-id>/
    │   ├── test/<build-id>/
    │   └── coverage/<build-id>/
    ├── container/<target-triple>/<build-id>/
    └── source/
        ├── package.json
        ├── package.lock
        ├── src/
```

Only declared binding languages and artifact kinds need entries. Empty optional
directories may be omitted. `<build-id>` distinguishes configurations that have
the same target triple and profile label but different effective inputs.

`source/` is a consistent snapshot of owned sources, headers, resources, and
declared build inputs, preserving relative paths. Its manifests agree with the
metadata snapshots. Generated inputs require either their contents or the source
and declared recipe needed to reproduce them. Dependency sources belong to their
own locked packages; system SDKs and tools are explicit external requirements.

## 4. File formats and roles

All stored paths are package-relative, normalized UTF-8 paths. Readers reject
absolute paths and traversal outside the package. Linker arguments use structured
records, not executable shell fragments. Symlinks, if supported, remain internal.

```sh
interface.sev       REQUIRED
    compiler-facing public semantic interface

.o / .obj            BUILD PRODUCT
    native codegen units

.a / .lib            DEFAULT REUSABLE STATIC IMPLEMENTATION
    archive of native codegen units

.so/.dylib/.dll      OPTIONAL DYNAMIC IMPLEMENTATION
    generated only when requested

.mlirbc              OPTIONAL SPECIALIZATION IMPLEMENTATION
    generics/JIT/target specialization where native code isn't sufficient

executable           BINARY TARGET
    output of main.sev
```

## 5. Concrete metadata objects

`Digest` is a 32-byte SHA-256 value, displayed as 64 lowercase hexadecimal digits.
Strings are UTF-8. Ordered sequences and keyed maps have explicit schemas.
The records below are logical data types, not compiler in-memory struct layouts.

| Object | Required fields |
| --- | --- |
| `PackageIdentity` | `origin: string`, `name: string`, `version: string` |
| `SourceRevision` | `package: PackageIdentity`, `content_id: Digest`, `commit: string?`, `published_at: UTC timestamp` |
| `DependencyRef` | `alias: string`, `revision: SourceRevisionRef`, `target: string`, `features: list[string]` |
| `SymbolRef` | `revision: SourceRevisionRef`, `declaration: Digest` |
| `CompilerIdentity` | `compiler_build: string`, `metadata_version: u32`, `mir_version: u32`, `severian_abi: string` |
| `TargetSpec` | `triple: string`, `cpu: string`, `features: list[string]`, `data_layout: string`, `runtime_abi: string` |
| `FileRecord` | `path: string`, `digest: Digest`, `bytes: u64`, `executable: bool`, `role: enum` |
| `ArtifactRecord` | `id: Digest`, `format: enum`, `path: string`, `bytes: u64`, `build_id: Digest`, `members: list[ArtifactRef]` |
| `Realization` | `build_id: Digest`, `revision: SourceRevisionRef`, `target: TargetSpec`, `compiler: CompilerIdentity`, `interface: ArtifactRef`, `artifacts: list[ArtifactRef]`, `dependencies: list[RealizationRef]` |
| `LinkRequirement` | `kind: static/dynamic/framework`, `artifact_or_system_name: string`, `symbols: list[string]`, `ordering: list[ArtifactRef]`, `whole_archive: bool` |
| `RuntimeRequirement` | `provider: string`, `abi: string`, `loader_name: string?`, `search_policy: enum`, `initialize: SymbolRef?`, `finalize: SymbolRef?` |
| `GenericInstanceKey` | `template: SymbolRef`, `substitution: list[GenericArgument]`, `build_context: Digest` |

`SourceRevisionRef` contains only PackageIdentity and ContentId, excluding commit
and publication time. `AbiId` hashes the canonical foreign contract and TargetSpec;
it identifies a binding ABI, not the implementation's source or build revision.
`TypeRef`, `ArtifactRef` and `RealizationRef` identify indexed types, ArtifactIds and
BuildIds respectively. Package origin is a logical source identity, not a checkout path.
`GenericArgument` is a tagged type reference, canonical constant value, or other
supported generic parameter; specialization keys include every effective argument.
Archive members use `(archive: ArtifactRef, member_name: string)` locators when
loose object files are omitted. Member names are unique within a published archive.
Each generated TOML index/record declares `schema_version: u32`; readers reject
unsupported major schemas rather than treating unknown requirements as optional.

`metadata/source-index.json` owns source `FileRecord`s. `metadata/artifacts.json`
owns generated payload `ArtifactRecord`s, including interfaces and bindings.
`metadata/realizations/<build-id>.json` owns the `Realization`, link requirements,
runtime requirements, and entry points. References point to these records;
consumers do not infer dependency identities or ABI contracts from filenames.

System-library requirements distinguish build-time linker names, runtime loader
names, and ABI constraints. CPU features are requirements, not descriptive labels.
Rust, C, Python, LLVM and MLIR provider/toolchain identities are recorded separately.
For cross compilation, host macro/build tools have host realizations and libraries
have target realizations; neither is reused for the other by package name alone.

## 3. Published package layout

Publication includes completed payloads and source, excluding local state:

```text
~/.severian/packages/registry/geometry/1.2.0/<content-id>/
├── index.json                           # Revision/realization discovery
└── realizations/<publication-id>/
    └── package.pkg/
        ├── package.pkgi/                # Same interface/binding layout as above
        ├── metadata/                   # Snapshots, inventories, link/run records
        ├── artifacts/<target-triple>/<profile>/<build-id>/
        │   ├── object/
        │   ├── archive/
        │   ├── dynamic/
        │   └── ir/
        ├── bin/<target-triple>/<profile>/<build-id>/
        ├── container/<target-triple>/<build-id>/
        └── source/
            ├── package.json
            ├── package.lock
            ├── src/
```

`index.json` entries are installed atomically; existing realization payloads are
immutable. Adding a realization replaces the index atomically, retaining older
entries. The term immutable revision refers to source identity and payloads,
not to the discoverability index's bytes.

| Directory | Role | Published? |
| --- | --- | --- |
| `package.pkgi/` | Severian declarations, interop contracts, generated bindings | Yes |
| `metadata/` | Identity, source/artifact inventories, dependency/link/run requirements | Yes |
| `artifacts/` | Completed reusable implementations and specialization material | Selected compatible sets |
| `build/` | Dependency tracking, compiler incremental state, staging and locks | Never |
| `cache/` | Disposable local decoding, discovery and download accelerators | Never |
| `bin/` | Runnable application binaries and launchers | Declared runnable targets |
| `debug/` | Symbols, profiling, test and coverage outputs | Never |
| `container/` | OCI image layouts or immutable image descriptors | Declared container targets |
| `source/` | Complete owned source/build-input snapshot | Always |

Test executables and instrumentation reports belong in `debug/test/` and the
other debug subdirectories. Debug-only artifacts are excluded even if another
directory contains them. Release binaries MUST NOT require excluded debug files.

Default storage is `~/.severian/packages/`. `SEVERIAN_HOME` overrides
`~/.severian`; `SEVERIAN_REGISTRY` overrides the registry directory directly.

- `registry/index/<name>.json`: version/revision/realization discovery and provenance.
- `registry/<name>/<version>/<content-id>/`: revision index and published payloads.
- `build/<content-id>/<build-id>/package.pkg/`: local realizations rebuilt from
  published source; published payloads are never modified to accommodate a client.
- `git/db/` and `git/checkouts/`: Git source acquisition, separate from publications.

Publications with the same source but different targets/toolchains have distinct
realizations. Republishing the same PublicationId is a no-op. Registry updates
publish verified payloads before atomically exposing their index entries.
Consumers use indexes or explicit paths; they do not scan the whole filesystem.
Package-relative source paths and exact dependency identities permit relocation.
Checkout-only dependency locations are discovery hints, not published build paths.

## 6. Compiler interface and interop contracts

`interface.bin` uses magic `SEVPKGI\0`, a format major/minor, producer identity,
and a section directory. Header integers are little-endian; section kinds and
versions are `u32`; offsets and lengths are `u64`. Each section records its
encoding/compression kind and decoded length. Decoders validate bounds, supported
versions and allocation limits before reading indexed records.

The interface has no raw pointers or process-local IDs. Interned string/type
tables and relative offsets support mapped input and lazy decoding. Decoded
objects may allocate; memory mapping does not promise universal zero-copy access.

| Interface record/table | Contents |
| --- | --- |
| `PackageRoot` | Revision, compiler/target identity, references to indexed tables |
| `ExportTable` | Public name/namespace to stable `SymbolRef`, including re-exports |
| `DeclarationRecord` | Kind, visibility, signature/type reference, constraints, effects, implementation reference |
| `TypeRecord` | Type kind, parameters, fields/variants, referenced declarations, representation rules |
| `CallableContract` | Parameter/result types, defaults, generics, constraints, ownership and borrowing relationships |
| `TraitImplementation` | Trait, implementing type, substitutions, constraints and associated members |
| `ImplementationRef` | Native symbol/artifact, generic MIR body, intrinsic, or host macro artifact |
| `AbiLayout` | Target-specific size, alignment, offsets, tags, calling convention and lowered argument/result representation |
| `ForeignExport` | Exported symbol, ABI types, ownership transfers, release function, error and unwind contract |
| `SourceMap` | Source-index references and spans for diagnostics and macro provenance |

Operator/grammar declarations, macros, traits, constraints, and effects needed
across the package boundary MUST survive encoding. Public generics carry checked
implementation material and supporting private declarations; opaque native code
alone cannot instantiate a new type substitution. Unsupported interfaces require
a rebuild from shipped source, never guessed layouts or lost ownership contracts.

Rust, Python and C do not automatically read Severian metadata. Binding tooling
uses the interop records to generate wrappers and selects compatible artifacts.
The initial common boundary is an explicit C ABI: fixed-width values, specified
record layouts, pointer/length pairs and opaque handles with release functions.
Severian memory descriptors require documented ABI lowering or wrapper conversion.

Borrowed results record their relationship to inputs; owned results identify who
releases them and which allocator/runtime owns storage. Error representation and
callback lifetimes are explicit. Unwinding across foreign boundaries is prohibited
unless a supported ABI contract allows it. Python capsules are runtime handles,
not serialized pointers in `interface.bin`. Hash equality is not a memory-safety proof.


## References

- [Rust compiler metadata and libraries](https://rustc-dev-guide.rust-lang.org/backend/libs-and-metadata.html)
- [Cargo build caches and their scope](https://doc.rust-lang.org/cargo/reference/build-cache.html)
- [Rust linkage and foreign library formats](https://doc.rust-lang.org/reference/linkage.html)
- [MLIR bytecode](https://mlir.llvm.org/docs/BytecodeFormat/)
- [MLIR Python bindings and runtime interop](https://mlir.llvm.org/docs/Bindings/Python/)
