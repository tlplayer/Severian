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

- `package.toml` defines the package, targets, exports, dependencies, and options.
- `package.lock` pins exact dependency identities and the resolved graph.
- `package.pkg/` is the generated package realization directory.
- `package.pkgi/` is a DIRECTORY containing compiler interfaces and interop files.
- Source MUST ship in every published package, including native provider sources.
- Compatible dependencies MUST reuse metadata and compiled implementations.
- Source availability MUST NOT cause dependency recompilation on every import.
- `library/package` owns these semantics; CLI commands delegate to its APIs.

A package target (library, binary, test, or host macro) is a compilation boundary.
Source modules organize declarations inside that target. Importing a module does
not create another dependency build. Compiler-selected codegen units may produce
multiple objects; they need not correspond one-to-one with source files.

Exports declared with `package.export(foo)`, `with package.export:`, or
`[package].export` MUST populate one canonical export model. Published metadata
includes supporting private declarations needed for public layouts and generics.

## 2. Working package and local build layout

```text
geometry/
├── package.toml
├── package.lock
├── src/
├── native/                              # When this package owns native providers
└── package.pkg/
    ├── package.pkgi/
    │   ├── index.toml                   # Available interfaces and binding sets
    │   ├── severian/<build-id>/interface.bin
    │   ├── c/<abi-id>/geometry.h
    │   ├── rust/<abi-id>/Cargo.toml
    │   ├── rust/<abi-id>/src/lib.rs
    │   └── python/<abi-id>/geometry.pyi
    ├── metadata/
    │   ├── package.toml                 # Snapshot of package definition
    │   ├── package.lock                 # Snapshot of exact resolution
    │   ├── source-index.toml
    │   ├── artifacts.toml
    │   ├── publication.toml             # Written when preparing publication
    │   └── realizations/<build-id>.toml
    ├── artifacts/<target-triple>/<profile>/<build-id>/
    │   ├── object/
    │   ├── archive/
    │   ├── dynamic/
    │   └── ir/
    ├── build/<build-id>/
    │   ├── inputs.toml
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
        ├── package.toml
        ├── package.lock
        ├── src/
        └── native/
```

Only declared binding languages and artifact kinds need entries. Empty optional
directories may be omitted. `<build-id>` distinguishes configurations that have
the same target triple and profile label but different effective inputs.

`source/` is a consistent snapshot of owned sources, headers, resources, and
declared build inputs, preserving relative paths. Its manifests agree with the
metadata snapshots. Generated inputs require either their contents or the source
and declared recipe needed to reproduce them. Dependency sources belong to their
own locked packages; system SDKs and tools are explicit external requirements.

## 3. Published package layout

Publication includes completed payloads and source, excluding local state:

```text
~/.severian/packages/registry/geometry/1.2.0/<content-id>/
├── index.toml                           # Revision/realization discovery
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
            ├── package.toml
            ├── package.lock
            ├── src/
            └── native/
```

`index.toml` entries are installed atomically; existing realization payloads are
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

## 4. File formats and roles

All stored paths are package-relative, normalized UTF-8 paths. Readers reject
absolute paths and traversal outside the package. Linker arguments use structured
records, not executable shell fragments. Symlinks, if supported, remain internal.

| File type | Format and role |
| --- | --- |
| `package.toml`, `package.lock` | TOML package definition and exact dependency graph |
| `package.pkgi/index.toml` | TOML mapping target/build/ABI identities to interfaces and bindings |
| `interface.bin` | Versioned Severian binary metadata with indexed, lazily decoded sections |
| `.h` | Generated C declarations for an explicit exported ABI |
| `.rs`, Rust `Cargo.toml` | Generated Rust wrapper crate using the recorded foreign ABI |
| `.py`, `.pyi` | Python wrappers and type stubs; native extensions remain in `artifacts/dynamic/` |
| `.o`, `.obj` | ELF, Mach-O, or COFF relocatable native objects, according to target |
| `.a`, `.lib` | Native archives; metadata distinguishes static archives from import libraries |
| `.so`, `.dylib`, `.dll` | Platform shared libraries with recorded runtime dependencies |
| `.pyd`, Python-tagged `.so` | Python extension modules with Python ABI/platform requirements |
| `.mir` | Versioned Severian binary MIR for generics/inlining; not Rust MIR |
| `.mlirbc`, `.mlir` | MLIR bytecode and optional textual MLIR; record dialect versions and lowering stage |
| `.bc`, `.ll` | LLVM bitcode and optional textual LLVM IR; record producer and target compatibility |
| `.rlib`, `.rmeta` | Optional Rust-provider artifacts, consumed only by compatible Rust tooling |
| Executable, `.exe` | Native runnable target in `bin/`; platform determines naming |
| `.d`, `.dep` | Compiler dependency information in local `build/dep-info/` |
| `.pdb`, `.dSYM`, `.dwo` | Platform debug information under local `debug/symbols/` |
| `.profraw`, `.profdata`, `.json`, `.lcov` | Local profiling, testing and coverage reports |
| `oci-layout`, `index.json`, OCI blobs | Container image layout with content-addressed layers |

Native archives are standard linker archives containing package-owned objects
and a symbol index. An archive MAY embed an exact copy of `interface.bin`; its
metadata reference identifies the canonical interface artifact, not a new format.
Ordinary archives MUST NOT each copy the complete transitive dependency graph.
Bundled foreign static-library distributions explicitly declare embedded members.

Library emission excludes application `main`. Exported linker symbols incorporate
package/declaration identity; private helpers have correct object-level visibility.
Package initializer symbols are unique, with dependency ordering and once-only
initialization recorded. Relocations are preserved in native objects.

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

`metadata/source-index.toml` owns source `FileRecord`s. `metadata/artifacts.toml`
owns generated payload `ArtifactRecord`s, including interfaces and bindings.
`metadata/realizations/<build-id>.toml` owns the `Realization`, link requirements,
runtime requirements, and entry points. References point to these records;
consumers do not infer dependency identities or ABI contracts from filenames.

System-library requirements distinguish build-time linker names, runtime loader
names, and ABI constraints. CPU features are requirements, not descriptive labels.
Rust, C, Python, LLVM and MLIR provider/toolchain identities are recorded separately.
For cross compilation, host macro/build tools have host realizations and libraries
have target realizations; neither is reused for the other by package name alone.

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

## 7. Versions, identities, and avoiding duplicate hashes

Display provenance as `version:commit:content-id:time`, for example a `1.2.0`
release with an optional Git commit and UTC publication time. Commit and time are
provenance; they MUST NOT force a new build when effective inputs are unchanged.
Different source revisions may retain the same semantic version. Locks pin the
content identity; the registry index chooses the latest published revision only
when resolution intentionally requests an unpinned selection.

All structural hashes use a versioned, domain-separated, length-delimited encoding.
Sort maps, feature sets and path inventories; preserve semantically ordered lists.
Never hash pretty-printed TOML, absolute checkout paths, mtimes or temporary paths
as structural identity. File digests still hash exact file bytes.

| Identity | Computed once from | Excludes |
| --- | --- | --- |
| `ContentId` | Canonical package definition and sorted owned build-input `(relative path, file digest)` entries | Generated outputs, lockfile, commit/time; manifest is not counted again as a source entry |
| Declaration identity | Logical declaration path, kind, stable overload disambiguation | Consumer numbering, source line numbers, unrelated declaration ordering |
| `BuildId` | ContentId, package target name/kind, TargetSpec, effective profile/options/features, compiler/ABI/toolchain identities, exact dependency realizations, declared generated/environment inputs | Output destination, timestamps, its own output digests |
| `ArtifactId` | Completed payload bytes | The inventory that refers to the payload |
| `PublicationId` | Canonical source inventory, selected payload inventory, interface index and realization-record digests | The publication record itself, archive bytes, publication timestamp |

Rules preventing duplicate work, inconsistent hashes and hash cycles:

1. Hash a physical input once per validated snapshot; reuse that result wherever
   the same bytes appear. Manifest copies under `source/` and `metadata/` do not
   create additional semantic inputs. Formatting-only manifest changes may change
   its byte digest without changing the canonical package definition.
2. Source and artifact inventories are the authoritative digest tables. Other
   metadata uses references. `ArtifactId` IS the artifact byte digest: do not add
   parallel `artifact_hash`, `checksum`, and `binary_hash` calculations for it.
3. The interface's ArtifactId identifies its encoded payload. A future semantic
   interface fingerprint is separate only if it intentionally ignores irrelevant
   encoding differences; v1 does not require this extra hash or fine-grained reuse.
4. `package.lock` records exact dependency identities; BuildId includes their
   resolved graph. ContentId excludes the lock so a lock containing its own root
   identity cannot recursively change that identity. Lock text formatting is not
   a new dependency graph.
5. Inventories and realization records do not list themselves as payloads.
   PublicationId is computed after those records are complete. Any transport
   archive digest is recorded externally by the registry, never inside itself.
6. PackageStore interns each exact package/target/build identity once. A diamond
   dependency graph references one realization rather than creating copies per
   importer. GenericInstanceKey deduplicates identical specializations per build
   context; different targets, features or substitutions can need distinct builds.
7. Untrusted mutable inputs require validation. Timestamp/size caches are local
   accelerators, not proof of identity. Installed immutable payloads may reuse
   verified records while their immutability guarantee holds; corruption invalidates
   reuse. Source hashing MUST be distinguished from parsing or recompilation.

## 8. Build, import, link, and prelude reuse

The build planner deduplicates `(revision, target, effective configuration)` units
before invoking the compiler. `package.lock` stabilizes resolution; it is not a
compiled-output cache. Matching semantic versions alone do not establish reuse.

1. Resolve explicit dependency paths or exact locked registry revisions. An
   unpinned standalone import selects the latest compatible published revision.
2. Locate a matching realization and validate required payloads and compatibility.
3. If fresh, skip dependency parsing, semantic analysis and ordinary codegen.
   Load only requested interface records, then select implementation artifacts.
4. Compile/check the consumer against exported type, ownership and effect contracts.
   Generate required generic specializations from encoded implementation material.
5. Link existing dependency objects/archives or record dynamic runtime dependencies.
   Reachability selects required implementations; importing a package is not a build.
6. If a realization is absent, corrupt or incompatible, build the dependency from
   its shipped source once and reuse that realization for subsequent consumers.

`build/<build-id>/inputs.toml` records observed inputs and references the outputs.
Use a per-build lock, private staging, and an atomic successful-completion record
only after output verification. Failure/interruption never marks a build fresh.
Input changes during a build invalidate completion. Concurrent consumers wait for
the same build rather than each launching it. Incremental compiler state can speed
up a necessary rebuild but is distinct from skipping a fresh dependency entirely.

V1 may conservatively invalidate consumers when a dependency realization changes.
It MUST already reuse unchanged dependencies after consumer-only edits, different
output paths, unrelated working directories, and repeated test/build invocations.

The prelude is a published import/export set with exact provider dependencies.
Compiler installation prepares compatible provider realizations. Each invocation
loads their metadata and implementations through PackageStore. The prelude source
ships, but is not flattened into every consumer. A new compatible prelude selection
invalidates affected consumer keys; locked selections stay fixed until updated.

## 9. Registry, publication, and commands

Default storage is `~/.severian/packages/`. `SEVERIAN_HOME` overrides
`~/.severian`; `SEVERIAN_REGISTRY` overrides the registry directory directly.

- `registry/index/<name>.toml`: version/revision/realization discovery and provenance.
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

| Command | Required behavior |
| --- | --- |
| `sev init` | Create package definition with all supported options, defaults and commented alternatives; provide the standard package pipeline |
| `sev build` | Resolve existing locked graph, reuse/build targets, populate working realization |
| `sev check` | Load/produce metadata without requiring native codegen; mark metadata-only results as non-linkable |
| `sev test` | Reuse dependencies, build test targets locally, run with explicit resource/time bounds |
| `sev publish <name> --local` | Validate name, source completeness and selected completed artifacts; exclude build/cache/debug; install atomically |
| `sev add <name>` | Resolve and atomically update manifest/lock without compiling the dependency |
| `sev update <name>` | Intentionally refresh selected dependency resolution; subsequent build evaluates freshness |
| `sev file.sev` | Use explicit dependencies or compatible published packages/prelude from any directory |
| `sev clean` | Remove selected local generated state; preserve authored source and registry payloads |

Export packages MUST NOT contain a metadata-only check result masquerading as a
linkable library. Native artifacts may be omitted for an unsupported target only
when the publication explicitly declares source-rebuild availability for it.

## 10. Acceptance and recovery

Native integration tests MUST build and publish source plus artifacts, move the
producer checkout away, and compile two different consumers in unrelated directories.
Check results, zero ordinary dependency recompiles, and reuse of published prelude
providers. Test generics, diamond dependencies, distinct configurations, source and
lock changes, relocation, damaged outputs, interrupted builds, concurrent consumers,
foreign bindings, and absence of build/cache/debug files from published payloads.

Emit separate events for `source-hash`, `metadata-load`, `parse`, `semantic`,
`specialize`, `codegen`, `cache-hit`, and `link`. File reads alone do not prove a
recompile. Missing interfaces, stale results or timeout failures reject acceptance.
Run baseline and candidate with identical explicit time/memory bounds; record both
absolute limits and allowed regression ratios. Do not silently raise limits to pass.
Install only after the gate passes, retaining the previous compiler, prelude set and
lock identities so a rejected build can be rolled back without rebuilding them.

## References

- [Rust compiler metadata and libraries](https://rustc-dev-guide.rust-lang.org/backend/libs-and-metadata.html)
- [Cargo build caches and their scope](https://doc.rust-lang.org/cargo/reference/build-cache.html)
- [Rust linkage and foreign library formats](https://doc.rust-lang.org/reference/linkage.html)
- [MLIR bytecode](https://mlir.llvm.org/docs/BytecodeFormat/)
- [MLIR Python bindings and runtime interop](https://mlir.llvm.org/docs/Bindings/Python/)
