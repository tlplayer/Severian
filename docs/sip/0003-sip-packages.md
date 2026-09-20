# SIP-0003: Package Interfaces, Realizations, and Incremental Dependency Builds

Status: Draft
Type: Package
Authors: Timothy Player
Created: 2026-09-11
Updated: 2026-09-19
Target: Package system

## Contract and compilation boundary

This SIP specifies required behavior; it does not assert implementation status.
It supersedes source-optional publication and older XDG registry layouts.

- `package.json` defines the package, targets, exports, dependencies, and options.
- `package.lock` pins exact dependency identities and the resolved graph.
- `package.pkg/` is the generated package realization directory.
- `package.pkg/package.pkgi/` is a DIRECTORY containing compiler interfaces in .sevi format
- Source MUST ship in every published package, including native provider sources.
- Compatible dependencies MUST reuse metadata and compiled implementations.
- Source availability MUST NOT cause dependency recompilation on every import.
- `library/package` owns these semantics; CLI commands delegate to its APIs.

A package  (library, binary, test, or host macro) is a compilation boundary.
Source modules organize declarations inside that target. Importing a module does
not create another dependency build. Compiler-selected codegen units may produce
multiple objects; they need not correspond one-to-one with source files.

The package library has the following sub libraries with the following responsibilities
- interface: this file handles .sevi file creation and routing to metadata. It's the surface that all packages communicate through
- metadata: handles every question the compiler has about *.sev/*.sevi dependencies/.o/.so/.a and gets that information from the compiler and retransmits it back underneath the sevi for the package to keep implementation underneath interface
- linker handles linking .o/.so files for dynamic information retrieval and loading pulls from metadata
- dependency: resolving dependencies through the metadata
- archive: handles archives of files 
- profile: handles timing/memory usage to understand how the program behaves over time/input and puts that information in debug
- diagnostic: handles observability into the software bugs, coverage, linting etc.
- test: which handles testing code/artifacts in debug
- build: handles the built artifact outputs
- cache: hot items which would take up a lot of space if left unclean 

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
| `sev clean` | Remove bloat, old builds, cache/* build/* etc.  |

Export packages MUST NOT contain a metadata-only check result masquerading as a
linkable library. Native artifacts may be omitted for an unsupported target only
when the publication explicitly declares source-rebuild availability for it.

##  Working package and local build layout

```text

#Build/ABI Id is in the format YYYY-MM-DD-HH-min-commit(a12c2a30)-hash(last 5 chars)-source-symbol
The ID is used to determine if recompilation is needed/not from a quick glance

geometry/
├── package.json
├── package.lock
├── src/ #Source code
└── package.pkg/
    ├── package.pkgi/
    │   ├── index.json              # Available interfaces and binding sets
    │   ├── severian/<build-id>/lib.sevi 
    │   ├── c/<abi-id>/geometry.h # Optional C interop
    │   ├── rust/<abi-id>/Cargo.toml # Optional Rust interop
    │   ├── rust/<abi-id>/src/lib.rs
    │   └── python/<abi-id>/geometry.pyi # Optional Python interop
    ├── metadata/
    │   ├── realizations/<build-id>.json
    │   ├── symbols/<build-id>.json
    │   ├── dependencies/<build-id>.json
    │   └── layouts/<abi-id>.json
    ├── artifacts/<target-triple>/<build-id>/
    │   ├── object/
    │   ├── archive/
    │   ├── dynamic/
    │   └── ir/
    ├── build/<build-id>/<targets> #Release artifacts for targets
    ├── cache/ #temporary artifacts used for hot operations
    ├── bin/<target-triple>/<build-id>/
    ├── debug/
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

##  File formats and roles

All stored paths are package-relative, normalized UTF-8 paths. Readers reject
absolute paths and traversal outside the package. Linker arguments use structured
records, not executable shell fragments. Symlinks, if supported, remain internal.

```sh
interface.sevi       REQUIRED
    compiler-facing public semantic interface

.o            BUILD PRODUCT
    native codegen units

.a/.lib            DEFAULT REUSABLE STATIC IMPLEMENTATION
    archive of native codegen units

.so/.dylib/.dll       DYNAMIC IMPLEMENTATION
    

.mlirbc               SPECIALIZATION IMPLEMENTATION
    generics/JIT/target specialization where native code isn't sufficient

executable           BINARY TARGET
    output of main.sev
```


##  Published package layout

Publication includes completed payloads and source, excluding local state:

```text
~/.severian/packages/registry/geometry/1.2.0/<content-id>/
├── index.json                           # Revision/realization discovery
└──<publication-id>/
    └── package.pkg/
        ├── package.pkgi/                # Same interface/binding layout as above
        ├── metadata/                   # Snapshots, inventories, link/run records├── symbols/
            └── <symbol-id>.json
            ├── types/
            │   └── <type-id>.json
            ├── layouts/
            │   └── <abi-id>.json
            ├── dependencies/
            │   └── <build-id>.json
            └── realizations/ #... add more here as needed
    └── <build-id>.json
        ├── artifacts/<target-triple>/<build-id>/
        │   ├── object/
        │   ├── archive/
        │   ├── dynamic/
        │   └── ir/
        ├── bin/<target-triple>/<build-id>/
        ├── container/<target-triple>/<build-id>/
        └── source/
            ├── package.json
            ├── package.lock
            ├── src/
```
## Diagram

caller.sev
   ↓
resolve geometry.foobar
   ↓
lib.sevi
   ↓
metadata/symbols/geometry.foobar
   ↓
select realization
   ↓
artifacts/x86_64.../<build-id>/object/geometry.o
   ↓
caller.o + geometry.o
   ↓
link

                         implementation
                        /              \
interface -> metadata -> .o/.so       MLIR
                        |               |
                        |           specialize
                        |               |
                        +-------> .o <--+
                                   |
                                  link


## Test

The test for this process is as follows:

### lib.sev

The interface is what consumers use and what the pkgi/ folder contains
```sev
trait FooBar
    def foobar(a:T,b:T) -> int
```

### mod.sev implements the interface 
```
class FooBarClass: Foobar
    def foobar(a:T,b:T) -> int: FooBar
        return a+b
```

This should result in monomorphized implementations of the polymorphic types



## References

- [Rust compiler metadata and libraries](https://rustc-dev-guide.rust-lang.org/backend/libs-and-metadata.html)
- [Cargo build caches and their scope](https://doc.rust-lang.org/cargo/reference/build-cache.html)
- [Rust linkage and foreign library formats](https://doc.rust-lang.org/reference/linkage.html)
- [MLIR bytecode](https://mlir.llvm.org/docs/BytecodeFormat/)
- [MLIR Python bindings and runtime interop](https://mlir.llvm.org/docs/Bindings/Python/)
