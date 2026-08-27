# Building and packaging

This directory is the executable specification for Severian packages. A change
to manifest discovery, dependency resolution, build output placement, or the
`.pkg` format should update this example and its tests in the same change.

The design keeps four things separate:

| Object | Meaning | Authored by |
| --- | --- | --- |
| `package.toml` | Desired package, targets, dependencies, and policy | Developer |
| `package.lock` | Exact dependency resolution (format fixture today) | `sev` |
| `target/` | Local build cache and development outputs | `sev build` / `sev test` |
| `.pkg` | Versioned distributable realization of a library | `sev build` |

`target/` is disposable. A `.pkg` is the versioned distribution boundary; the
current compiler writes version 1 library bundles but does not consume them yet.
Neither object is a substitute for the manifest or lockfile.

## Vocabulary

- A **package** is the unit described by one `package.toml`.
- A **declared target** is something the package produces: `[lib]` or `[[bin]]`.
- A **module** is one `.sev` source file in a package.
- An **interface** is the public identity and type information consumers need.
- An **implementation** supplies bodies for an interface.
- A **platform** is a compilation destination such as `x86_64-linux-gnu`.
- An **artifact** is one realization of a declared target for a platform and
  profile.

Using “target” for both a binary and a platform makes resolution ambiguous, so
manifests and metadata retain this distinction.

## Checked-in source layout

```text
05-building/
├── README.md
├── package.toml
├── package.lock
└── src/
    ├── lib.sev
    ├── math.sev
    └── main.sev
```

The library root imports its private modules. The binary imports the library
root. A dependency imports another package through the alias declared in
`[dependencies]`; that alias is also the namespace used by qualified names such
as `geometry.Point`.

## Manifest contract

The checked-in [package.toml](package.toml) intentionally uses the complete
configuration vocabulary currently understood by `sev`. Unknown compiler
configuration keys are rejected. `sev config defaults` prints the catalog and
`sev config sync` adds newly introduced settings without replacing explicit
choices.

The structural part is:

```toml
[package]
name = "05-building"
version = "0.1.0"
edition = "2026"
default-run = "05-building"

[[bin]]
name = "05-building"
path = "src/main.sev"

[lib]
name = "building"
path = "src/lib.sev"

[dependencies]
geometry = "0.3.1"
# shapes = { package = "geometry", version = "0.3.1" }

[dev-dependencies]
test-support = "0.1.0"
```

Rules:

1. Package names identify published packages; dependency keys are local aliases.
2. Every `[[bin]]` has an explicit unique name and source path.
3. `[lib]` has one public module root. Its name defaults to the package name.
4. `package.default-run` is required when more than one binary exists.
5. A dependency must expose `[lib]`. Binaries are not importable interfaces.
6. Version dependencies resolve by package identity from the selected registry;
   a dependency key is only its local import alias. Path dependencies are an
   explicit development override and resolve relative to their manifest.
7. Runtime builds use `[dependencies]`; root-package tests may additionally use
   `[dev-dependencies]`.
8. Source paths and package archive entries may not escape their package root.

Exact-version dependencies may resolve from the default local filesystem
registry (or one selected by `SEVERIAN_REGISTRY`). `sev publish` writes both the
versioned `.pkg` artifact and its source realization there. Remote registry
transport, Git dependencies, ranges, and authentication remain explicit
errors; the resolver never silently selects unrelated source.

The black-box golden path is
[`registry_publish_consume.sh`](../../test/validation/packages/registry_publish_consume.sh).
It creates a library with `sev new`, publishes it to an isolated registry,
creates an unrelated application with `sev new`, consumes the library using
only registry package declarations, and builds and runs the application. The
CLI contract suite executes the same script.

Transitive package closure is covered by
[`registry_transitive_tensor.sh`](../../test/validation/packages/registry_transitive_tensor.sh).
It publishes `tensor`, a matrix package that performs tensor matmul, and a
service package that calls the matrix package. A third application declares
only the service. The resolver discovers the service's published dependency
edges, while still rejecting a direct import of the undeclared matrix package.
The test also removes one transitive release temporarily and requires the
diagnostic to report the complete application-to-service-to-matrix chain.

## Lockfile contract

Lockfile generation and consumption are not implemented yet. The checked-in
`package.lock` is therefore a format fixture, not an input used by today’s
resolver. Once implemented, it is tool-owned generated data and should be
committed for applications. It records exact package identities, versions,
sources, revisions, checksums, features, and dependency edges. It does not
record the build machine, current profile, output path, credentials, or
environment variables.

Resolution must be deterministic:

- entries have one canonical record per package identity;
- dependency edges refer to locked identities, not loose names;
- checksums cover fetched package content;
- path dependencies are canonicalized before cycle detection;
- normal builds do not rewrite a lockfile whose resolution is unchanged.

This example has no external dependencies, so its lockfile contains only the
root package. Invented registry entries do not belong in an executable example.

## Commands and local output

From this directory:

```bash
sev check
sev test
sev build
sev run
sev run --bin 05-building
sev build --profile release
sev check --emit mir --bin 05-building
```

The current local output layout is:

```text
target/
└── <platform>/
    └── <profile>/
        ├── bin/
        │   └── 05-building
        ├── pkg/
        │   └── building-0.1.0.pkg
        └── tests/
            └── run-<invocation>/
```

The platform is `host` unless overridden by `build.target` or `--target`; the
profile is `dev` unless overridden by `build.profile` or `--profile`.

## `.pkg` compatibility boundary

The current `SEVPKG` version 1 writer emits a deterministic library source
bundle containing the library name and every reachable module owned by that
package. Executable artifacts remain in `target/<platform>/<profile>/bin`.
This small writer is implemented today. Its magic, version, and byte-level test
fixtures define the compatibility contract for the future reader.

The intended next archive version is a logical container with these sections:

```text
building.pkg/
├── metadata/
│   ├── package.toml       # frozen source manifest
│   ├── package.lock       # exact dependency graph
│   ├── build.toml         # compiler, profile, and reproducibility data
│   ├── artifacts.toml     # artifact index and compatibility requirements
│   └── checksums.toml     # digest for every indexed object
├── interface/
│   └── building.pkgi      # public declarations and stable identities
├── source/                # optional; controlled by publish.include-source
│   ├── lib.sev
│   └── math.sev
├── artifacts/
│   └── <platform>/
│       └── <profile>/
│           ├── native/
│           ├── llvm/
│           ├── mlir/
│           └── stablehlo/
├── evidence/              # optional test, coverage, profile, and debug data
└── policy/                # optional runtime requirements
    ├── network.toml
    └── container.toml
```

This is a logical layout; an implementation may use a binary index rather than
a ZIP filesystem. Archive paths are normalized UTF-8, relative,
slash-separated, sorted before encoding, and forbidden from containing `..`,
absolute roots, or escaping symlinks. Checksums cover canonical bytes, not
extraction metadata.

The interface is first-class because consumers should not need implementation
source merely to type-check imports. Source inclusion and interface inclusion
are separate publication choices. A package that omits source must include a
compatible artifact for every platform it claims to support.

## Artifact selection

Running or consuming a `.pkg` follows one deterministic order:

```text
1. Select the declared target by name and kind.
2. Select an exact compatible native artifact for the requested platform.
3. Otherwise select a compatible compiler/backend artifact.
4. Otherwise rebuild from included source using the included lockfile.
5. Otherwise use an explicitly permitted container recipe or embedded OCI image.
6. Otherwise report each rejected candidate and the missing requirement.
```

Selection considers the platform triple, Severian runtime ABI, backend format
version, CPU/GPU features, required system libraries, and package capabilities.
It must not silently run an incompatible artifact or weaken package policy.

## Runtime policy and containers

The `network` library describes how source performs I/O. Package network policy
describes what a particular executable needs from its environment. It belongs
beside container policy because native processes, VMs, remote jobs, and
containers all share the same requirements.

Network policy may declare named ingress and egress endpoints, DNS/TLS/proxy
requirements, timeouts, retry policy, and resource limits. Container policy has
three explicit modes:

```toml
[container]
mode = "none"       # no fallback
# mode = "recipe"   # construction metadata is present
# mode = "embedded" # an OCI image is present in the package
```

Package installation never gains host access merely because a fallback exists.
Native tools, network access, filesystem mounts, devices, and credentials remain
declared capabilities. A force option may acknowledge a compatibility or trust
warning, but it does not silently bypass the program's safety model.

## Versioning invariants

- Manifest additions are backward-compatible only when old readers can ignore
  them safely; otherwise the manifest format needs an explicit version.
- `.pkg` and `.pkgi` carry independent format versions.
- Public declaration identities do not depend on source order or build paths.
- Package archives do not contain absolute host paths or build credentials.
- Reproducible builds keep timestamps and machine observations out of hashed
  artifact identity.
- Readers reject unknown mandatory capabilities and malformed indexes before
  extracting or executing content.

## Implementation status

| Capability | Status |
| --- | --- |
| Manifest discovery and local path dependency graph | Implemented |
| Binary and library targets | Implemented |
| Dev dependencies for root tests | Implemented |
| Catalog-backed configuration and profile overlays | Implemented |
| Local target layout shown above | Implemented |
| `SEVPKG` v1 reachable-source library writer | Implemented |
| Consuming an emitted `.pkg` as a dependency | Not implemented |
| `sev publish` and exact-version local registry source consumption | Implemented |
| Registry publish/consume golden-path validation | Implemented |
| Published transitive dependency closure and import isolation | Implemented |
| General package interfaces (`.pkgi`) | Partial; primitive interface records exist |
| Remote registry/Git resolution and lockfile generation | Reserved, not implemented |
| Rich indexed `.pkg` archive and artifact selection | Design contract |
| Network/container policy enforcement | Design contract |

Keeping this table honest is part of the example. Documentation must not claim
that a future distribution feature is produced by today’s `sev build`.
