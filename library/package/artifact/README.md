# package.artifact

Data models and API contracts for artifacts produced by packages and artifacts
consumed from dependencies. Import this package using an alias such as `artifact`.

`ArtifactKind` describes a payload's role; `ArtifactFormat` describes its encoding.
For example, an ELF file may be an object, dynamic library or executable. A `.lib`
may be a static library or an import library. The producer supplies both fields.

| Payload | Kind | Typical formats | Uses |
| --- | --- | --- | --- |
| `.sev` | Source | SeverianSource | Compile |
| `.sevi` | Interface | SeverianInterface | Import |
| Metadata records | Metadata | Json | Inspect |
| `.o`, `.obj` | Object | Elf, Coff, MachO | Link |
| `.a`, static `.lib` | StaticLibrary | Archive, ThinArchive | Link |
| `.so`, `.dylib`, `.dll` | DynamicLibrary | Elf, MachO, Coff | Link, Load |
| Import `.lib` | ImportLibrary | Archive | Link |
| `.mir`, `.mlir`, `.mlirbc`, `.ll`, `.bc` | IntermediateRepresentation | SeverianMir, MlirText, MlirBytecode, LlvmText, LlvmBitcode | Specialize, Compile |
| `.rlib`, `.rmeta` | StaticLibrary, Metadata | RustLibrary, RustMetadata | Compile, Inspect |
| `.h`, `.rs`, `.pyi` | ForeignBinding | CHeader, RustSource, PythonStub | Import, Compile |
| Executable | Executable | Elf, Coff, MachO, Wasm | Execute |
| Debug companions | DebugInformation | Dwarf, Pdb, Dsym | Inspect |
| Package archive | PackageArchive | SeverianPackage | Package |
| Container payload | ContainerImage | OciImage | Execute, Package |

`ArtifactRef` identifies a package content revision, build and artifact. Metadata
maps declarations to these references. `ArtifactRequest` carries the intended use,
compatibility requirements and optional symbol, entry and specialization identity.
`ArtifactBinding` describes the selected artifact and ordered supporting inputs.
For example, a Windows import library can refer to its runtime DLL through an
`ArtifactRelation`; source, interface, metadata and debug companions use the same
reference model. External SDK and system libraries use `ExternalRequirement`.

`ArtifactOutput` describes a planned output. `ArtifactRecord` describes a completed
artifact with checksummed files. Stored paths are relative to the owning
`package.pkg/`; directory artifacts inventory their files individually.

The public traits are `ArtifactCatalog` (describe and record inventories),
`ArtifactProducer` (produce requested outputs), and `ArtifactConsumer` (bind inputs
for a requested use). These are declarations for future implementations. This
package does not yet implement storage, codecs, artifact production, loading or
adapters for the existing metadata and linker records.
